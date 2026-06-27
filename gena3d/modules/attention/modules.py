from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
import math


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h


class MultiHeadAttentionWeighted(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask_weight: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                if mask_weight is not None:
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    h = self.weighted_scaled_dot_product_attention(q, k, v, mask_weight)
                else:
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
            else:
                if mask_weight is not None:
                    k, v = kv.unbind(dim=2)
                    h = self.weighted_scaled_dot_product_attention(q, k, v, mask_weight)
                else:
                    h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

    def weighted_scaled_dot_product_attention(self, q, k, v, mask_weight, eps=1e-6):
        # q: (B, L_q, num_heads, head_dim)
        # k: (B, L_k, num_heads, head_dim)
        # v: (B, L_k, num_heads, head_dim)
        # mask_weight: (B, L_k, 1)
        assert mask_weight.shape[1] == k.shape[1] - 5
        d = k.size(-1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_logits = q @ k.transpose(-2, -1) / math.sqrt(d) # (B, num_heads, L_q, L_k)

        B, L_k, _ = mask_weight.shape
        cls_weight = torch.ones(B, 5, device=mask_weight.device, dtype=mask_weight.dtype)
        mask_weight = torch.cat([cls_weight, mask_weight.squeeze(2)], dim=1) # (B, L_k + 5)
        mask_bias = torch.log(mask_weight.unsqueeze(1).unsqueeze(1) + eps)
        attn_logits = attn_logits + mask_bias

        attn = torch.softmax(attn_logits, dim=-1)
        output = attn @ v
        return output.permute(0, 2, 1, 3) # (B, L_q, num_heads, head_dim)
    
    
class MultiHeadAttentionGeo(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        if_gating: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        
        self.gating = if_gating
        # ========= Gating attention ========= #
        if self.gating:
            # gating net for preserving voxel geometry.
            self.gating_mlp = nn.Sequential(
                nn.Linear(self.ctx_channels, self.ctx_channels // 2),
                nn.ReLU(),
                nn.Linear(self.ctx_channels // 2, 1),
                nn.Sigmoid()
            )
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask_weight: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                # if mask_weight is not None:
                #     q = self.q_rms_norm(q)
                #     k, v = kv.unbind(dim=2)
                #     k = self.k_rms_norm(k)
                #     h = self.weighted_scaled_dot_product_attention(q, k, v, mask_weight)
                # else:
                if self.gating and context is not None:
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    voxel_gate = self.gating_mlp(context)  # (B, L_k, 1)
                    h = self.voxel_gated_scaled_dot_product_attention(q, k, v, voxel_gate)
                else:
                    q = self.q_rms_norm(q)
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    h = self.scaled_dot_product_attention(q, k, v)
            else:
                # if mask_weight is not None:
                #     k, v = kv.unbind(dim=2)
                #     h = self.weighted_scaled_dot_product_attention(q, k, v, mask_weight)
                # else:
                if self.gating and context is not None:
                    k, v = kv.unbind(dim=2)
                    voxel_gate = self.gating_mlp(context)  # (B, L_k, 1)
                    h = self.voxel_gated_scaled_dot_product_attention(q, k, v, voxel_gate)
                else:
                    h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

    def voxel_gated_scaled_dot_product_attention(self, q, k, v, voxel_gate=None, eps=1e-6):
        """
        Voxel-gated Scaled Dot-Product Attention
        Args:
            q: (B, L_q, num_heads, head_dim)
            k: (B, L_k, num_heads, head_dim)
            v: (B, L_k, num_heads, head_dim)
            voxel_gate: (B, L_k, 1)  # gating weight derived from voxel feature
            eps: small constant for numerical stability
        Returns:
            output: (B, L_q, num_heads, head_dim)
        """
        B, L_q, num_heads, head_dim = q.shape
        L_k = k.shape[1]

        # 转换维度到 (B, num_heads, L, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, num_heads, L_q, L_k)

        # --- Voxel gating ---
        # voxel_gate: (B, L_k, 1) -> (B, 1, 1, L_k) for broadcasting
        geom = voxel_gate.clamp_min(eps)                 # (B, L_k, 1)
        geom = geom.squeeze(-1)                          # (B, L_k)
        geom_bias = torch.log(geom)                      # (B, L_k)
        geom_bias = geom_bias.unsqueeze(1).unsqueeze(1)

        attn_logits.add_(geom_bias)

        # Softmax
        attn = torch.softmax(attn_logits, dim=-1)

        # value out
        output = torch.matmul(attn, v)  # (B, num_heads, L_q, head_dim)

        return output.permute(0, 2, 1, 3)