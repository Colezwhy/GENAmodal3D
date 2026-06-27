from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder
import math


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h


class SparseMultiHeadAttentionWeighted(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None, mask_weight: Optional[Union[SparseTensor, torch.Tensor]] = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            if mask_weight is not None:
                h = self.sparse_scaled_dot_product_attention_weighted(q, kv, mask_weight)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h

    def sparse_scaled_dot_product_attention_weighted(self, q, kv, mask_weight, eps=1e-6):
        s = q
        q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
        q = q.feats # [T_Q, H, C]
        T_q, H_q, C_q = q.shape
        
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        N, L, _, H_kv, C_kv = kv.shape
        kv_seqlen = [L] * N
        kv = kv.reshape(N * L, 2, H_kv, C_kv)   # [T_KV, 2, H, C]
        k, v = kv.unbind(dim=1) # [T_KV, H, C]

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # handle cls for mask_patcher
        B, T_mask, _ = mask_weight.shape
        cls_weight = torch.ones(B, 5, device=mask_weight.device, dtype=mask_weight.dtype)
        mask_weight = torch.cat([cls_weight, mask_weight.squeeze(2)], dim=1)
        mask_weight = mask_weight.reshape(B*(T_mask+5)).unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, T_KV, 1, 1]
        mask_bias = torch.log(mask_weight + eps)  # [1, T_KV, 1, 1]

        q = q.permute(0, 2, 1, 3) # [1, num_heads, T_Q, head_dim]
        k = k.permute(0, 2, 1, 3) # [1, num_heads, T_KV, head_dim]
        v = v.permute(0, 2, 1, 3) # [1, num_heads, T_KV, head_dim]



        attn_logits = q @ k.transpose(-2, -1) / math.sqrt(C_kv)
        
        attn_logits += mask_bias
        attn_weights = F.softmax(attn_logits, dim=-1)
        output = attn_weights @ v
        output = output.permute(0, 2, 1, 3).squeeze(0)

        return s.replace(output)
