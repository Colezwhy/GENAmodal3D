import torch,random
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as L
from torchvision import transforms
import gena3d.models as models
from gena3d.models import SparseStructureFlowModelMaskAsCondWeighted, SparseStructureFlowMultiBranchModel
from .data_utils import img_pil_to_tensor, mask_pil_to_tensor, filter_by_depth, voxelize, write_ply
from dit.diffusion_ss import GaussianDiffusion, get_betas
from safetensors.torch import load_file
from dependency.Pi3.pi3.models.pi3 import Pi3
from dependency.Pi3.pi3.utils.geometry import depth_edge
import uuid
import torch, os

# 确保缓存目录固定在 home 下
os.environ["TORCH_HOME"] = os.path.expanduser("scratch/junwei/.cache/torch")
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("scratch/junwei/.cache")

# patching masks,
class mask_patcher(nn.Module):
    def __init__(self):
        super(mask_patcher, self).__init__()

    def forward(self, mask, patch_size=14):
        """
        Inputs:
            mask: tensor, size [B, H, W] (e.g., the original mask might not be 518x518)
            patch_size: the size of each patch, default is 14 (since 518/14=37)
        Outputs:
            patch_ratio: tensor, size [B, 37, 37], where each element represents the ratio of 1's in the corresponding patch of the mask
        """
        # 1. Resize the mask if the shape is not (518, 518) 
        B, N, H, W = mask.shape
        if (H, W) != (518, 518): 
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(B, N, 518, 518), mode='nearest')  # [B, 1, 518, 518]
        else:
            mask = mask.float()
        
        # 2. Use unfold to divide the mask into non-overlapping patches
        # Unfold parameters: dimension, window size, stride
        patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # At this point, patches have shape [B, 1, 37, 37, 14, 14]
        
        # 3. For each patch, compute the ratio of ones in the mask by taking the mean
        patch_ratio = patches.mean(dim=(-1, -2))  # Shape becomes [B, 1, 37, 37]
        
        # 4. Remove the channel dimension
        # patch_ratio = patch_ratio.squeeze(1)  # Final shape is [B, 37, 37]
        
        return patch_ratio

class Network_ss(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True, full_train=False):
        super(Network_ss, self).__init__()

        self.cfg = cfg
        
        self.ss_model = SparseStructureFlowMultiBranchModel(resolution=16, 
                                                        in_channels=8, 
                                                        out_channels=8, 
                                                        model_channels=1024, 
                                                        cond_channels=1024, 
                                                        num_blocks=24, 
                                                        num_heads=16, 
                                                        mlp_ratio=4, 
                                                        patch_size=1, 
                                                        pe_mode='ape', 
                                                        qk_rms_norm=True, 
                                                        use_fp16=False, 
                                                        use_checkpoint=True, 
                                                        use_gate=cfg.use_gate)

        # diffusion model
        betas = get_betas(schedule_type='linear', b_start=0.0001, b_end=0.02, time_num=1000)
        self.diffusion = GaussianDiffusion(betas, loss_type='mse', model_mean_type='eps', model_var_type='fixedsmall')
        # ============== Loading image encoder models... ============== #
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
        # To use a local checkout of dinov2 instead of downloading from torch.hub, use:
        # dinov2_model = torch.hub.load('./dinov2', 'dinov2_vitl14_reg', source='local', pretrained=True)
        dinov2_model.eval()
        for param in dinov2_model.parameters():
            param.requires_grad = False
        self.img_encoder = dinov2_model
        # ============== Loading image encoder models... ============== #
        
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        self.img_encoder_feat_dim = 1024

        # ============== loading stereo models and voxel encoding model. =============== #
        self.stereo_model = Pi3.from_pretrained('yyfz233/Pi3').eval() 
        self.voxel_encoder = models.from_pretrained('microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16').eval()
        # if loading to self.device? Here we do not have self.device and only load onto 'cuda'.
        for param in self.stereo_model.parameters():
            param.requires_grad = False
        for param in self.voxel_encoder.parameters():
            param.requires_grad = False
        self.mask_patcher = mask_patcher()

        self.load_pretrained_ss()
        if not full_train:
            self.freeze_except_last_cross_attn()

    def on_train_epoch_start(self):
        dataset = self.trainer.train_dataloader.dataset  # 直接取出当前dataset
        new_n = random.choice([1, 2, 3, 5])
        dataset.update_views(new_n)
        print(f'[INFO]: updated training sparse views {new_n}!')

    
    def freeze_except_last_cross_attn(self):
        for name, param in self.ss_model.named_parameters(): # training the new cross attn, vo
            if "cross_attn_voxel" in name:
                param.requires_grad = True
                print(f"Trainable: {name}")
            else:
                param.requires_grad = False
                print(f"Frozen: {name}")

    def load_pretrained_ss(self):
        ss_model_weight = load_file("./ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors")
        self.ss_model.load_state_dict(ss_model_weight, strict=False)
        print("[INFO]: loaded Sparse Structure model!")
    
    
    @torch.no_grad()
    def stereo_pc_gen(self, scene_image, scene_mask = None):
        """
        Run stereo model with sparse input images. And obtain the mask-aligned point cloud.
        input: N, H, W, C
        Currently using PI3 as default..
        """
        # 1. Prepare input data
        # The load_images_as_tensor function will print the loading path
        imgs = scene_image.permute(0, 3, 1, 2) # (N, 3, H, W)
        # 2. Infer
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.stereo_model(imgs[None]) # Add batch dimension
                
        # Optional: loading mask images.
        if scene_mask is not None:
            # should already be tensor dtype
            # 4. process mask @@ in res: points:(1, N, H, W, 3), totally these points...
            # masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
            # non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03) 
            # masks = torch.logical_and(masks, non_edge)[0]
            final_mask = scene_mask > 0.5
            if final_mask.sum() == 0:
                pos, col = res['points'][0], imgs.permute(0, 2, 3, 1) 
                # print('[INFO]: unstable sample in training, no confident point cloud generated.')
                # currently discard this during training..
            else:
                pos, col = res['points'][0][final_mask], imgs.permute(0, 2, 3, 1)[final_mask]
        else:
            masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
            non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
            masks = torch.logical_and(masks, non_edge)[0]
            pos, col = res['points'][0][masks], imgs.permute(0, 2, 3, 1)[masks]
        return [pos, col]
    

    @torch.no_grad()
    def encode_pc(self, scene_image, mask=None, resolution: int = 64) -> torch.Tensor:
        """
        Get the conditioning information of possible occluded regions.
        """
        feats = []
        for i, img in enumerate(scene_image):
            pos, col = self.stereo_pc_gen(img, mask[i])
            # filename = f"/scratch/junwei/voxels_during_training/pointcloud_{uuid.uuid4().hex[:8]}.ply"
            # # for test input evaluation ..
            # write_ply(pos.cpu(), col, filename)
            voxel = voxelize(pos) # only incorporate position during training. for alignment of encode ss latent?
            coords = ((torch.tensor(voxel) + 0.5) * resolution).int().contiguous()
            ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
            ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
            ss = ss[None].to(self.device).float() # should be on to the gpu assigned in this class..
            feat_voxel = self.vae_encode(ss)
            feat_voxel = feat_voxel[0] # voxel features, for later condition.
            feats.append(feat_voxel)
        feats_voxel = torch.stack(feats, dim=0) # (B, dim_feat_voxel) # 
        return feats_voxel
    
    @torch.no_grad()
    def vae_encode(self, ss):
        """
        Encode the voxels with VAE encoder.
        serving as conditions.
        """
        latent = self.voxel_encoder(ss, sample_posterior=False) # [1, 8, 16, 16, 16]
        assert torch.isfinite(latent).all(), "Non-finite latent"
        return latent
    
    @torch.no_grad()
    def encode_image(self, inp_imgs):
        """
        Input images already torch.tensor dtype, (B, K, H, W, C) Batch, n_views, H, W, C
        """
        B, K, H, W, C = inp_imgs.shape
        inp_imgs = inp_imgs.permute(0, 1, 4, 2, 3) # B K C H W
        inp_imgs = torch.stack([self.image_cond_model_transform(inp_imgs[i]) for i in range(B)], dim=0)
        inp_imgs = inp_imgs.reshape(B*K, C, H, W)
        encode_imgs = self.img_encoder(inp_imgs, is_training=True)['x_prenorm']
        BK, N, D = encode_imgs.shape
        encode_imgs = encode_imgs.reshape(B, K, N, D)
        # batch manipulation, identical layernorm.
        encode_imgs = F.layer_norm(encode_imgs, (D,))
        # batch manipulation, identical layernorm.
        # encode_imgs = F.layer_norm(encode_imgs, encode_imgs.shape[-1:])
        return encode_imgs

    def logit_normal(self, std, mean, size, device):
        # Generate samples from logit-normal distribution
        normal_samples = torch.randn(size, device=device) * std + mean
        logit_samples = torch.sigmoid(normal_samples)  # Apply logit transformation
        return logit_samples

    def _denoise(self, x, t, c):
        t = t * 1000
        out = self.ss_model(x, t, c)
        return out

#TODO@: here mainly working on this to load features...
    def forward(self, batch, if_vis=False):
        orig_img = batch['orig_img'] # of no use during training..
        cond_inpaint = batch['cond_inpaint']
        occlusion_mask = batch['occlusion_mask']
        occluded = batch['occluded']
        visibility_mask = batch['visibility']
        z_s = batch['ss_latent']
        B,_,_,_,_ = z_s.shape
        B, N, H, W, C = cond_inpaint.shape  

        V3 = 16 ** 3
        # extract image features, droprate 10%
        random_num = random.random()
        if 0.1 < random_num:
            with torch.no_grad():
                # diverse images... B, N, H, W, C 
                # the first dimension should not be manipulated
                # should be batch operation
                encoded_cond_img = self.encode_image(cond_inpaint) # (B, N, k, D) D is 1024
                encoded_cond_img = encoded_cond_img.reshape(B, 1374 * N, 1024)
                pc_feat = self.encode_pc(occluded, visibility_mask) # [B, 8, 16, 16, 16]
                B, C_, V, V, V = pc_feat.shape
                V3 = V ** 3
                vox_flat = pc_feat.view(B, C_, V3).permute(0, 2, 1)  # [B, V3, 8]
                vox_aligned = vox_flat.repeat(1, 1, 1024 // C_)
                # TODO@:how to deal with pc_feat, check out the pc_feat dimensions
                
                # patched_visibility_mask = self.mask_patcher(visibility_mask)
                # visibility_mask_token = patched_visibility_mask.view(B, -1).unsqueeze(-1).repeat(1, 1, 1024)
                # patched_occlusion_mask = self.mask_patcher(occlusion_mask)
                
                # occlusion_mask_token = patched_occlusion_mask.view(B, -1).unsqueeze(-1).repeat(1, 1, 1024)
                
                # print(encoded_cond_img.shape, occlusion_mask_token.shape, vox_aligned.shape)
                cond_img = torch.cat([encoded_cond_img, vox_aligned], dim=1) #  occlusion_mask_token,
        else:
            cond_img = torch.zeros((B, 1374 * N + V3, 1024)).to(cond_inpaint.device).to(cond_inpaint.dtype) # 37*37*N + 

        t = self.logit_normal(1.0, 1.0, (B,), device="cuda") # follow trellis training details
        
        loss, _ = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=z_s, t=t, y=cond_img)
        loss = loss.mean()

        return None, loss
