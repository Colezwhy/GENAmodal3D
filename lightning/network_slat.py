import torch,random
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import pytorch_lightning as L
from torchvision import transforms

from gena3d.models import SLatFlowModelMaskAsCondWeighted

from dit.diffusion_slat import GaussianDiffusion, get_betas

from safetensors.torch import load_file

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
        if mask.shape != (518, 518): 
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(518, 518), mode='nearest')  # [B, 1, 518, 518]
        else:
            mask = mask.unsqueeze(1).float()
        
        # 2. Use unfold to divide the mask into non-overlapping patches
        # Unfold parameters: dimension, window size, stride
        patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # At this point, patches have shape [B, 1, 37, 37, 14, 14]
        
        # 3. For each patch, compute the ratio of ones in the mask by taking the mean
        patch_ratio = patches.mean(dim=(-1, -2))  # Shape becomes [B, 1, 37, 37]
        
        # 4. Remove the channel dimension
        patch_ratio = patch_ratio.squeeze(1)  # Final shape is [B, 37, 37]
        
        return patch_ratio


class Network_slat(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network_slat, self).__init__()

        self.cfg = cfg
        
        self.slat_model = SLatFlowModelMaskAsCondWeighted(resolution=64, 
                                                in_channels=8, 
                                                out_channels=8, 
                                                model_channels=1024, 
                                                cond_channels=1024, 
                                                num_blocks=24, 
                                                num_heads=16, 
                                                mlp_ratio=4, 
                                                patch_size=2, 
                                                num_io_res_blocks=2, 
                                                io_block_channels=[128], 
                                                pe_mode='ape', 
                                                qk_rms_norm=True, 
                                                use_fp16=False, 
                                                use_checkpoint=True,
                                                mask_cond_type='mask_patcher'
                                                )
        
        for _, param in self.slat_model.named_parameters():
            param.requires_grad = True

        # diffusion model
        betas = get_betas(schedule_type='linear', b_start=0.0001, b_end=0.02, time_num=1000)
        self.diffusion = GaussianDiffusion(betas, loss_type='mse', model_mean_type='eps', model_var_type='fixedsmall')

        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
        dinov2_model.eval()
        for param in dinov2_model.parameters():
            param.requires_grad = False
        self.img_encoder = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        self.img_encoder_feat_dim = 1024

        self.std = torch.tensor([2.377650737762451,2.386378288269043,2.124418020248413,2.1748552322387695,2.663944721221924,2.371192216873169,2.6217446327209473,2.684523105621338])[None]
        self.mean = torch.tensor([-2.1687545776367188, -0.004347046371549368,-0.13352349400520325,-0.08418072760105133,-0.5271206498146057,0.7238689064979553,-1.1414450407028198,1.2039363384246826])[None]

        self.mask_patcher = mask_patcher()

        self.load_pretrained_slat()

    def load_pretrained_slat(self):
        slat_model_weight = load_file("./ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors")
        self.slat_model.load_state_dict(slat_model_weight, strict=False)
        print("load slat model")
        
    def encode_image(self, inp_imgs):
        B, _, _, _ = inp_imgs.shape
        inp_imgs = inp_imgs.permute(0, 3, 1, 2)
        inp_imgs = torch.stack([self.image_cond_model_transform(inp_imgs[i]) for i in range(B)], dim=0)
        encode_imgs = self.img_encoder(inp_imgs, is_training=True)['x_prenorm']#['x_norm_patchtokens'] # [B*N, 37*37, 1024]
        encode_imgs = F.layer_norm(encode_imgs, encode_imgs.shape[-1:])
        return encode_imgs

    def logit_normal(self, std, mean, size, device):
        # Generate samples from logit-normal distribution
        normal_samples = torch.randn(size, device=device) * std + mean
        logit_samples = torch.sigmoid(normal_samples)  # Apply logit transformation
        return logit_samples

    def _denoise(self, x, t, c):
        t = t * 1000
        out = self.slat_model(x, t, c)
        return out

    def forward(self, batch, if_vis=False):
        cond_img = batch['cond_img_518_masked']
        visibility_mask = batch['visibility_mask']
        occlusion_mask = batch['occlusion_mask']
        slat_paths = batch['slat_path']
        B = len(slat_paths)
        sp_feats, sp_coords = [], []
        for idx, slat_path in enumerate(slat_paths):
            slat_data = np.load(slat_path)
            coords = torch.from_numpy(slat_data['coords']).int().to(self.device)
            feats = torch.from_numpy(slat_data['feats']).float().to(self.device)
            feats = (feats - self.mean.to(self.device)) / self.std.to(self.device)
            coords = torch.cat([idx*torch.ones_like(coords[:, 0]).unsqueeze(1), coords], dim=1)
            sp_feats.append(feats)
            sp_coords.append(coords)

        # extract image features
        if random.random() > 0.1:
            with torch.no_grad():
                encoded_cond_img = self.encode_image(cond_img)
                visibility_mask = self.mask_patcher(1-visibility_mask)
                visibility_mask = visibility_mask.view(B, -1).unsqueeze(-1).repeat(1, 1, 1024)
                occlusion_mask = self.mask_patcher(occlusion_mask)
                occlusion_mask = occlusion_mask.view(B, -1).unsqueeze(-1).repeat(1, 1, 1024)
                cond_img = torch.cat([encoded_cond_img, visibility_mask, occlusion_mask], dim=1)
        else:
            cond_img = torch.zeros((B,1374+37*37+37*37,1024)).to(cond_img.device).to(cond_img.dtype)
        
        t = self.logit_normal(1.0, 1.0, (B,), device="cuda") # see trellis training details
        
        loss, _ = self.diffusion.p_losses(denoise_fn=self._denoise, data_start={'feats': sp_feats, 'coords': sp_coords}, t=t, y=cond_img, multi_t=True)
        loss = loss.mean()

        return None, loss
