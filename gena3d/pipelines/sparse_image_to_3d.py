from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import cv2
from ..utils.data_utils import img_pil_to_tensor, mask_pil_to_tensor, filter_by_depth, voxelize, write_ply
import torchvision.utils as vutils
import os
import uuid

# stereo model
from dependency.Pi3.pi3.models.pi3 import Pi3
from dependency.Pi3.pi3.utils.geometry import depth_edge
import gena3d.models as models
import open3d as o3d
import utils3d
from ..renderers import OctreeRenderer
from ..representations.octree import DfsOctree as Octree

class mask_patcher(nn.Module):
    def __init__(self):
        super(mask_patcher, self).__init__()

    def forward(self, mask, patch_size=14):
        mask = F.interpolate(mask.float(), size=(518, 518), mode='nearest')  # [B, 1, 518, 518]
        
        patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            
        patch_ratio = patches.mean(dim=(-1, -2))  # [B, 1, 37, 37]
        
        patch_ratio = patch_ratio.squeeze(1)  # [B, 37, 37]
        
        return patch_ratio
    
class SparseImageTo3D(Pipeline):
    """
    Pipeline for inferring AnyRec sparse input images-to-3D models.
    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)
        
        
    @staticmethod
    def from_pretrained_with_custom(
        path: str,
        custom_models: dict = None  # {"sparse_structure_decoder": MyModel(), "slat_decoder_gs": MyModel(), ...}
    ) -> "SparseImageTo3D":
        """
        Load a pretrained pipeline from Hugging Face or local path, 
        optionally replacing specific modules with custom models.

        Args:
            path (str): Path to pretrained pipeline (local or Hugging Face).
            custom_models (dict): Optional dict of custom modules to replace. 
                                  Keys can be 'sparse_structure_decoder', 
                                  'slat_decoder_gs', 'slat_decoder_mesh', 
                                  'slat_flow_model', 'image_cond_model', etc.
        """
        # 1️⃣ 加载原 pipeline
        pipeline = super(SparseImageTo3D, SparseImageTo3D).from_pretrained(path)
        
        # 2️⃣ 创建新的 pipeline 实例
        new_pipeline = SparseImageTo3D()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # 3️⃣ 初始化 sampler
        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args['sparse_structure_sampler']['name']
        )(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(
            samplers, args['slat_sampler']['name']
        )(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        # 4️⃣ 初始化 image_cond_model
        new_pipeline._init_image_cond_model(args['image_cond_model'])

        # 5️⃣ 初始化 stereo model / VAE encoder
        new_pipeline._init_stereo_model()
        new_pipeline._init_vae_encoder()
        new_pipeline.mask_patcher = mask_patcher()
        # 6️⃣ 替换自定义模块
        if custom_models is not None:
            for k, v in custom_models.items():
                if k in new_pipeline.models.keys():
                    print(f"[INFO] Replacing {k} with custom model {v.__class__.__name__}")
                    new_pipeline.models[k] = v
                else:
                    print(f"[WARNING] Pipeline has no attribute '{k}', skipping replacement.")

        return new_pipeline
    
    
    @staticmethod
    def from_pretrained(path: str) -> "SparseImageTo3D":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(SparseImageTo3D, SparseImageTo3D).from_pretrained(path)
        new_pipeline = SparseImageTo3D()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args
        # loading models...
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']
        # print(f'[INFO]: initializing image conditioning model: {args['image_cond_model']}')
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        # stereo
        # print(f'[INFO]: initializing stereo model: {args['stereo_model']}')
        new_pipeline._init_stereo_model(args['stereo_model'])
        # pc cond
        # print(f'[INFO]: initializing point cloud conditioning model: {args['pc_cond_model']}')
        new_pipeline._init_vae_encoder(args['vae_encoder'])
        new_pipeline.mask_patcher = mask_patcher()
        return new_pipeline
    
    
    # ====================== point cloud condition model ======================= #
    def _init_vae_encoder(self): # point cloud conditioning model
        """
        Initialize the point cloud conditioning model.
        """
        # initialize VAE model
        enc_pretrained = 'microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16'
        encoder = models.from_pretrained(enc_pretrained).eval().to(self.device)
        self.models['vae_encoder'] = encoder
    
    # ====================== stereo model initialization ======================= #
    def _init_stereo_model(self): # stereo model point cloud generation.
        """
        Initialize the stereo condition generation model.
        """
        stereo_model = Pi3.from_pretrained('yyfz233/Pi3').to(self.device).eval()
        self.models['stereo_model'] = stereo_model
    
    def _init_image_cond_model(self, name: str): # DINO v2 image conditioning.
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True, force_reload=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
    
    def resize_to_nearest_14_multiple(self, image):
        W, H = image.size  # PIL 的 size 是 (width, height)

        # 找到小于等于 H 和 W 的最近 14 的倍数
        new_H = (H // 14) * 14
        new_W = (W // 14) * 14

        # resize 图像
        resized_image = image.resize((new_W, new_H), Image.LANCZOS)

        return resized_image

    def preprocess_image_w_mask(self, input, mask, kernel_size=3):
        image = np.array(input).astype(np.float32) / 255
        mask_ori = np.array(mask).astype(np.float32)
        mask = (mask_ori < 127).astype(np.uint8)
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_occlude = cv2.dilate(mask, kernel, iterations=1) 
        else:
            mask_occlude = mask
        mask_bg = (mask_ori>230).astype(np.uint8)
        mask = mask_occlude | mask_bg
        image = image * (1 - mask[:, :, None])
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
        mask_occ = np.zeros(mask.shape)
        mask_occ[mask_occlude==1] = 1
        return image, mask, mask_occ

    def preprocess_image_aligned(self, input_img: Image.Image, occluded: Image.Image, mask_img: Image.Image, vis_mask: Image.Image):
        """
        Preprocess input_img like preprocess_image, and apply the same transformation
        to occluded and mask_img so they remain aligned.
        """
        # ----------------- 1. 前景提取 -----------------
        has_alpha = False
        if input_img.mode == 'RGBA':
            alpha = np.array(input_img)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        if has_alpha:
            output = input_img
        else:
            input_img = input_img.convert('RGB')
            max_size = max(input_img.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input_img = input_img.resize(
                    (int(input_img.width * scale), int(input_img.height * scale)),
                    Image.Resampling.LANCZOS
                )
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input_img, session=self.rembg_session)

        # ----------------- 2. 计算前景 bbox -----------------
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox_coords = np.argwhere(alpha > 0.8 * 255)
        xmin, ymin = np.min(bbox_coords[:, 1]), np.min(bbox_coords[:, 0])
        xmax, ymax = np.max(bbox_coords[:, 1]), np.max(bbox_coords[:, 0])
        
        center = (xmin + xmax) / 2, (ymin + ymax) / 2
        size = max(xmax - xmin, ymax - ymin)
        size = int(size * 1.2)
        bbox = (int(center[0] - size // 2), int(center[1] - size // 2),
                int(center[0] + size // 2), int(center[1] + size // 2))

        # ----------------- 3. 裁剪 & resize -----------------
        def crop_and_resize(img: Image.Image):
            img_cropped = img.crop(bbox)
            img_resized = img_cropped.resize((518, 518), Image.Resampling.LANCZOS)
            return img_resized

        # 对 input 图
        input_output = crop_and_resize(output)
        input_np = np.array(input_output).astype(np.float32) / 255
        input_np[:, :, :3] = input_np[:, :, :3] * input_np[:, :, 3:4]  # alpha 融合
        input_output = Image.fromarray((input_np[:, :, :3] * 255).astype(np.uint8))

        # 对 extra 图 (RGB)
        occluded_output = crop_and_resize(occluded)

        # 对 mask 图 (保留整数/0-1)
        mask_output = crop_and_resize(mask_img)
        mask_output = mask_output.convert('L')  # 确保单通道
        mask_np = np.array(mask_output)
        mask_np = (mask_np > 127).astype(np.uint8) * 255  # 二值化
        mask_output = Image.fromarray(mask_np)
        
        # 对 mask 图 (保留整数/0-1)
        vis_mask_output = crop_and_resize(vis_mask)
        vis_mask_output = vis_mask_output.convert('L')  # 确保单通道
        vis_mask_np = np.array(vis_mask_output)
        vis_mask_np = (vis_mask_np > 127).astype(np.uint8) * 255  # 二值化
        vis_mask_output = Image.fromarray(vis_mask_np)

        return input_output, occluded_output, mask_output, vis_mask_output

# ============== including remove background ============== #
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            # ============ Returning stacked image conditions ============= #
            image = torch.stack(image).to(self.device)
            # ============ Returning stacked image conditions ============= #
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        # since layernorm is independent to each other in the calculation process, these patch tokens are still independent.
        # patchtokens = F.layer_norm(features, features.shape[-1:])
        
        # ----------------  encode images independently and concat them. ---------------- #
        B, N, D = features.shape
        normed_list = []
        for i in range(B):
            sample = features[i]  # [N, D]
            normed = F.layer_norm(sample, (D,))  # normalization in the last feat dim.
            normed_list.append(normed)
        # concat to patchtokens
        patchtokens = torch.concat(normed_list, dim=0)
        patchtokens = patchtokens.unsqueeze(0) # [B, N, D]
        # ----------------  encode images independently and concat them. ---------------- #
        return patchtokens
    
    @torch.no_grad()
    def stereo_pc_gen(self, scene_image, scene_mask = None, use_scene_mask=False):
        """
        Run stereo model with sparse input images. And obtain the mask-aligned point cloud.
        input: N, H, W, C
        """
        # 1. Prepare input data
        # The load_images_as_tensor function will print the loading path

        imgs = scene_image.permute(0, 3, 1, 2) # (N, 3, H, W)
        # 2. Infer
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.models['stereo_model'](imgs[None]) # Add batch dimension
                
        # Optional: loading mask images.
        if scene_mask is not None:
            masks_input = mask_pil_to_tensor(scene_mask, interval=1).to(self.device)
            # 4. process mask @@ in res: points:(1, N, H, W, 3), totally these points...
            final_mask = masks_input > 0.5
            if final_mask.sum() == 0:
                pos, col = res['points'][0], imgs.permute(0, 2, 3, 1) 
                print('[INFO]: empty occlusion mask.')
            else:
                pos, col = res['points'][0][final_mask], imgs.permute(0, 2, 3, 1)[final_mask]
        else:
            masks = torch.sigmoid(res['conf'][..., 0]) > 0.0
            non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
            masks = torch.logical_and(masks, non_edge)[0]
            pos, col = res['points'][0][masks], imgs.permute(0, 2, 3, 1)[masks]
        return [pos, col]
    

    @torch.no_grad()
    def encode_pc(self, scene_image, mask=None, use_scene_mask=False, resolution: int = 64) -> torch.Tensor:
        """
        Get the conditioning information of possible occluded regions.
        """
        img = torch.from_numpy(np.array(scene_image).astype(np.float32) / 255.).to(self.device)
        pos, col = self.stereo_pc_gen(img, mask, use_scene_mask)
        # =========== pipeline input evaluation =========== #
        # =========== pipeline input evaluation =========== #
        voxel = voxelize(pos) # only incorporate position during training. for alignment of encode ss latent?
        coords = ((torch.tensor(voxel) + 0.5) * resolution).int().contiguous()
        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        ss = ss[None].to(self.device).float() # should be on to the gpu assigned in this class..
        feat_voxel = self.vae_encode(ss)
        return feat_voxel
    
    @torch.no_grad()
    def vae_encode(self, ss):
        """
        Encode the voxels with VAE encoder.
        serving as conditions.
        """
        latent = self.models['vae_encoder'](ss, sample_posterior=False) # [1, 8, 16, 16, 16]
        assert torch.isfinite(latent).all(), "Non-finite latent"
        return latent
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        # base cond
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
        

    def get_cond_full(self, image: Union[torch.Tensor, list[Image.Image]],
                    occluded: Union[torch.Tensor, list[Image.Image]],
                    mask: Union[torch.Tensor, list[Image.Image]]=None, 
                    scene_img: Union[torch.Tensor, list[Image.Image]]=None, 
                    use_scene_mask: bool = False) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.
            mask: Union[torch.Tensor, list[Image.Image]]: the mask indicator.
            

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image) # multiple images or single image  #BND
        # stereo generation & vae encode
        if use_scene_mask:
            cond_pc = self.encode_pc(scene_img, mask, use_scene_mask)
        else:
            cond_pc = self.encode_pc(occluded, mask, use_scene_mask)
        B, C_, V, V, V = cond_pc.shape
        V3 = V ** 3
        cond_pc = cond_pc.view(B, C_, V3).permute(0, 2, 1)  # [B, V3, 8]
        cond_pc = cond_pc.repeat(1, 1, 1024 // C_)
        cond = torch.cat([cond, cond_pc], dim=1)
        
        # in all, the condition components are: [img1, img2, ..., imgN, | mask/occ(1-N), | cond_pc]
        
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
        
    def get_cond_no_pc(self, image: Union[torch.Tensor, list[Image.Image]],
                    mask: Union[torch.Tensor, list[Image.Image]]=None, 
                    occ_mask: Union[torch.Tensor, list[Image.Image]]=None, 
                    use_mask: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.
            mask: Union[torch.Tensor, list[Image.Image]]: the mask indicator.
            depth
            scene
            use_mask

        Returns:
            dict: The conditioning information
        """
        b = len(image)
        cond = self.encode_image(image) # multiple images or single image
        cond = cond.view(b, -1, 1024)    # [B, T_img, 1024]
        if use_mask and mask is not None:
            mask = [torch.from_numpy(np.array(m).astype(np.float32)/255.0).unsqueeze(0).float() for m in mask]
            mask = torch.stack(mask).to(self.device)
            masks_occ = [torch.from_numpy(np.array(m).astype(np.float32)/255.0).unsqueeze(0).float() for m in occ_mask]
            masks_occ = torch.stack(masks_occ).to(self.device)
            mask = self.mask_patcher(mask)
            masked_feat = mask.view(b, 37 * 37).unsqueeze(-1).repeat(1, 1, 1024)
            masks_occ = self.mask_patcher(masks_occ)
            masks_occ = masks_occ.view(b, 37 * 37).unsqueeze(-1).repeat(1, 1, 1024)

            
            cond = torch.cat([cond, masked_feat, masks_occ], dim=1)
        # in all, the condition components without point clouds are: [img1, img2, ..., imgN, | mask/occ(1-N) | all zeros]
        print('==========CONDITIONSHAPE:========:', {cond.shape})
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
        
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        get_voxel_vis=False
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        if get_voxel_vis:
            # ---------- 初始化渲染器 ----------
            renderer = OctreeRenderer()
            renderer.rendering_options.resolution = 512
            renderer.rendering_options.near = 0.8
            renderer.rendering_options.far = 1.6
            renderer.rendering_options.bg_color = (0, 0, 0)
            renderer.rendering_options.ssaa = 4
            renderer.pipe.primitive = 'voxel'

            # ---------- 设置多视角 ----------
            yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            yaws = [y + yaws_offset for y in yaws]
            pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

            exts = []
            ints = []
            for yaw, p in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(p),
                    np.cos(yaw) * np.cos(p),
                    np.sin(p),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(40)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(
                    orig, 
                    torch.tensor([0, 0, 0], device='cuda', dtype=torch.float32), 
                    torch.tensor([0, 0, 1], device='cuda', dtype=torch.float32)
                )
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                exts.append(extrinsics)
                ints.append(intrinsics)

            # ---------- 生成体素表示 ----------
            x_0 = decoder(z_s)  # 输出 shape: [batch_size, 1, R, R, R]
            images = []
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords_vis = torch.nonzero(x_0[0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords_vis.float() / resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(resolution)),
                dtype=torch.uint8,
                device='cuda'
            )
            # ---------- 多视角渲染拼接 ----------
            image = torch.zeros(3, 1024, 1024, device='cuda')
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1),
                        512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            images = torch.stack(images)  # [batch_size, 3, 1024, 1024]
            return coords, images
        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    @torch.no_grad()
    def run(
        self,
        image: Union[Image.Image, List[Image.Image]],
        occluded: Union[Image.Image, List[Image.Image]],
        mask: Union[Image.Image, List[Image.Image]]=None,
        occ_mask: Union[Image.Image, List[Image.Image]]=None, 
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
        scene_image = None, 
        depth = None,
        use_mask = False,
        get_voxel_vis = True,
        custom_model = None     
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if isinstance(image, list):
            if preprocess_image:
                processed_images = []
                processed_occluded = []
                processed_occ_mask = []
                processed_vis_mask = []

                # 遍历每个 sample
                for i in range(len(image)):
                    input_img = image[i]
                    extra_img = occluded[i] if occluded is not None else None
                    mask_img = occ_mask[i] if occ_mask is not None else None
                    vis_mask = mask[i] if mask is not None else None

                    # 调用新的 preprocess 函数
                    input_out, extra_out, mask_out, vis_mask_out = self.preprocess_image_aligned(
                        input_img,
                        extra_img if extra_img is not None else input_img,  # 如果 occluded 没有，就用 input 占位
                        mask_img if mask_img is not None else input_img,  # 同理
                        vis_mask if vis_mask is not None else input_img,
                    )

                    processed_images.append(input_out)
                    if extra_img is not None:
                        processed_occluded.append(extra_out)
                    if mask_img is not None:
                        processed_occ_mask.append(mask_out)
                    if vis_mask is not None:
                        processed_vis_mask.append(vis_mask_out)

                # 替换原来的列表
                image = processed_images
                if occluded is not None:
                    occlude_img = processed_occluded
                if occ_mask is not None:
                    occ_mask = processed_occ_mask
                if mask is not None:
                    mask = processed_vis_mask
                # image = [self.preprocess_image(img) for img in image]
                # occlude_img = [self.preprocess_image(occ) for occ in occluded]
                # if occ_mask is not None:
                #     occ_mask = [self.preprocess_image(occ_msk) for occ_msk in occ_mask]
        else:
            if preprocess_image:
                    # 假设 image 和 occluded 都是单张图片，occ_mask 是 list
                input_img = image
                extra_img = occluded
                mask_imgs = occ_mask

                # 调用新的 preprocess_image_with_extra
                input_out, extra_out, mask_out_list = self.preprocess_image_aligned(
                    input_img,
                    extra_img,
                    mask_imgs[0] if mask_imgs[0] is not None else input_img,
                    rembg_session=getattr(self, 'rembg_session', None)
                )

                # 替换原来的变量
                image = [input_out]
                occlude_img = [extra_out]
                if occ_mask is not None:
                    occ_mask = mask_out_list if isinstance(mask_out_list, list) else [mask_out_list]
                
                
                # image = [self.preprocess_image(image)]
                # occlude_img = [self.preprocess_image(occluded)]
                # if occ_mask is not None:
                #     occ_mask = [self.preprocess_image(occ_msk) for occ_msk in occ_mask]
        # here we do not process mask...
        image = [im.convert('RGB') for im in image]
        occlude_img = [im.convert('RGB') for im in occlude_img]
        print(image[0].mode)
        cond = self.get_cond_full(image, occlude_img, mask) 
        torch.manual_seed(seed)
        if custom_model is not None:
            coords, images = self.sample_sparse_structure_custom_model(cond, num_samples, sparse_structure_sampler_params, custom_model=custom_model, get_voxel_vis=get_voxel_vis) # cond full
        else:
            coords, images = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, get_voxel_vis=get_voxel_vis) # cond full
        cond_nopc = self.get_cond_no_pc(image, mask, occ_mask)
        cond_nopc['neg_cond'] = cond_nopc['neg_cond'][:1]
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(image), slat_steps, mode='stochastic'):
            slat = self.sample_slat(cond_nopc, coords, slat_sampler_params)
        # slat = self.sample_slat(cond_nopc, coords, slat_sampler_params) # cond_nopc
        return self.decode_slat(slat, formats)
    
    @torch.no_grad()
    def run_scene(
        self,
        image: Union[Image.Image, List[Image.Image]],
        occluded: Union[Image.Image, List[Image.Image]],
        mask: Union[Image.Image, List[Image.Image]]=None,
        occ_mask: Union[Image.Image, List[Image.Image]]=None, 
        scene_masks: Union[Image.Image, List[Image.Image]]=None,
        scene_images: Union[Image.Image, List[Image.Image]]=None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
        depth = None,
        use_mask = False,
        get_voxel_vis = True,
        custom_model = None     
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if isinstance(image, list):
            if preprocess_image:
                # resize to dino v2 results.
                scene_images = [self.resize_to_nearest_14_multiple(si) for si in scene_images]
                scene_masks = [self.resize_to_nearest_14_multiple(si) for si in scene_masks]
                smsk_list = []
                for smsk in scene_masks:
                    vis_mask = np.array(smsk.convert('RGB'), dtype=np.uint8)
                    is_background = np.all(vis_mask < 10, axis=-1)
                    # 非背景 = 真实mask区域
                    vis_mask = (~is_background).astype(np.uint8)
                    vis_mask = Image.fromarray(vis_mask * 255)
                    smsk_list.append(vis_mask)
                scene_masks = smsk_list
                processed_images = []
                processed_occluded = []
                processed_occ_mask = []
                processed_vis_mask = []

                # 遍历每个 sample
                for i in range(len(image)):
                    input_img = self.resize_to_nearest_14_multiple(image[i])
                    extra_img = self.resize_to_nearest_14_multiple(occluded[i]) if occluded is not None else None
                    mask_img = self.resize_to_nearest_14_multiple(occ_mask[i]) if occ_mask is not None else None
                    vis_mask = self.resize_to_nearest_14_multiple(mask[i]) if mask is not None else None
                    # # scene inference needs.
                    # vis_mask = np.array(vis_mask.convert('RGB'), dtype=np.uint8)
                    # # 背景 = 所有通道 < bg_thresh
                    # is_background = np.all(vis_mask < 10, axis=-1)
                    # # 非背景 = 真实mask区域
                    # vis_mask = (~is_background).astype(np.uint8)
                    # vis_mask = Image.fromarray(vis_mask * 255)
                    # scene_masks.append(vis_mask) # scene masks...

                    # 调用新的 preprocess 函数
                    input_out, extra_out, mask_out, vis_mask_out = self.preprocess_image_aligned(
                        input_img,
                        extra_img if extra_img is not None else input_img,  # 如果 occluded 没有，就用 input 占位
                        mask_img if mask_img is not None else input_img,  # 同理
                        vis_mask if vis_mask is not None else input_img,
                    )

                    processed_images.append(input_out)
                    if extra_img is not None:
                        processed_occluded.append(extra_out)
                    if mask_img is not None:
                        processed_occ_mask.append(mask_out)
                    if vis_mask is not None:
                        processed_vis_mask.append(vis_mask_out)

                # 替换原来的列表
                image = processed_images
                if occluded is not None:
                    occlude_img = processed_occluded
                if occ_mask is not None:
                    occ_mask = processed_occ_mask
                if mask is not None:
                    mask = processed_vis_mask
                # image = [self.preprocess_image(img) for img in image]
                # occlude_img = [self.preprocess_image(occ) for occ in occluded]
                # if occ_mask is not None:
                #     occ_mask = [self.preprocess_image(occ_msk) for occ_msk in occ_mask]
        else:
            if preprocess_image:
                    # 假设 image 和 occluded 都是单张图片，occ_mask 是 list
                input_img = image
                extra_img = occluded
                mask_imgs = occ_mask

                # 调用新的 preprocess_image_with_extra
                input_out, extra_out, mask_out_list = self.preprocess_image_aligned(
                    input_img,
                    extra_img,
                    mask_imgs[0] if mask_imgs[0] is not None else input_img,
                    rembg_session=getattr(self, 'rembg_session', None)
                )

                # 替换原来的变量
                image = [input_out]
                occlude_img = [extra_out]
                if occ_mask is not None:
                    occ_mask = mask_out_list if isinstance(mask_out_list, list) else [mask_out_list]
                
                
                # image = [self.preprocess_image(image)]
                # occlude_img = [self.preprocess_image(occluded)]
                # if occ_mask is not None:
                #     occ_mask = [self.preprocess_image(occ_msk) for occ_msk in occ_mask]
        # here we do not process mask...
        # here occlude img has no use.... only scene_masks and scene_images has something to do.
        cond = self.get_cond_full(image, occlude_img, scene_masks, scene_img=scene_images, use_scene_mask=True) 
        torch.manual_seed(seed)
        if custom_model is not None:
            coords, images = self.sample_sparse_structure_custom_model(cond, num_samples, sparse_structure_sampler_params, custom_model=custom_model, get_voxel_vis=get_voxel_vis) # cond full
        else:
            coords, images = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, get_voxel_vis=get_voxel_vis) # cond full
        
        image = [im.convert('RGB') for im in image]
        
        cond_nopc = self.get_cond_no_pc(image, mask, occ_mask)
        slat = self.sample_slat(cond_nopc, coords, slat_sampler_params) # cond_nopc
        return self.decode_slat(slat, formats)
    
    
    def sample_sparse_structure_custom_model(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        custom_model = None,
        get_voxel_vis = True
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        assert custom_model is not None, 'No custom model loaded.'
        flow_model = custom_model
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        if get_voxel_vis:
            # ---------- 初始化渲染器 ----------
            renderer = OctreeRenderer()
            renderer.rendering_options.resolution = 512
            renderer.rendering_options.near = 0.8
            renderer.rendering_options.far = 1.6
            renderer.rendering_options.bg_color = (0, 0, 0)
            renderer.rendering_options.ssaa = 4
            renderer.pipe.primitive = 'voxel'

            # ---------- 设置多视角 ----------
            yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            yaws = [y + yaws_offset for y in yaws]
            pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

            exts = []
            ints = []
            for yaw, p in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(p),
                    np.cos(yaw) * np.cos(p),
                    np.sin(p),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(40)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(
                    orig, 
                    torch.tensor([0, 0, 0], device='cuda'), 
                    torch.tensor([0, 0, 1], device='cuda')
                )
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                exts.append(extrinsics)
                ints.append(intrinsics)

            # ---------- 生成体素表示 ----------
            x_0 = decoder(z_s)  # 输出 shape: [batch_size, 1, R, R, R]
            images = []
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords_vis = torch.nonzero(x_0[0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords_vis.float() / resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(resolution)),
                dtype=torch.uint8,
                device='cuda'
            )
            # ---------- 多视角渲染拼接 ----------
            image = torch.zeros(3, 1024, 1024, device='cuda')
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1),
                        512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            images = torch.stack(images)  # [3, 1024, 1024]
            return coords, images
        return coords
    
    @torch.no_grad()
    def vis_ss(
        self,
        image: Union[Image.Image, List[Image.Image]],
        occluded: Union[Image.Image, List[Image.Image]],
        mask: Union[Image.Image, List[Image.Image]]=None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        scene_image = None, 
        depth = None,
        use_mask = False,
        get_voxel_vis = True,
        custom_model = None,
        save_path = None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(img) for img in image]
                occluded = [self.preprocess_image(occ) for occ in occluded]
        else:
            if preprocess_image:
                image = [self.preprocess_image(image)]
                occluded = [self.preprocess_image(occluded)]
        # here we do not process mask...
        cond = self.get_cond_full(image, occluded, mask, scene_image=scene_image, depth_image=depth, use_mask=use_mask) 
        torch.manual_seed(seed)
    
        if custom_model is not None:
            coords, images = self.sample_sparse_structure_custom_model(cond, num_samples, sparse_structure_sampler_params, custom_model=custom_model, get_voxel_vis=get_voxel_vis) # cond full
        else:
            coords, images = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, get_voxel_vis=get_voxel_vis) # cond full
        
        # # visualize sparse structure voxels.
        points = coords[:, 1:].cpu().numpy().astype(np.float32)
        # Create PC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Adding colors
        colors = np.tile(np.array([[0.2, 0.7, 1.0]]), (points.shape[0], 1))  # 蓝色
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Saveply
        if save_path is None:
            o3d.io.write_point_cloud("sparse_structure_voxel.ply", pcd)
            print("[INFO]: saved Sparse Structure Visualization to [sparse_structure_voxel.ply]")
        else:
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"[INFO]: saved Sparse Structure Visualization to {save_path}!")
        # visualize sparse structure voxels.
        
        for i, img in enumerate(images):
            vutils.save_image(img, os.path.join(os.path.dirname(save_path), f'voxel_feat_{save_path[-5]}.png'))
        cond_nopc = self.get_cond_no_pc(image, mask)
        # slat = self.sample_slat(cond_nopc, coords, slat_sampler_params) # cond_nopc
        return True# self.decode_slat(slat, formats)
    
    
    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    # @torch.no_grad()
    # def run_multi_image(
    #     self,
    #     images: List[Image.Image],
    #     num_samples: int = 1,
    #     seed: int = 42,
    #     sparse_structure_sampler_params: dict = {},
    #     slat_sampler_params: dict = {},
    #     formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    #     preprocess_image: bool = True,
    #     mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    #     get_voxel_vis = True
    # ) -> dict:
    #     """
    #     Run the pipeline with multiple images as condition

    #     Args:
    #         images (List[Image.Image]): The multi-view images of the assets
    #         num_samples (int): The number of samples to generate.
    #         sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
    #         slat_sampler_params (dict): Additional parameters for the structured latent sampler.
    #         preprocess_image (bool): Whether to preprocess the image.
    #     """
    #     if preprocess_image:
    #         images = [self.preprocess_image(image) for image in images]
    #     cond = self.get_cond(images)
    #     cond['neg_cond'] = cond['neg_cond'][:1]
    #     torch.manual_seed(seed)
    #     ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
    #     with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
    #         coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, get_voxel_vis=get_voxel_vis,)
    #     slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
    #     with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
    #         slat = self.sample_slat(cond, coords, slat_sampler_params)
    #     return self.decode_slat(slat, formats)


#

# images_path = [
#     image_name for image_name in os.listdir(Scene_path)
#     if image_name.endswith(".png") and image_name != "scene.jpg" and "mask" not in image_name
# ]
# images_path = sorted(images_path) 
# images = [
#     Image.open(os.path.join(Scene_path, image_name))
#     for image_name in images_path
# ] 
# mask_images_path = [
#     image_name for image_name in os.listdir(Scene_path)
#     if image_name.endswith(".png") and image_name != "scene.jpg" and "mask" in image_name and "masked_scene" not in image_name
# ]

# mask_images_path = sorted(mask_images_path)
# mask_images = [
#     Image.open(os.path.join(Scene_path, image_name))
#     for image_name in mask_images_path
#@ TODO: instantly we do not discard images for convenience, check above for read in information
    @torch.no_grad()
    def run_multi_image(
        self,
        image: List[Image.Image],
        scene_image: List[Image.Image],
        depth_image: List[Image.Image],
        mask: List[Image.Image] = None,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
        preprocess_image: bool = True,
        get_voxel_vis: bool = True,
    ):
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            scene_image (List[Image.Image]): Input global scene image prompt
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
            get_voxel_vis (bool): Whether to get voxel visualization.
        """
        # ====== get segments and centralize ====== #
        image = []
        for i in range(len(scene_image)):
            scene_img_array = np.array(scene_image[i])
            mask_array = np.array(mask[i])
            mask_array = (mask_array > 0).astype(np.uint8)
            seg = scene_img_array * mask_array[..., None]  # mask[..., None] 变成 (H, W, 1) 便于广播
            seg_img = Image.fromarray(seg)
            if preprocess_image:
                image[i] = self.preprocess_image(seg_img)
        # ====== get segments and centralize ====== #
        
        # ====== union of condition for sparse input ====== #  
        cond_full = self.get_cond_full(
            image,
            mask,
            scene_image,
            depth_image,
        )
        torch.manual_seed(seed)

        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(scene_image), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(
                cond_full,
                sparse_structure_sampler_params,
                get_voxel_vis=get_voxel_vis,
            )

        cond_no_pc = self.get_cond_no_pc(
            image,
            mask,
            scene_image,
            depth_image,
        )

        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(scene_image), slat_steps, mode=mode):
            slat = self.sample_slat(
                cond_no_pc,
                coords,
                slat_sampler_params
            )

        return self.decode_slat(slat, formats)
    
    