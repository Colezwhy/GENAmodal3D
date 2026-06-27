import numpy as np
import os
import random
from PIL import Image
import torch
import json
from dataLoader.utils import generate_random_mask, fov_to_ixt
from .data_utils import img_pil_to_tensor, mask_pil_to_tensor, filter_by_depth, voxelize

class GENA3DData(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(GENA3DData, self).__init__()
        self.data_root_abo = cfg.data_root_abo
        self.data_root_3df = cfg.data_root_3df
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        self.img_downscale = self.img_size/512
        self.data_root = None   
        
        self.n_view = 2 # default value
        self.n_views = [1, 2, 3, 5]
        
        self.category = "abo_3df"

        if self.category == "combined":
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}')])
            self.split_num = len(scenes_name)
        elif self.category == "abo":
            # with open("./dataset/ABO/abo.txt", "r") as f:
            #     base_scenes = f.readlines()
            # base_scenes = [f.strip() for f in base_scenes]
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}')])
            self.split_num = 4250
            # currently we use abo dataset as a simple test..
        elif self.category == "3df":
            with open("./dataset/3dfuture/3dfuture.txt", "r") as f:
                base_scenes = f.readlines()
            base_scenes = [f.strip() for f in base_scenes]
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}') and f in base_scenes])
            self.split_num = 8000
        elif self.category == "hssd":
            with open("./dataset/hssd/hssd.txt", "r") as f:
                base_scenes = f.readlines()
            base_scenes = [f.strip() for f in base_scenes]
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}') and f in base_scenes])
            self.split_num = 6000
        elif self.category == "abo_3df":
            scenes_name_abo = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root_abo, 'renders'))) if os.path.isdir(f'{self.data_root_abo}/renders/{f}')])
            scenes_name_3df = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root_3df, 'renders'))) if os.path.isdir(f'{self.data_root_3df}/renders/{f}')])
            scenes_name = np.concatenate([scenes_name_abo, scenes_name_3df])
            self.split_num = 13500
            
            
        # self.split_num = 50

        if self.split=='train':
            self.scenes_name = scenes_name[:self.split_num]
        else:
            if self.category == "combined":
                self.scenes_name = scenes_name[-100:]
            else:
                self.scenes_name = scenes_name[self.split_num:]

        print(self.category, " ", self.split, ": ", len(self.scenes_name))

    def update_views(self, new_n):
        print(f'[INFO]: Using {new_n} in this epoch..')
        self.n_view = new_n
        
    def build_metas(self, scene):
        # originally from 'renders', here we only take 'inpainted_cond'
        if os.path.exists(os.path.join(self.data_root_abo, 'inpainted_cond', scene, f'transforms.json')):
            jsonfile = os.path.join(self.data_root_abo, 'inpainted_cond', scene, f'transforms.json')
            self.data_root = self.data_root_abo
        else:
            jsonfile = os.path.join(self.data_root_3df, 'inpainted_cond', scene, f'transforms.json')
            self.data_root = self.data_root_3df
        json_info = json.load(open(jsonfile))
        b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        scene_info = {'ixts': [], 'c2ws': [], 'w2cs':[], 'img_paths': [], 'fovx': [], 'fovy':[], "occluded": [], "inpainted_cond_paths": [], "occlusion_mask": [], "visibility": []}
        positions = []
        # Here we only load the inpainted condition images, basically 12 inpainted images from all views...
        for idx, frame in enumerate(json_info['frames']):
            filename = frame['file_path']
            c2w = np.array(frame['transform_matrix'])
            c2w = c2w @ b2c
            fov = frame["camera_angle_x"]
            ixt = fov_to_ixt(np.array([fov, fov]), np.array([512, 512]))
            scene_info['ixts'].append(ixt.astype(np.float32))
            scene_info['c2ws'].append(c2w.astype(np.float32))
            scene_info['w2cs'].append(np.linalg.inv(c2w.astype(np.float32)))
            img_path = os.path.join(self.data_root, 'renders', scene, filename)
            scene_info['img_paths'].append(img_path)
            scene_info['fovx'].append(fov)
            scene_info['fovy'].append(fov)
            positions.append(c2w[:3,3])
            # ========= TODO@: maybe more complicated mask generation process? Should throw away the exising part or not?
            mask_path = os.path.join(self.data_root, 'occluded', scene, filename) # FOR 3D POINT CLOUD
            inpainted_cond_path = os.path.join(self.data_root, 'inpainted_cond', scene, filename) # FOR input sparse view conditions
            occlusion_path = os.path.join(self.data_root, 'occ_mask', scene, filename) # FOR the 2nd added cross attention
            visibility_path = os.path.join(self.data_root, 'visibility', scene, filename) # Point cloud generation
            
            scene_info['occluded'].append(mask_path) # 3d consistent occluded regions
            scene_info['inpainted_cond_paths'].append(inpainted_cond_path)
            scene_info['occlusion_mask'].append(occlusion_path)
            scene_info['visibility'].append(visibility_path)
            # ========= TODO@: maybe more complicated mask generation process? Should throw away the exising part or not?
        
        
        scene_info['ss_latent_path'] = os.path.join(self.data_root, 'ss_latents/ss_enc_conv3d_16l8_fp16', scene + '.npz')
        scene_info['slat_path'] = os.path.join(self.data_root, 'latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16', scene + '.npz')
    
        return scene_info
        

    def __getitem__(self, index):
        scene_name = self.scenes_name[index]
        scene_info = self.build_metas(scene_name)
        
        ss_latent = np.load(scene_info['ss_latent_path'])['mean']
        ss_latent = torch.from_numpy(ss_latent).float()
        
        bg_color = np.zeros(3).astype(np.float32)
        # only inpainted_cond images..
        num_frames = len(scene_info['img_paths'])
        # choose a random frame
        # frame_idx = random.randint(0, num_frames-1)
        frame_indices = random.sample(range(num_frames), self.n_view)
        
        img_518, cond_inpaint_518, occlusion_mask, occluded, visibility = self.read_image(scene_info, frame_indices, bg_color)
        # 'orig_img': original image for reference, 'cond_inpaint': inpainting image as condition, 'occlusion_mask': occlusion mask as condition for 2nd stage, 'occluded': occluded for stereo gen.
        ret = {'orig_img': img_518, 'cond_inpaint': cond_inpaint_518, 'ss_latent': ss_latent, 'slat_path': scene_info['slat_path'], 'occlusion_mask': occlusion_mask, 'occluded': occluded, 'visibility': visibility}
        return ret

    def read_image(self, scene, view_idxes, bg_color):
        imgs_list = []
        cond_inpaint_list = []
        occ_mask_list = []
        occluded_list = []
        visibility_list = []
    
        for view_idx in view_idxes:
            img_path = scene['img_paths'][view_idx]
            img = Image.open(img_path).convert("RGBA")
            img_518 = img.resize((518, 518), Image.Resampling.LANCZOS)

            img_518 = np.array(img_518).astype(np.float32) / 255.
            alpha = img_518[..., -1:]
            img_518 = (img_518[..., :3] * img_518[..., -1:] + bg_color*(1 - img_518[..., -1:])).astype(np.float32)
            imgs_list.append(img_518)
            
            inpaint_cond_path = scene['inpainted_cond_paths'][view_idx]
            cond_inpaint = Image.open(inpaint_cond_path).convert("RGBA")
            cond_inpaint_518 = cond_inpaint.resize((518, 518), Image.Resampling.LANCZOS)

            cond_inpaint_518 = np.array(cond_inpaint_518).astype(np.float32) / 255.
            alpha = cond_inpaint_518[..., -1:]
            cond_inpaint_518 = (cond_inpaint_518[..., :3] * cond_inpaint_518[..., -1:] + bg_color*(1 - cond_inpaint_518[..., -1:])).astype(np.float32)
            cond_inpaint_list.append(cond_inpaint_518)
            
            if os.path.exists(scene['occlusion_mask'][view_idx]):
                occlusion_mask = Image.open(scene['occlusion_mask'][view_idx]).convert('L')
                occ_518 = occlusion_mask.resize((518, 518), Image.Resampling.LANCZOS)
                occ_518 = np.array(occ_518).astype(np.float32) / 255.
                occ_mask_list.append(occ_518)
            else:
                occ_518 = np.zeros((518, 518), dtype=np.float32)
                occ_mask_list.append(occ_518)
            if os.path.exists(scene['occluded'][view_idx]):
                occluded = Image.open(scene['occluded'][view_idx]).convert("RGB")
                occluded = occluded.resize((518, 518), Image.Resampling.LANCZOS)
                occluded = np.array(occluded).astype(np.float32) / 255.
                occluded_list.append(occluded)
            else:
                occluded = Image.open(scene['occluded'][view_idx].replace('occluded', 'inpainted')).convert("RGB")
                occluded = occluded.resize((518, 518), Image.Resampling.LANCZOS)
                occluded = np.array(occluded).astype(np.float32) / 255.
                occluded_list.append(occluded)
        
            # ------------- do not need ------------- #
            if os.path.exists(scene['visibility'][view_idx]):
                visibility_mask = Image.open(scene['visibility'][view_idx]).convert('L')
                vis_518 = visibility_mask.resize((518, 518), Image.Resampling.LANCZOS)
                # true_occlusion = Image.open(scene['occluded']).convert('L') # the occluded mask
                # occlusion_mask = generate_random_mask(height=518, width=518)
                # bg_mask = (alpha<0.1)
                # visibility_mask = occlusion_mask | bg_mask
                # visibility_mask = 1 - visibility_mask
                visibility_list.append(vis_518)
            else:
                visibility_mask = (alpha > 0.1).squeeze(-1)
                visibility_list.append(visibility_mask)
            # ------------- do not need ------------- #
        
        img_518 = np.stack(imgs_list, axis=0)      # B HW 3
        cond_inpaint_518 = np.stack(cond_inpaint_list, axis=0)      # B HW 3
        occ_mask_518 = np.stack(occ_mask_list, axis=0) # B HW
        occluded_518 = np.stack(occluded_list, axis=0) # B HW 3
        visibility_518 = np.stack(visibility_list, axis=0)

        return img_518.astype(np.float32), cond_inpaint_518.astype(np.float32), occ_mask_518.astype(np.float32), occluded_518.astype(np.float32), visibility_518.astype(np.float32)


    # def read_image(self, scene, view_idx, bg_color):
    #     img_path = scene['img_paths'][view_idx]
    #     img = Image.open(img_path).convert("RGBA")
    #     img_518 = img.resize((518, 518), Image.Resampling.LANCZOS)

    #     img_518 = np.array(img_518).astype(np.float32) / 255.
    #     alpha = img_518[..., -1:]
    #     img_518 = (img_518[..., :3] * img_518[..., -1:] + bg_color*(1 - img_518[..., -1:])).astype(np.float32)
        
    #     occlusion_mask = Image.open(scene['img_paths'][view_idx]).convert('RGBA')
    #     occ_518 = occlusion_mask.resize((518, 518), Image.Resampling.LANCZOS)
    #     visibility_mask = Image.open(scene['mask_paths'][view_idx]).convert('L')
    #     vis_518 = visibility_mask.resize((518, 518), Image.Resampling.LANCZOS)
    #     true_occlusion = Image.open(scene['occluded']).convert('L') # the occluded mask
    #     # occlusion_mask = generate_random_mask(height=518, width=518)
    #     bg_mask = (alpha<0.1)
    #     visibility_mask = occlusion_mask | bg_mask
    #     visibility_mask = 1 - visibility_mask
        
    #     # apply the mask
    #     img_518_masked = img_518 * visibility_mask

    #     return img_518.astype(np.float32), img_518_masked.astype(np.float32), visibility_mask.squeeze().astype(np.float32), occlusion_mask.squeeze().astype(np.float32)
    
    def __len__(self):
        return len(self.scenes_name)
    
    # # ======= maybe we need to modify this part? ======= # 
    # def __iter__(self):
    #     self.n_view = random.choice(self.n_views)
    #     return super().__iter__()

