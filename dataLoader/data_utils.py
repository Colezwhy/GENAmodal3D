from typing import *
import math
import torch
import numpy as np
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import os.path as osp
from torchvision import transforms
import cv2
from PIL import Image
import os
from plyfile import PlyData, PlyElement
import open3d as o3d

def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image. Automatically moves the channel dimension 
    (if it has size 3) to the last axis before converting.

    Args:
        tensor (torch.Tensor): Input tensor. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor

    return array_to_pil(array)


def array_to_pil(array):
    """
    Converts a NumPy array to a PIL image. Automatically:
        - Squeezes dimensions of size 1.
        - Moves the channel dimension (if it has size 3) to the last axis.
    
    Args:
        array (np.ndarray): Input array. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    # Remove singleton dimensions
    array = np.squeeze(array)
    
    # Ensure the array has the channel dimension as the last axis
    if array.ndim == 3 and array.shape[0] == 3:  # If the channel is the first axis
        array = np.transpose(array, (1, 2, 0))  # Move channel to the last axis
    
    # Handle single-channel grayscale images
    if array.ndim == 2:  # [H, W]
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:  # [H, W, C] with 3 channels
        return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for PIL conversion: {array.shape}")


def rotate_target_dim_to_last_axis(x, target_dim=3):
    shape = x.shape
    axis_to_move = -1
    # Iterate backwards to find the first occurrence from the end 
    # (which corresponds to the last dimension of size 3 in the original order).
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == target_dim:
            axis_to_move = i
            break

    # 2. If the axis is found and it's not already in the last position, move it.
    if axis_to_move != -1 and axis_to_move != len(shape) - 1:
        # Create the new dimension order.
        dims_order = list(range(len(shape)))
        dims_order.pop(axis_to_move)
        dims_order.append(axis_to_move)
        
        # Use permute to reorder the dimensions.
        ret = x.transpose(*dims_order)
    else:
        ret = x

    return ret

def voxelize(
    xyz,
    rgb=None,
    min_bound=(-0.5,-0.5,-0.5), 
    max_bound=(0.5,0.5,0.5),
    voxel_size = 1/64,          
    radius_outlier = 1/64,
    min_neighbors = 5,
    ) -> None:
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)
    
    if rgb is None:
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)
        
        hue = 0.7 * normalized_coord[:,0] + 0.2 * normalized_coord[:,1] + 0.1 * normalized_coord[:,2]
        hsv = np.stack([hue, 0.9*np.ones_like(hue), 0.8*np.ones_like(hue)], axis=1)

        c = hsv[:,2:] * hsv[:,1:2]
        x = c * (1 - np.abs( (hsv[:,0:1]*6) % 2 - 1 ))
        m = hsv[:,2:] - c
        
        rgb = np.zeros_like(hsv)
        cond = (0 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 1)
        rgb[cond] = np.hstack([c[cond], x[cond], np.zeros_like(x[cond])])
        cond = (1 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 2)
        rgb[cond] = np.hstack([x[cond], c[cond], np.zeros_like(x[cond])])
        cond = (2 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), c[cond], x[cond]])
        cond = (3 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), x[cond], c[cond]])
        cond = (4 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 5)
        rgb[cond] = np.hstack([x[cond], np.zeros_like(x[cond]), c[cond]])
        cond = (5 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x[cond]), x[cond]])
        rgb = (rgb + m)

    # ====== clamping ======
    xyz = np.clip(xyz, np.array(min_bound)+1e-6, np.array(max_bound)-1e-6)

    # ====== create pc ======
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if radius_outlier is not None:
        pcd, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius_outlier)
    
    # ====== voxelization ======
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=voxel_size,
        min_bound=min_bound,
        max_bound=max_bound
    )

    # ====== extract center, normalize ======
    voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    voxels = (voxels + 0.5) / (1/voxel_size) - 0.5
    return voxels


def filter_by_depth(rgb_img, depth_img, instance_mask):
    """
    Find foreground regions
    """
    rgb = np.array(rgb_img)
    depth = np.array(depth_img)
    mask = np.array(instance_mask)

    mask = (mask > 0).astype(np.uint8)

    if depth.dtype != np.float32 and depth.dtype != np.float64:
        depth = depth.astype(np.float32)
    valid_depth = depth[mask > 0]
    if len(valid_depth) == 0:
        raise ValueError("No valid pixels in depth map")
    min_depth = valid_depth.min()
    result = rgb.copy()
    condition = (mask == 0) & (depth < min_depth)
    result[condition] = 0
    return Image.fromarray(result)


def img_pil_to_tensor(image, interval=1, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if isinstance(list, image):
        for i in range(0, len(image), interval):
            try:
                sources.append(image[i].convert('RGB'))
            except Exception as e:
                print(f"Could not transfer image.")
    else:
        raise ValueError(f"Unsupported variable. Must be a list of PIL Image.")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


# ===== Loading input binary masks ===== #
def mask_pil_to_tensor(mask, interval=1, PIXEL_LIMIT=255000, threshold=0.1):
    """
    Loads binary mask images (background black, foreground non-black) 
    from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, H, W] PyTorch bool tensor.
    """
    sources = []
    
    # --- 1. Load mask paths or video frames ---
    if isinstance(list, mask):
        for i in range(0, len(mask), interval):
            try:
                sources.append(mask[i].convert('L'))  # To grayscale
            except Exception as e:
                print(f"Could not load mask")
    else:
        raise ValueError(f"Unsupported path. Must be a list of PIL.Image binary mask.")

    if not sources:
        print("No masks found or loaded.")
        return torch.empty(0, dtype=torch.bool)

    # --- 2. Determine a uniform target size based on the first mask ---
    first_mask = sources[0]
    W_orig, H_orig = first_mask.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    # --- 3. Resize masks and convert to bool tensor ---
    tensor_list = []
    to_tensor_transform = transforms.ToTensor()

    for mask_pil in sources:
        try:
            resized_mask = mask_pil.resize((TARGET_W, TARGET_H), Image.Resampling.NEAREST)
            mask_tensor = to_tensor_transform(resized_mask)  # [1,H,W]
            mask_tensor = (mask_tensor.squeeze(0) > threshold)
            tensor_list.append(mask_tensor)
        except Exception as e:
            print(f"Error processing a mask: {e}")

    if not tensor_list:
        print("No masks were successfully processed.")
        return torch.empty(0, dtype=torch.bool)

    # --- 4. Stack tensors into [N, H, W] ---
    return torch.stack(tensor_list, dim=0)

def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


# ===== Loading input binary masks ===== #
def load_binary_masks_as_tensor(path='data/masks', interval=1, PIXEL_LIMIT=255000, threshold=0.1):
    """
    Loads binary mask images (background black, foreground non-black) 
    from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, H, W] PyTorch bool tensor.
    """
    sources = []
    
    # --- 1. Load mask paths or video frames ---
    if osp.isdir(path):
        print(f"Loading masks from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('L'))  # 灰度图
            except Exception as e:
                print(f"Could not load mask {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading mask frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): 
            raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sources.append(Image.fromarray(gray_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No masks found or loaded.")
        return torch.empty(0, dtype=torch.bool)

    print(f"Found {len(sources)} masks/frames. Processing...")

    # --- 2. Determine a uniform target size based on the first mask ---
    first_mask = sources[0]
    W_orig, H_orig = first_mask.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All masks will be resized to: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize masks and convert to bool tensor ---
    tensor_list = []
    to_tensor_transform = transforms.ToTensor()  # 变成 [1, H, W], float [0,1]

    for mask_pil in sources:
        try:
            resized_mask = mask_pil.resize((TARGET_W, TARGET_H), Image.Resampling.NEAREST)
            mask_tensor = to_tensor_transform(resized_mask)  # [1,H,W]
            mask_tensor = (mask_tensor.squeeze(0) > threshold)  # 去掉通道维度 -> [H,W], bool
            tensor_list.append(mask_tensor)
        except Exception as e:
            print(f"Error processing a mask: {e}")

    if not tensor_list:
        print("No masks were successfully processed.")
        return torch.empty(0, dtype=torch.bool)

    # --- 4. Stack tensors into [N, H, W] ---
    return torch.stack(tensor_list, dim=0)  # bool 类型

def recursive_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = False,
) -> Any:
    """
    Recursively move all tensors in a data structure to a device.
    """
    if hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        return data


def load_balanced_group_indices(
    load: List[int],
    num_groups: int,
    equal_size: bool = False,
) -> List[List[int]]:
    """
    Split indices into groups with balanced load.
    """
    if equal_size:
        group_size = len(load) // num_groups
    indices = np.argsort(load)[::-1]
    groups = [[] for _ in range(num_groups)]
    group_load = np.zeros(num_groups)
    for idx in indices:
        min_group_idx = np.argmin(group_load)
        groups[min_group_idx].append(idx)
        if equal_size and len(groups[min_group_idx]) == group_size:
            group_load[min_group_idx] = float('inf')
        else:
            group_load[min_group_idx] += load[idx]
    return groups


def cycle(data_loader: DataLoader) -> Iterator:
    while True:
        for data in data_loader:
            if isinstance(data_loader.sampler, ResumableSampler):
                data_loader.sampler.idx += data_loader.batch_size   # type: ignore[attr-defined]
            yield data
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.epoch += 1
        if isinstance(data_loader.sampler, ResumableSampler):
            data_loader.sampler.epoch += 1
            data_loader.sampler.idx = 0
        

class ResumableSampler(Sampler):
    """
    Distributed sampler that is resumable.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.idx = 0
        self.drop_last = drop_last
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        
        # resume from previous state
        indices = indices[self.idx:]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, int]:
        return {
            'epoch': self.epoch,
            'idx': self.idx,
        }
        
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.idx = state_dict['idx']
        

class BalancedResumableSampler(ResumableSampler):
    """
    Distributed sampler that is resumable and balances the load among the processes.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ) -> None:
        assert hasattr(dataset, 'loads'), 'Dataset must have "loads" attribute to use BalancedResumableSampler'
        super().__init__(dataset, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.loads = dataset.loads
        
    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # balance load among processes
        num_batches = len(indices) // (self.batch_size * self.world_size)
        balanced_indices = []
        for i in range(num_batches):
            start_idx = i * self.batch_size * self.world_size
            end_idx = (i + 1) * self.batch_size * self.world_size
            batch_indices = indices[start_idx:end_idx]
            batch_loads = [self.loads[idx] for idx in batch_indices]
            groups = load_balanced_group_indices(batch_loads, self.world_size, equal_size=True)
            balanced_indices.extend([batch_indices[j] for j in groups[self.rank]])
        
        # resume from previous state
        indices = balanced_indices[self.idx:]

        return iter(indices)
