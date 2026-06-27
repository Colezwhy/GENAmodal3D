import torch, math
import numpy as np
from sklearn.cluster import KMeans
import re
import cv2

def init_volume_grid(bound=0.45, num_pts_each_axis=32):
    # Define the range and number of points  
    start = -bound
    stop = bound
    num_points = num_pts_each_axis  # Adjust the number of points to your preference  
    
    # Create a linear space for each axis  
    x = np.linspace(start, stop, num_points)  
    y = np.linspace(start, stop, num_points)  
    z = np.linspace(start, stop, num_points)  
    
    # Create a 3D grid of points using meshgrid  
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    
    # Stack the grid points in a single array of shape (N, 3)  
    xyz = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  
    
    return xyz

def build_rays_torch(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale
    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    XYZ = torch.cat((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, torch.ones_like(X[:, :, None])), dim=-1).to(c2ws)
    
    i2ws = torch.inverse(ixts).permute(0,2,1) @ c2ws[:,:3, :3].permute(0,2,1)
    XYZ = torch.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(1,H,1,1)
    rays_o = rays_o.repeat(1,1,W,1)
    rays = torch.cat((rays_o, XYZ), dim=-1)
    return rays

def build_rays(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale

    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, np.ones_like(X[:, :, None])), axis=-1)
    i2ws = np.linalg.inv(ixts).transpose(0,2,1) @ c2ws[:,:3, :3].transpose(0,2,1)
    XYZ = np.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(H, axis=1)
    rays_o = rays_o.repeat(W, axis=2)
    rays = np.concatenate((rays_o, XYZ), axis=-1)
    return rays.astype(np.float32)

def build_rays_ortho(c2ws, H, W, scale=1.0):

    c2ws_rot = c2ws[:,:3,:3]
    c2ws_t = c2ws[:,:3,3].reshape(-1,1,3)
            
    rays_d = torch.zeros(1,1,3).to(c2ws)
    rays_d[...,-1] = 1.0
    rays_d = rays_d @ c2ws_rot.transpose(1,2)
    rays_d = rays_d[:,None].expand(-1,H,W,-1)
            
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = torch.from_numpy(X[:, :, None] + 0.5).float()/W * 2 - 1.0
    Y = torch.from_numpy(Y[:, :, None] + 0.5).float()/H * 2 - 1.0
    XYZ = torch.cat((X*scale, Y*scale, torch.zeros_like(X)), dim=-1).to(c2ws)
    XYZ = XYZ.view(1,-1,3)
    rays_o = XYZ @ c2ws_rot.transpose(1,2) + c2ws_t
    rays = torch.cat((rays_o.view(rays_d.shape), rays_d), dim=-1)
    return rays

def KMean(xyz, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init=10, random_state=20211202)
    kmeans.fit(xyz)
    labels = kmeans.labels_
    
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels==i)[0]
        clusters.append(idx.astype(np.uint8))

    return clusters

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def intrinsic_to_fov(K, w=None, h=None):
    # Extract the focal lengths from the intrinsic matrix
    fx = K[0, 0]
    fy = K[1, 1]
    
    w = K[0, 2]*2 if w is None else w
    h = K[1, 2]*2 if h is None else h
    
    # Calculating field of view
    fov_x = 2 * np.arctan2(w, 2 * fx) 
    fov_y = 2 * np.arctan2(h, 2 * fy)
    
    return fov_x, fov_y

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def pose_sub_selete(poses, N_poses):
    # to select N_poses poses that close to the equatorial
    # Define the camera's local up-vector and the world's up-vector
    camera_up_vector = np.array([0, 1, 0])
    world_up_vector = np.array([1, 0, 0])

    # Extract the rotation matrices from the transformation matrices
    rotations = poses[:, :3, :3]  # Shape: [num_poses, 3, 3]

    # Transform the camera's up-vector to world coordinates for all poses
    camera_up_world = np.einsum('ijk,k->ij', rotations, camera_up_vector)  # Shape: [num_poses, 3]

    # Normalize the transformed up-vectors
    camera_up_world_norm = camera_up_world / np.linalg.norm(camera_up_world, axis=1)[:, np.newaxis]

    # Calculate the dot product with the world up-vector to find cosines of angles
    cos_angles = np.dot(camera_up_world_norm, world_up_vector)

    # Calculate angles in radians using arccos
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    # Select the poses with the smallest angles (closest to the equatorial plane)
    indices = np.argsort(angles)[:N_poses]
    
    return indices

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



def generate_mae_mask(image_shape=(3,512,512), patch_size=16, mask_ratio=0.3):
    _, height, width = image_shape
    
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_patches = num_patches_h * num_patches_w
    
    num_masked_patches = int(total_patches * mask_ratio)
    
    mask = np.ones(total_patches, dtype=np.float32)
    mask[:num_masked_patches] = 0
    
    np.random.shuffle(mask)
    
    mask = mask.reshape(num_patches_h, num_patches_w)
    
    mask = np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)
    
    return mask


def generate_random_mask(width=512, height=512):
    """Generates a stable mask with a fewer number of coherent blocks."""

    img = np.zeros((width, height, 3), np.uint8)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random lines (reduced count)
    for _ in range(np.random.randint(1, 3)):
        x1, x2 = np.random.randint(1, width), np.random.randint(1, width)
        y1, y2 = np.random.randint(1, height), np.random.randint(1, height)
        thickness = np.random.randint(3, size)
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    # Draw random circles (reduced count)
    for _ in range(np.random.randint(1, 3)):
        x1, y1 = np.random.randint(1, width), np.random.randint(1, height)
        radius = np.random.randint(3, size)
        cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

    # Draw random ellipses (reduced count)
    for _ in range(np.random.randint(1, 3)):
        x1, y1 = np.random.randint(1, width), np.random.randint(1, height)
        s1, s2 = np.random.randint(3, size), np.random.randint(3, size)
        a1, a2 = np.random.randint(0, 180), np.random.randint(0, 360)
        thickness = np.random.randint(3, size)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, 0, a2, (1, 1, 1), thickness)

    # Add coherent block masks (reduced number of blocks)
    num_blocks = np.random.randint(3, 7)
    block_size_range = (25, 75)  # Larger block sizes for stability
    for _ in range(num_blocks):
        block_height = np.random.randint(*block_size_range)
        block_width = np.random.randint(*block_size_range)

        start_x = np.random.randint(0, max(0, width - block_width))
        start_y = np.random.randint(0, max(0, height - block_height))

        img[start_y:start_y + block_height, start_x:start_x + block_width] = (1, 1, 1)

        # Dilate the block for smoother edges
        kernel_size = np.random.randint(5, 9)  # Slightly larger kernel for stability
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img = cv2.dilate(img, kernel, iterations=1)

    return img[:, :, 0:1]