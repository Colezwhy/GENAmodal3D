import open3d as o3d
import numpy as np
import json
from PIL import Image
import random
import os
from tqdm import tqdm

seed = 42
random.seed(seed)  # set random seed for reproducibility
np.random.seed(seed)  # set random seed for reproducibility

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt


# construct adjacency list for triangles
def build_adjacency_list(triangles):
    adjacency_list = {i: set() for i in range(len(triangles))}
    edges_to_faces = {}
    for face_idx, triangle in enumerate(triangles):
        edges = [
            tuple(sorted((triangle[0], triangle[1]))),
            tuple(sorted((triangle[1], triangle[2]))),
            tuple(sorted((triangle[2], triangle[0]))),
        ]
        for edge in edges:
            if edge not in edges_to_faces:
                edges_to_faces[edge] = []
            edges_to_faces[edge].append(face_idx)
    
    for edge, faces in edges_to_faces.items():
        if len(faces) > 1:
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    adjacency_list[faces[i]].add(faces[j])
                    adjacency_list[faces[j]].add(faces[i])
    return adjacency_list


# use random walk to select connected faces
def select_connected_faces(adjacency_list, num_faces):
    visited = set()
    selected_faces = []
    start_face = np.random.choice(list(adjacency_list.keys()))
    queue = [start_face]
    while queue and len(selected_faces) < num_faces:
        face = queue.pop(0)
        if face not in visited:
            visited.add(face)
            selected_faces.append(face)
            queue.extend(list(adjacency_list[face] - visited))  # add unvisited connected faces
    return selected_faces

data_root = "./test" # The evaluation data folder. It should include 'renders' folder under it, which contains the rendered images, mesh.ply and transforms.json files
cond_root  = os.path.join(data_root, "renders")
mesh_root = os.path.join(data_root, "renders")
output_dir = os.path.join(data_root, "renders_mask") # The output folder for the masked images
os.makedirs(output_dir, exist_ok=True)

scene_list = os.listdir(mesh_root)

render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultUnlit"

for scene in scene_list:
    print("processing scene: ", scene)
    mesh_path = os.path.join(mesh_root, scene, "mesh.ply")
    camera_path = os.path.join(cond_root, scene, "transforms.json")
    os.makedirs(os.path.join(output_dir, scene), exist_ok=True)

    # read mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)


    # randomly set 10% of the faces to be black
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # create a color array with the same number of vertices
    vertex_colors = np.ones((len(vertices), 3)) * 0.5

    adjacency_list = build_adjacency_list(triangles)

    # randomly select a portion of triangles
    num_faces = len(triangles)
    mask_ratio = np.random.uniform(0.2, 0.5)
    num_black_faces = int(num_faces * mask_ratio)  # set 20% to 50% of faces to be black
    selected_faces_set = set()

    while len(selected_faces_set) < num_black_faces:
        # use random walk to select connected faces
        new_faces = select_connected_faces(adjacency_list, num_black_faces - len(selected_faces_set))
        # add newly selected faces to the set to avoid duplicates
        selected_faces_set.update(new_faces)
        
    black_face_indices = list(selected_faces_set)


    # set selected triangle face vertex colors to black
    black_vertex_indices = np.unique(triangles[black_face_indices].flatten())
    vertex_colors[black_vertex_indices] = [0, 0, 0]  # set to black

    # update Mesh vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # read cameras
    with open(camera_path, "r") as f:
        camera_info = json.load(f)
        
    b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])    
    cameras = []
    
    render.scene.add_geometry("mesh", mesh, material)

    for idx, frame in enumerate(camera_info["frames"]):
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        fov = frame["camera_angle_x"]
        ixt = fov_to_ixt(np.array([fov, fov]), np.array([512, 512]))
        intrinsic.set_intrinsics(width=512, height=512, fx=ixt[0,0], fy=ixt[1,1], cx=ixt[0,2], cy=ixt[1,2])
        
        c2w = frame["transform_matrix"]
        c2w = c2w @ b2c
        w2c = np.linalg.inv(c2w)
        
        render.setup_camera(intrinsic, w2c)
        image = render.render_to_image()

        # save image
        image_np = np.asarray(image)
        rgb_image = Image.fromarray(image_np)
        rgb_image.save(os.path.join(output_dir, scene, f"{idx:03d}.png"))
        
    render.scene.remove_geometry("mesh")