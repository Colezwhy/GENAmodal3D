import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import argparse
from pathlib import Path

import numpy as np
import imageio
import cv2
import trimesh
from PIL import Image

from gena3d.pipelines import SparseImageTo3D
from gena3d.utils import render_utils, postprocessing_utils

# ----------------------------------------------------------------------------- #
# The full, self-contained pipeline (sparse-structure / SLAT decoders + our
# fine-tuned flow models + pipeline config) is hosted on the Hugging Face Hub at
# HF_REPO_ID. It is downloaded automatically on first run. See
# publish_model_to_hf.py for how this repo is produced.
# ----------------------------------------------------------------------------- #
HF_REPO_ID = "Colezwhy/GENA3D"

# Standard file names expected inside every input example directory.
AMODAL_NAME = "amodal_completion.png"      # amodal-completed (inpainted) RGB
OCCLUDED_NAME = "sd_img_cut.png"           # cropped occluded RGB
VISIBILITY_NAME = "visibility_instance_mask.png"
OCC_MASK_NAME = "occ_mask0.png"


def extract_glb(gs, mesh, mesh_simplify=0.95, texture_size=1024, export_path="output.glb"):
    """
    Extract a GLB file from the 3D model.

    Args:
        gs: The generated gaussian representation.
        mesh: The generated mesh.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb.export(export_path)
    return export_path


def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs

    mesh.export(filename)


def build_pipeline(hf_repo_id=HF_REPO_ID):
    """Build the SparseImageTo3D pipeline fully from the GENA3D Hub repo.

    The repo is self-contained: its pipeline.json references our fine-tuned
    sparse-structure / SLAT flow models (with the matching architecture configs)
    alongside the unchanged decoders, so no Amodal3R weights are needed.
    """
    pipeline = SparseImageTo3D.from_pretrained_with_custom(hf_repo_id)
    pipeline.cuda()
    return pipeline


def load_inputs(input_dirs):
    """Each input dir must contain the four standard files (see *_NAME constants)."""
    images, occluded, visibility, occ_mask = [], [], [], []
    for d in input_dirs:
        d = Path(d)
        images.append(Image.open(d / AMODAL_NAME))
        occluded.append(Image.open(d / OCCLUDED_NAME))
        visibility.append(Image.open(d / VISIBILITY_NAME))
        occ_mask.append(Image.open(d / OCC_MASK_NAME))
    return images, occluded, visibility, occ_mask


def has_input_set(d):
    """True if directory `d` directly contains the four required files."""
    d = Path(d)
    return all((d / n).exists() for n in (AMODAL_NAME, OCCLUDED_NAME, VISIBILITY_NAME, OCC_MASK_NAME))


def resolve_views(demo_dir):
    """Resolve the view directories inside a single demo folder.

    A demo folder is either:
      * single-view  -> the four files sit directly in the folder, or
      * multi-view   -> one sub-directory per view, each holding the four files.
    The views are fused into a single reconstruction.
    """
    demo_dir = Path(demo_dir)
    if has_input_set(demo_dir):
        return [demo_dir]
    return [d for d in sorted(demo_dir.iterdir()) if d.is_dir() and has_input_set(d)]


def discover_demos(root="examples"):
    """Return the demo folders (direct children of `root`) that hold a valid input set."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [str(d) for d in sorted(root.iterdir()) if d.is_dir() and resolve_views(d)]


def parse_args():
    parser = argparse.ArgumentParser(description="GENA3D amodal 3D reconstruction inference.")
    parser.add_argument("--input", default=None,
                        help="A single demo folder. Either contains the four files "
                             f"({AMODAL_NAME}, {OCCLUDED_NAME}, {VISIBILITY_NAME}, {OCC_MASK_NAME}) "
                             "directly (single view), or one sub-folder per view (the views are "
                             "fused into one reconstruction). If omitted, the first demo under "
                             "examples/ is used.")
    parser.add_argument("--output_dir", default="./output/demo",
                        help="Directory to write the rendered video / images / meshes.")
    parser.add_argument("--hf_repo_id", default=HF_REPO_ID,
                        help="Hugging Face repo hosting the GENA3D fine-tuned weights.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_cfg", type=float, default=7.5)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_cfg", type=float, default=3.0)
    parser.add_argument("--mesh_simplify", type=float, default=0.5)
    parser.add_argument("--texture_size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    demo = args.input
    if not demo:
        demos = discover_demos()
        if not demos:
            raise SystemExit("No demo folders found under examples/. Pass --input explicitly.")
        demo = demos[0]
        print(f"No --input given; using demo: {demo}")

    views = resolve_views(demo)
    if not views:
        raise SystemExit(
            f"'{demo}' is not a valid demo folder: it must contain the four files directly "
            "or one sub-folder per view."
        )
    print(f"Running demo '{Path(demo).name}' with {len(views)} view(s): {[str(v) for v in views]}")

    output_dir = Path(args.output_dir) / Path(demo).name
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(args.hf_repo_id)
    images, occluded, visibility, occ_mask = load_inputs(views)

    # Run the pipeline
    outputs = pipeline.run(
        images,
        occluded,
        mask=visibility,
        occ_mask=occ_mask,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_cfg,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_cfg,
        },
    )

    # save as gif
    video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0], bg_color=(1, 1, 1))['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
    imageio.mimsave(str(output_dir / "sample_multi.gif"), video, fps=30)

    # save multi-view gs
    gaussian = outputs['gaussian'][0]
    multi_view_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1), pitch=0)
    multi_view_gs = multi_view_gs['color']
    for i in range(8):
        output = cv2.cvtColor(multi_view_gs[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{i:03d}_gs.png"), output)

    mesh = outputs['mesh'][0]

    # save mesh
    save_mesh(mesh, str(output_dir / "mesh.ply"))

    multi_view_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
    multi_view_mesh = multi_view_mesh['normal']
    for i in range(8):
        output = cv2.cvtColor(multi_view_mesh[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{i:03d}_mesh.png"), output)

    # export glb
    glb_path = output_dir / "mesh.glb"
    extract_glb(outputs['gaussian'][0], outputs['mesh'][0], args.mesh_simplify, args.texture_size, str(glb_path))

    print(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    main()
