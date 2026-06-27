"""Publish a *self-contained* GENA3D pipeline to the Hugging Face Hub.

The resulting repo has no runtime dependency on Sm0kyWu/Amodal3R: it bundles
  - pipeline.json (samplers / normalization / image-cond config),
  - the 3 unchanged decoders (pulled once from the Amodal3R repo),
  - our two fine-tuned flow models (your local weights + matching config JSONs).

After this, inference_gena3d.py can load everything via
    SparseImageTo3D.from_pretrained_with_custom("Colezwhy/GENA3D")

Usage:
    hf auth login                      # token needs *write* access
    python publish_model_to_hf.py --repo-id Colezwhy/GENA3D

Foundation models (DINOv2 via torch.hub, Pi3 = yyfz233/Pi3, the SS VAE encoder =
microsoft/TRELLIS-image-large) are still fetched from their original homes at
runtime; they are not part of this repo.
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import create_repo, snapshot_download, upload_folder

# Source of the unchanged decoders + base pipeline config.
BASE_REPO = "Sm0kyWu/Amodal3R"
BASE_SUBDIR = "Amodal3R_ckpts"

# Local weights to publish (override on the CLI if your paths differ).
DEFAULT_SS_LOCAL = "/jumbo/yuwingtai/junwei/3Dgen/Amodal3R/train_sparse_strcuture_full_train_12epoch_abo_3df/ss_stereo_full_train_abo_3df.safetensors"
DEFAULT_SLAT_LOCAL = "/jumbo/yuwingtai/junwei/3Dgen/Amodal3R/ckpts/slat_flow_img_dit_L_64l8p2_fp16_doubleattn_weighted.safetensors"

# Layout of the published repo.
DST_SUBDIR = "ckpts"
DECODERS = {  # pipeline-key -> file basename (unchanged, copied from BASE_REPO)
    "sparse_structure_decoder": "ss_dec_conv3d_16l8_fp16",
    "slat_decoder_gs": "slat_dec_gs_swin8_B_64l8gs32_fp16",
    "slat_decoder_mesh": "slat_dec_mesh_swin8_B_64l8m256c_fp16",
}
SS_FLOW_BASENAME = "ss_flow_multibranch_gate_abo3df"
SLAT_FLOW_BASENAME = "slat_flow_doubleattn_weighted"

# Architecture configs for the two fine-tuned flow models. These MUST match the
# constructor args the weights were trained with (verified to load strict=True).
SS_FLOW_CONFIG = {
    "name": "SparseStructureFlowMultiBranchModel",
    "args": {
        "resolution": 16, "in_channels": 8, "out_channels": 8,
        "model_channels": 1024, "cond_channels": 1024, "num_blocks": 24,
        "num_heads": 16, "mlp_ratio": 4, "patch_size": 1, "pe_mode": "ape",
        "qk_rms_norm": True, "use_fp16": False, "use_gate": True,
    },
}
SLAT_FLOW_CONFIG = {
    "name": "SLatFlowModelMaskAsCondWeighted",
    "args": {
        "resolution": 64, "in_channels": 8, "out_channels": 8,
        "model_channels": 1024, "cond_channels": 1024, "num_blocks": 24,
        "num_heads": 16, "mlp_ratio": 4, "patch_size": 2,
        "num_io_res_blocks": 2, "io_block_channels": [128], "pe_mode": "ape",
        "qk_rms_norm": True, "use_fp16": False, "mask_cond_type": "mask_patcher",
        "view_wise": False,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Publish a self-contained GENA3D pipeline to the HF Hub.")
    parser.add_argument("--repo-id", default="Colezwhy/GENA3D")
    parser.add_argument("--ss-local", default=DEFAULT_SS_LOCAL)
    parser.add_argument("--slat-local", default=DEFAULT_SLAT_LOCAL)
    parser.add_argument("--staging", default="./hf_staging", help="Local dir to assemble the repo before upload.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Assemble locally but do not create/upload the repo.")
    args = parser.parse_args()

    staging = Path(args.staging)
    dst_ckpts = staging / DST_SUBDIR
    dst_ckpts.mkdir(parents=True, exist_ok=True)

    # 1) Pull only the decoders + base pipeline.json (skip the big base flow weights).
    print(f"Fetching decoders + pipeline.json from {BASE_REPO} ...")
    allow = ["pipeline.json"] + [f"{BASE_SUBDIR}/{b}.*" for b in DECODERS.values()]
    base_dir = Path(snapshot_download(BASE_REPO, allow_patterns=allow))

    # 2) Copy the unchanged decoders.
    models_map = {}
    for key, basename in DECODERS.items():
        for ext in (".json", ".safetensors"):
            shutil.copyfile(base_dir / BASE_SUBDIR / f"{basename}{ext}", dst_ckpts / f"{basename}{ext}")
        models_map[key] = f"{DST_SUBDIR}/{basename}"
        print(f"  decoder: {key} -> {DST_SUBDIR}/{basename}")

    # 3) Add the two fine-tuned flow models (weights + config JSON).
    for key, basename, cfg, local in [
        ("sparse_structure_flow_model", SS_FLOW_BASENAME, SS_FLOW_CONFIG, args.ss_local),
        ("slat_flow_model", SLAT_FLOW_BASENAME, SLAT_FLOW_CONFIG, args.slat_local),
    ]:
        (dst_ckpts / f"{basename}.json").write_text(json.dumps(cfg, indent=4))
        shutil.copyfile(local, dst_ckpts / f"{basename}.safetensors")
        models_map[key] = f"{DST_SUBDIR}/{basename}"
        print(f"  flow:    {key} -> {DST_SUBDIR}/{basename}  (from {local})")

    # 4) Write pipeline.json: keep samplers / normalization / image_cond, swap the models map.
    base_pipeline = json.loads((base_dir / "pipeline.json").read_text())
    base_pipeline["name"] = "GENA3DImageTo3DPipeline"
    base_pipeline["args"]["models"] = models_map
    (staging / "pipeline.json").write_text(json.dumps(base_pipeline, indent=4))
    print(f"Assembled repo at {staging}")

    if args.dry_run:
        print("Dry run: skipping repo creation/upload.")
        return

    # 5) Create + upload.
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    upload_folder(folder_path=str(staging), repo_id=args.repo_id, repo_type="model")
    print(f"Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
