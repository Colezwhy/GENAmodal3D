import os
import argparse
import pandas as pd
import json

def add_args(parser: argparse.ArgumentParser):
    pass
    # parser.add_argument('--data_root', type=str, required=True,
    #                     help='Path to GSO dataset root containing mesh files')
    # parser.add_argument('--captions_json', type=str, default=None,
    #                     help='Optional JSON file containing captions for meshes')

def get_metadata(**kwargs):
    """
    根据 GSO 数据集生成 metadata.csv，列名和现有 CSV 对齐
    """
    data_root = kwargs['output_dir']
    captions_json = kwargs.get('captions_json', None)

    # 读取 captions，如果有提供 JSON 文件
    captions_dict = {}
    if captions_json is not None:
        with open(captions_json, 'r') as f:
            captions_dict = json.load(f)

    records = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith(".obj") or f.endswith(".ply"):
                # rint(root, f)
                # mesh_path = os.path.join(root, f)
                mesh_path = os.path.join(root)
                # sha256 用文件名生成（可替换为真正的 hash）
                sha256 = os.path.basename(os.path.dirname(root))
                file_identifier = os.path.relpath(mesh_path, start=data_root).replace("\\", "/")
                captions = captions_dict.get(file_identifier, [])
                fi = os.path.join('/jumbo/yuwingtai/junwei/GSO', file_identifier)
                if os.path.exists(os.path.join(fi, 'model.obj')) and os.path.exists(os.path.join(fi, 'model.mtl')) and os.path.exists(os.path.join(fi, 'texture.png')):
                    records.append({
                        'sha256': sha256,
                        'file_identifier': file_identifier,
                        'aesthetic_score': 1.0,
                        'captions': json.dumps(captions, ensure_ascii=False),
                        'rendered': False,
                        'voxelized': False,
                        'num_voxels': 0,
                        'cond_rendered': False,
                        'local_path': os.path.join(file_identifier, 'model.obj'),
                        'feature_dinov2_vitl14_reg': False,
                        'ss_latent_ss_enc_conv3d_16l8_fp16': False
                    })

    metadata = pd.DataFrame(records)
    return metadata

def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    metadata = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
             tqdm(total=len(metadata), desc=desc) as pbar:

            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    mesh_id = metadatum['sha256']
                    file = os.path.join(output_dir, local_path)

                    record = func(file, mesh_id)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {mesh_id}: {e}")
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")

    return pd.DataFrame.from_records(records)
