import os
import shutil
import tqdm

source = '/jumbo/yuwingtai/junwei/GSO/GSO-model'

for obj_name in tqdm(os.listdir(source), desc="Copying textures"):
    old_tex = os.path.join(source, obj_name, 'materials', 'textures', 'texture.png')
    new_tex = os.path.join(source, obj_name, 'meshes', 'texture.png')

    if not os.path.exists(old_tex):
        continue  # 跳过无纹理对象

    if not os.path.exists(os.path.join(source, obj_name, 'meshes')):
        continue  # 跳过无纹理对象
    
    if os.path.exists(new_tex):
        continue  # 跳过无纹理对象

    try:
        shutil.copy2(old_tex, new_tex)
        print(f"Copied texture for {obj_name}")
    except Exception as e:
        print(f"❌ Failed to copy for {obj_name}: {e}")
