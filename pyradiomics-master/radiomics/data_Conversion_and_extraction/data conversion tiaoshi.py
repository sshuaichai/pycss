import numpy as np
import nrrd
import nibabel as nib
import os

def convert_nrrd_to_nifti_correct_orientation(nrrd_path, nifti_path):
    # 读取NRRD图像和头信息
    data, header = nrrd.read(nrrd_path)

    # 尝试从header中提取space directions，如果失败则使用默认仿射矩阵
    try:
        space_directions = np.array(header['space directions'])
        if space_directions.shape == (3, 3):
            affine = np.eye(4)
            affine[:3, :3] = space_directions
        else:
            print(f"Warning: space directions shape is not 3x3 in {nrrd_path}. Using default affine matrix.")
            affine = np.eye(4)
    except KeyError:
        print(f"Warning: 'space directions' not found in {nrrd_path}. Using default affine matrix.")
        affine = np.eye(4)

    # 创建NIfTI图像并保存
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, nifti_path)
    print(f"Converted and saved: {nifti_path}")

# 示例用法
nrrd_folder = r"D:\zhuomian\140\tiaoshi_img"
nifti_folder = r"D:\zhuomian\140\tiaoshi_nii"
if not os.path.exists(nifti_folder):
    os.makedirs(nifti_folder)

for filename in os.listdir(nrrd_folder):
    if filename.endswith(".nrrd"):
        nrrd_path = os.path.join(nrrd_folder, filename)
        nifti_filename = filename.replace('.nrrd', '.nii.gz')
        nifti_path = os.path.join(nifti_folder, nifti_filename)
        convert_nrrd_to_nifti_correct_orientation(nrrd_path, nifti_path)
