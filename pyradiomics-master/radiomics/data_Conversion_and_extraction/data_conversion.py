import numpy as np
import nrrd
import nibabel as nib
import os

def convert_nrrd_to_nifti_correct_orientation(nrrd_path, nifti_path):
    # 读取NRRD图像和头信息
    data, header = nrrd.read(nrrd_path)

    # 从header中提取space directions和space origin
    space_directions = np.array(header.get('space directions', np.eye(3)))
    if space_directions.shape != (3, 3):
        # 如果space_directions不是3x3的矩阵，采取适当的错误处理或警告
        print(f"Warning: space directions shape is not 3x3 in {nrrd_path}")
        return

    affine = np.eye(4)
    affine[:3, :3] = space_directions  # 应用空间方向

    # 可选：根据需要调整仿射矩阵中的符号
    affine[0, 0] *= -1
    affine[1, 1] *= -1

    # 尝试创建NIfTI图像
    try:
        nifti_img = nib.Nifti1Image(data, affine)
        # 保存NIfTI图像
        nib.save(nifti_img, nifti_path)
        print(f"Converted and saved: {nifti_path}")
    except Exception as e:
        print(f"Error converting {nrrd_path} to NIfTI: {e}")

# 示例用法
nrrd_folder = r"D:\zhuomian\labelsTr2"
nifti_folder = r"D:\zhuomian\labelsTr2_nii"
if not os.path.exists(nifti_folder):
    os.makedirs(nifti_folder)

for filename in os.listdir(nrrd_folder):
    if filename.endswith(".nrrd"):
        nrrd_path = os.path.join(nrrd_folder, filename)
        nifti_filename = filename.replace('.nrrd', '.nii.gz')
        nifti_path = os.path.join(nifti_folder, nifti_filename)
        convert_nrrd_to_nifti_correct_orientation(nrrd_path, nifti_path)
