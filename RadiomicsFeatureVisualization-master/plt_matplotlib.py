import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def load_nifti_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def find_max_label_slice(label_array):
    # 计算每一层标签的面积（即标签像素的数量）
    label_areas = np.sum(label_array, axis=(1, 2))
    # 找到面积最大的层的索引
    return np.argmax(label_areas)
# 读取图像和标签
image_path = "./data/imagesTr/lung_3.nii.gz"
label_path = "./data/labelsTr/lung_3.nii.gz"
image = load_nifti_image(image_path)
label = load_nifti_image(label_path)

# 找到最大截面层
max_slice_index = find_max_label_slice(label)

# 可视化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image[max_slice_index], cmap="gray")
plt.title("Image at Max Label Slice")
plt.subplot(1, 2, 2)
plt.imshow(label[max_slice_index])
plt.title("Label at Max Label Slice")
plt.show()

#%% md






