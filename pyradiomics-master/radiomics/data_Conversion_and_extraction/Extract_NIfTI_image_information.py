import os
import nibabel as nib
import numpy as np


def list_images(directory, extensions=['.nii', '.nii.gz']):
  images = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if any(file.lower().endswith(ext) for ext in extensions):
        images.append(os.path.join(root, file))
  return images


def image_statistics(image_path):
  img = nib.load(image_path)
  data = img.get_fdata()

  stats = {
    'min': np.min(data),
    'max': np.max(data),
    'mean': np.mean(data),
    'std': np.std(data),
  }
  return stats


image_directory = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset020_Lung\imagesTr"
image_files = list_images(image_directory)

print("图像统计信息:")
for image_file in image_files:
  stats = image_statistics(image_file)
  print(f"{image_file}:")
  print(f"  最小值: {stats['min']}, 最大值: {stats['max']}, 均值: {stats['mean']}, 标准差: {stats['std']}")
'''
这个脚本首先列出指定目录下的所有 NIfTI 图像文件，然后对每个图像计算最小值、最大值、均值和标准差等统计信息。
通过比较不同图像的这些统计信息，您可以初步判断它们之间是否存在显著差异。
例如，如果图像之间的体素值范围差异很大，可能表明它们来自不同的扫描仪或使用了不同的扫描参数，这种情况下可能需要进行归一化处理。
'''
