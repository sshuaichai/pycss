from PIL import Image
import os

# 定义原始JPEG图片的路径
jpeg_image_path = "D:\\zhuomian\\pyradiomics\\RadiomicsFeatureVisualization-master\\FeatureMapByClass\\FeatureMap_class\\original_glcm_DifferenceEntropy.jpg"

# 定义保存TIFF图片的路径
tiff_image_path = "D:\\zhuomian\\pyradiomics\\RadiomicsFeatureVisualization-master\\FeatureMapByClass\\FeatureMap_tiff\\original_glcm_DifferenceEntropy.tiff"

# 确保目标目录存在
os.makedirs(os.path.dirname(tiff_image_path), exist_ok=True)

# 读取JPEG图片
image = Image.open(jpeg_image_path)

# 保存为TIFF格式
image.save(tiff_image_path, format='TIFF')

print(f"Image saved as TIFF at: {tiff_image_path}")
