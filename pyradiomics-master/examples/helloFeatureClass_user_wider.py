#!/usr/bin/env python

from radiomics import featureextractor
import SimpleITK as sitk
import os
import csv

def configure_extractor():
    """
    配置并返回一个特征提取器，该提取器启用了特定的特征集。
    """
    params = {
        'binWidth': 25,
        'label': 1,
        'sigma': [1, 2, 3],  # 用于LoG特征
        'resampledPixelSpacing': None,  # 不进行重采样
        'interpolator': sitk.sitkBSpline,
        'enableCExtensions': True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableAllFeatures()
    return extractor

def extract_features(image_path, mask_path, extractor):
    """
    使用配置好的提取器提取图像的特征。
    """
    return extractor.execute(image_path, mask_path)

def save_features_wide_format(features_list, output_path):
    """
    将提取的特征以宽表格格式保存到CSV文件。
    """
    all_features = set()
    for _, features in features_list:
        all_features.update(features.keys())
    all_features = sorted(list(all_features))

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image'] + all_features)
        for image_name, features in features_list:
            row = [image_name] + [features.get(feature, 'NA') for feature in all_features]
            writer.writerow(row)

def process_images(image_dir, mask_dir, output_path, extractor):
    """
    处理指定目录中的所有图像和掩码，提取特征并保存到一个宽表格格式的CSV文件。
    """
    features_list = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)  # Use the same file name for mask

        if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
            print(f"Skipping {image_name} as matching mask file is not found.")
            continue

        print(f'Processing {image_name}...')
        features = extract_features(image_path, mask_path, extractor)
        features_list.append((image_name, features))

    save_features_wide_format(features_list, output_path)

if __name__ == '__main__':
    image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Task02_Heart\resampled_images"
    mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Task02_Heart\resampled_masks"  # Update to the directory of resampled masks
    output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\all_features_wide.csv"

    extractor = configure_extractor()
    process_images(image_dir, mask_dir, output_path, extractor)
