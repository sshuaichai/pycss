#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import logging
import os
import csv
import SimpleITK as sitk
import six
import radiomics
from radiomics import featureextractor
import yaml

def load_params(params_file):
    with open(params_file, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    return params

def configure_progressbar():
    radiomics.setVerbosity(logging.INFO)
    try:
        import tqdm
        radiomics.progressReporter = tqdm.tqdm
    except ImportError:
        print("tqdm package is not installed. Progress bar functionality will be disabled.")
        radiomics.progressReporter = None

def process_images(image_dir, mask_dir, output_file, extractor):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    all_features = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)

        if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
            print(f"Skipping {image_name} as matching mask file is not found.")
            continue

        print(f'Processing {image_name}...')
        features = extractor.execute(image_path, mask_path, voxelBased=False)
        features['Image'] = image_name
        all_features.append(features)

    with open(output_file, 'w', newline='') as csvfile:
        writer = None
        for features in all_features:
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=features.keys())
                writer.writeheader()
            writer.writerow(features)

if __name__ == '__main__':
    image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset022_lung\imagesTr"
    mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset022_lung\labelsTr"
    output_file = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\Voxel_feature_R1B12.csv"
    params_file = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\exampleVoxel_R1B12.yaml"

    params = load_params(params_file)
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)  # Ensure params are passed correctly
    configure_progressbar()
    process_images(image_dir, mask_dir, output_file, extractor)
