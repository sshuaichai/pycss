#!/usr/bin/env python

from __future__ import print_function

import os
import numpy
import SimpleITK as sitk
import six

from radiomics import firstorder, glcm, glrlm, glszm, imageoperations, shape
'''例子
这里包括了一阶统计
基于形状（2D 和 3D）
灰度共生矩阵 (GLCM)
灰度游程矩阵 (GLRLM)
灰度大小区域矩阵 (GLSZM)
LoG特征和小波特征处理的一阶特征
'''
def process_image(image_path, mask_path, settings):
    print(f"Processing {image_path} with mask {mask_path}")

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # 如果启用，则对图像进行重采样（重采样图像会自动裁剪）
    interpolator = settings.get('interpolator')
    resampledPixelSpacing = settings.get('resampledPixelSpacing')
    if interpolator is not None and resampledPixelSpacing is not None:
        image, mask = imageoperations.resampleImage(image, mask, **settings)

    bb, correctedMask = imageoperations.checkMask(image, mask)
    if correctedMask is not None:
        mask = correctedMask
    image, mask = imageoperations.cropToTumorMask(image, mask, bb)

    # 计算并打印基本特征
    calculate_features(image, mask, settings)

    # 对LoG特征的处理
    if settings.get('applyLog', False):
        process_log_features(image, mask, settings)

    # 对小波变换特征的处理
    if settings.get('applyWavelet', False):
        process_wavelet_features(image, mask, settings)

def calculate_features(image, mask, settings):
    # 第一阶段特征
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
    firstOrderFeatures.enableAllFeatures()
    print('第一阶段特征:')
    results = firstOrderFeatures.execute()
    for (key, val) in six.iteritems(results):
        print('  ', key, ':', val)

    # 形状特征
    shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
    shapeFeatures.enableAllFeatures()
    print('形状特征:')
    results = shapeFeatures.execute()
    for (key, val) in six.iteritems(results):
        print('  ', key, ':', val)

    # GLCM特征
    glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
    glcmFeatures.enableAllFeatures()
    print('GLCM特征:')
    results = glcmFeatures.execute()
    for (key, val) in six.iteritems(results):
        print('  ', key, ':', val)

    # GLRLM特征
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
    glrlmFeatures.enableAllFeatures()
    print('GLRLM特征:')
    results = glrlmFeatures.execute()
    for (key, val) in six.iteritems(results):
        print('  ', key, ':', val)

    # GLSZM特征
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
    glszmFeatures.enableAllFeatures()
    print('GLSZM特征:')
    results = glszmFeatures.execute()
    for (key, val) in six.iteritems(results):
        print('  ', key, ':', val)

def process_log_features(image, mask, settings):
    sigmaValues = numpy.arange(5., 0., -.5)[::1]
    for logImage, imageTypeName, inputKwargs in imageoperations.getLoGImage(image, mask, sigma=sigmaValues):
        logFirstorderFeatures = firstorder.RadiomicsFirstOrder(logImage, mask, **inputKwargs)
        logFirstorderFeatures.enableAllFeatures()
        results = logFirstorderFeatures.execute()
        print(f'LoG特征 ({imageTypeName}):')
        for (key, val) in six.iteritems(results):
            laplacianFeatureName = f'{imageTypeName}_{key}'
            print('  ', laplacianFeatureName, ':', val)

def process_wavelet_features(image, mask, settings):
    for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
        waveletFirstOrderFeatures = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **inputKwargs)
        waveletFirstOrderFeatures.enableAllFeatures()
        results = waveletFirstOrderFeatures.execute()
        print(f'小波特征 ({decompositionName}):')
        for (key, val) in six.iteritems(results):
            waveletFeatureName = f'{decompositionName}_{key}'
            print('  ', waveletFeatureName, ':', val)

def process_images(image_dir, mask_dir, settings):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)  # 确保掩码文件与图像文件同名

        if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
            print(f"Skipping {image_name} as matching mask file is not found.")
            continue

        process_image(image_path, mask_path, settings)

if __name__ == "__main__":
    image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Task02_Heart\imagesTr"
    mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Task02_Heart\labelsTr"
    settings = {
      'binWidth': 25,
      'interpolator': sitk.sitkBSpline,
      'resampledPixelSpacing': None,
      'applyLog': True,  # 启用LoG特征计算
      'applyWavelet': True  # 启用小波特征计算
    }

    process_images(image_dir, mask_dir, settings)
