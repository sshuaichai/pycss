# 导入SKMRradiomics和原始radiomics包中的特征提取器
from SKMRradiomics import featureextractor as skmrradiomics
from radiomics import featureextractor as originalradiomcis
# import os  # 注释掉的os模块未使用
import csv  # 用于读写csv文件，但在此代码片段中未使用
import copy  # 用于复制数据结构，但在此代码片段中未使用
import logging  # 用于记录日志
import SimpleITK as sitk  # 用于医学图像处理
import glob  # 用于文件路径名模式匹配，但在此代码片段中未使用
import six  # Python 2和3兼容性库

# 定义一个函数来获取特征图
def GetFeatureMap(params_path, store_path, image_path, roi_path, voxelBasedSet=True):
    # 根据voxelBasedSet参数选择使用SKMRradiomics还是原始radiomics的特征提取器
    if not voxelBasedSet:
        # 使用SKMRradiomics的特征提取器
        extractor = skmrradiomics.RadiomicsFeaturesExtractor(params_path, store_path)
        result = extractor.execute(image_path, roi_path, voxelBased=voxelBasedSet)
    if voxelBasedSet:
        # 使用原始radiomics的特征提取器
        extractor = originalradiomcis.RadiomicsFeaturesExtractor(params_path, store_path)
        result = extractor.execute(image_path, roi_path, voxelBased=voxelBasedSet)
        # 遍历提取结果
        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                # 如果结果是SimpleITK图像，获取其形状并打印
                shape = (sitk.GetArrayFromImage(val)).shape
                print('feature_map shape is ', shape)
                # 将特征图保存为.nrrd文件
                sitk.WriteImage(val, store_path + '\\' + key + '.nrrd', True)
            else:  # 诊断信息
                print("\t%s: %s" % (key, val))

# 如果这个脚本作为主程序运行
if __name__ == "__main__":
    # 定义图像、ROI和保存路径
    image_path = r"D:\zhuomian\pyradiomics\RadiomicsFeatureVisualization-master\data\imagesTr1\lung_10.nii.gz"
    roi_path = r"D:\zhuomian\pyradiomics\RadiomicsFeatureVisualization-master\data\labelsTr1\lung_10.nii.gz"
    save_path = r"D:\zhuomian\pyradiomics\RadiomicsFeatureVisualization-master\FeatureMapByClass\output"
    # 调用GetFeatureMap函数
    GetFeatureMap(r'D:\zhuomian\pyradiomics\RadiomicsFeatureVisualization-master\RadiomicsParams.yaml', save_path, image_path, roi_path)
