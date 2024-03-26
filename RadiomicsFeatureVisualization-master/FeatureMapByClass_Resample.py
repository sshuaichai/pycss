#!/D:/anaconda python
# -*- coding: utf-8 -*-
# @Project : MyScript
# @FileName: FeatureMapByClass.py
# @IDE: PyCharm
# @Time  : 2020/3/9 21:16
# @Author : Jing.Z
# @Email : zhangjingmri@gmail.com
# @Desc : ==============================================
# 生命苦短，我用Python!!!
# ======================================================

# 导入必要的库
import os
import time
import six

import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# from SKMRradiomics import featureextractor  # 从SKMRradiomics库导入特征提取器
from radiomics import featureextractor  # 或从radiomics库导入特征提取器
from FeatureMapShow import FeatureMapVisualizition  # 导入特征图可视化工具

class FeatureMapper:
    """
    此类用于生成基于体素的放射组学特征图的候选图像。
    """

    def __init__(self):
        self.feature_pd = pd.DataFrame()  # 初始化用于存储特征数据的DataFrame
        self.selected_feature_list = []  # 初始化选定的特征列表
        self.store_path = r""  # 初始化存储路径
        self.kernelRadius = ''  # 初始化核半径
        self.sub_img_array = np.array([])  # 初始化子图像数组
        self.sub_roi_array = np.array([])  # 初始化子ROI数组

    def load(self, feature_csv_path, selected_feature_list):
        # 加载特征数据和选定的特征列表
        self.feature_pd = pd.read_csv(feature_csv_path, index_col=0)
        self.selected_feature_list = selected_feature_list

    def seek_single_candidate_case(self, feature_name, case_num):
        # 寻找单个候选案例
        sub_feature_pd = self.feature_pd[['label', feature_name]].copy()
        sub_feature_pd.sort_values(by=feature_name, inplace=True)
        sorted_index_list = sub_feature_pd.axes[0]
        max_case, min_case = sorted_index_list[-1], sorted_index_list[0]

        max_info = max_case + '(' + str(int(sub_feature_pd.at[max_case, 'label'])) + ')'
        min_info = min_case + '(' + str(int(sub_feature_pd.at[min_case, 'label'])) + ')'
        print('{} 的最大值案例：{}, 最小值案例：{}'.format(feature_name, max_info, min_info))
        top_case_list = list(sorted_index_list)[-case_num:]
        last_case_list = list(sorted_index_list)[:case_num]
        return top_case_list, last_case_list

    def seek_candidate_case(self, feature_csv_path, selected_feature_list, case_num):
        """
        从特征csv中根据特征值寻找候选图像，并打印候选案例。
        """
        self.load(feature_csv_path, selected_feature_list)
        candidate_case_list = []
        candidate_case_dict = {}
        for sub_feature in selected_feature_list:
            top_case_list, last_case_list = self.seek_single_candidate_case(sub_feature, case_num)
            candidate_case_dict[sub_feature] = {'top': top_case_list, 'last': last_case_list}

        # 寻找共同案例
        for sub_feature in list(candidate_case_dict.keys()):
            all_features = candidate_case_dict[sub_feature]['top'] + candidate_case_dict[sub_feature]['last']
            if len(candidate_case_list) == 0:
                candidate_case_list = all_features
            else:
                candidate_case_list = list(set(candidate_case_list).intersection(set(all_features)))

        # 检查共同案例
        for sub_feature in list(candidate_case_dict.keys()):
            sub_checked_case = list(set(candidate_case_dict[sub_feature]['top']).
                                    intersection(set(candidate_case_list)))

            candidate_case_dict[sub_feature]['top'] = [index + "(" + str(int(self.feature_pd.at[index, 'label'])) + ")"
                                                       for index in sub_checked_case]

            sub_checked_case = list(set(candidate_case_dict[sub_feature]['last']).
                                    intersection(set(candidate_case_list)))
            candidate_case_dict[sub_feature]['last'] = [index + "(" + str(int(self.feature_pd.at[index, 'label'])) + ")"
                                                        for index in sub_checked_case]

        df = pd.DataFrame.from_dict(candidate_case_dict, orient='index')
        print(df)

    @staticmethod
    def decode_feature_name(feature_name_list):
        # 解码特征名称
        sub_filter_name = ''
        img_setting = {'imageType': 'Original'}
        feature_dict = {}
        for sub_feature in feature_name_list:

            # 大特征类
            if sub_feature in ['firstorder', 'glcm', 'glrlm', 'ngtdm', 'glszm']:
                # 提取所有特征
                sub_feature_setting = {sub_feature: []}
                feature_dict.update(sub_feature_setting)

            else:
                img_type = sub_feature.split('_')[-3]
                if img_type.find('wavelet') != -1:
                    img_setting['imageType'] = 'Wavelet'
                    sub_filter_name = img_type.split('-')[-1]
                elif img_type.find('LOG') != -1:
                    img_setting['imageType'] = 'LoG'
                    sub_filter_name = img_type

                else:
                    img_setting['imageType'] = 'Original'

                feature_class = sub_feature.split('_')[-2]
                feature_name = sub_feature.split('_')[-1]

                if feature_class not in feature_dict.keys():
                    feature_dict[feature_class] = []
                    feature_dict[feature_class].append(feature_name)
                else:
                    feature_dict[feature_class].append(feature_name)
        print(img_setting)
        print(feature_dict)
        return img_setting, feature_dict, sub_filter_name

    # 裁剪图像以核半径为界，移除冗余切片以加速处理
    def crop_img(self, original_roi_path, original_img_path, store_key=''):
        roi = sitk.ReadImage(original_roi_path)

        roi_array = sitk.GetArrayFromImage(roi)
        max_roi_slice_index = np.argmax(np.sum(roi_array, axis=(1, 2)))

        z_range = [max_roi_slice_index - self.kernelRadius, max_roi_slice_index + self.kernelRadius + 1]
        x_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=0) > 0)[0]
        x_range = [min(x_index) - self.kernelRadius, max(x_index) + self.kernelRadius + 1]
        y_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=1) > 0)[0]
        y_range = [min(y_index) - self.kernelRadius, max(y_index) + self.kernelRadius + 1]


        cropped_roi_array = roi_array[z_range[0]:z_range[1]]
        cropped_roi = sitk.GetImageFromArray(cropped_roi_array)
        cropped_roi.SetDirection(roi.GetDirection())
        cropped_roi.SetOrigin(roi.GetOrigin())
        cropped_roi.SetSpacing(roi.GetSpacing())

        img = sitk.ReadImage(original_img_path)
        img_array = sitk.GetArrayFromImage(img)
        cropped_img_array = img_array[z_range[0]:z_range[1]]
        cropped_img = sitk.GetImageFromArray(cropped_img_array)
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.GetOrigin())
        cropped_img.SetSpacing(img.GetSpacing())

        roi_info = [roi.GetDirection(), roi.GetOrigin(), roi.GetSpacing()]
        img_info = [img.GetDirection(), img.GetOrigin(), img.GetSpacing()]
        index_dict = {0:'direction', 1:'origin', 2:'spacing'}
        start = 0
        for sub_roi_info, sub_img_info in zip(roi_info, img_info):
            if sub_roi_info != sub_img_info:
                print(index_dict[start], '失败')
                print('roi:', sub_roi_info)
                print('img:', sub_img_info)

        sitk.WriteImage(cropped_img, os.path.join(self.store_path, store_key + '_cropped_img.nii.gz'))
        sitk.WriteImage(cropped_roi, os.path.join(self.store_path, store_key + '_cropped_roi.nii.gz'))
        self.sub_img_array = np.transpose(cropped_img_array, (1, 2, 0))
        self.sub_roi_array = np.transpose(cropped_roi_array, (1, 2, 0))
        print('ROI大小: ', np.sum(cropped_roi))
        return cropped_img, cropped_roi

    def generate_feature_map(self, candidate_img_path, candidate_roi_path, kernelRadius, feature_name_list, store_path):
        """
        基于核半径生成特定特征图。

        参数
        ----------
        candidate_img_path: str, 候选图像路径;
        candidate_roi_path: str, 候选ROI路径;
        kernelRadius: integer, 指定用作中心体素半径的核的大小。因此实际大小是 2 * kernelRadius + 1。例如，值为1时产生一个3x3x3的核，值为2时产生5x5x5，等等。在2D提取的情况下，生成的核也将是一个2D形状（正方形而不是立方体）。
        feature_name_list: [str], [feature_name1, feature_name2,...] 或 ['glcm', 'glrlm']
        store_path: str;

        返回
        -------

        """

        start_time = time.time()
        self.kernelRadius = kernelRadius
        self.store_path = store_path
        parameter_path = r"D:\zhuomian\pyradiomics\RadiomicsFeatureVisualization-master\exampleVoxel_R3B12.yaml"
        setting_dict = {'label': 1, 'interpolator': 'sitkBSpline', 'correctMask': True,
                        'geometryTolerance': 10, 'kernelRadius': self.kernelRadius,
                        'maskedKernel': True, 'voxelBatch': 50}

        extractor = featureextractor.RadiomicsFeatureExtractor(parameter_path, self.store_path, **setting_dict)
        extractor.disableAllImageTypes()
        extractor.disableAllFeatures()

        img_setting, feature_dict, sub_filter_name = self.decode_feature_name(feature_name_list)
        extractor.enableImageTypeByName(**img_setting)
        extractor.enableFeaturesByName(**feature_dict)

        cropped_original_img, cropped_original_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key='original')

        if sub_filter_name:
            # 首先生成滤波图像以加速处理
            extractor.execute(candidate_img_path, candidate_roi_path, voxelBased=False)
            candidate_img_path = os.path.join(self.store_path, sub_filter_name+'.nii.gz')
            cropped_filter_img, cropped_filter_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key=sub_filter_name)
            result = extractor.execute(cropped_filter_img, cropped_filter_roi, voxelBased=True)
        #
        #
        else:
            result = extractor.execute(cropped_original_img, cropped_original_roi, voxelBased=True)
        # 无参数，glcm ,kr=5 ,646s ,裁剪图像, map shape (5, 132, 128)
        # 无参数，glcm ,kr=1 ,386s ,裁剪图像, map shape (3, 122, 128)

        # 无参数，glcm ,kr=1 ,566s ,无裁剪图像, map shape (5, 132, 128)

        # 提取原始图像

        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                shape = (sitk.GetArrayFromImage(val)).shape
                # 特征图
                sitk.WriteImage(val, store_path + '\\' + key + '.nrrd', True)


    def show_feature_map(self, show_img_path, show_roi_path, show_feature_map_path, store_path):
        feature_map_img = sitk.ReadImage(show_feature_map_path)
        feature_map_array = sitk.GetArrayFromImage(feature_map_img)
        feature_map_array.transpose(1, 2, 0)
        feature_map_visualization = FeatureMapVisualizition()
        feature_map_visualization.LoadData(show_img_path, show_roi_path, show_feature_map_path)

        # hsv/jet/gist_rainbow
        feature_map_visualization.Show(color_map='rainbow', store_path=store_path)
    ##color_map参数：
    # # 'jet': 这是一个从蓝色开始，经过青色、黄色，最后到红色的彩虹色映射，对比度较高。
    # # 'viridis': 这是Matplotlib的默认颜色映射之一，它从黄绿色渐变到深蓝色，视觉上非常鲜明。
    # # 'plasma': 从紫色到黄色的顺序颜色映射，对比度很高，适合于突出显示。
    # # 'inferno': 一个从黑色到红色再到黄色的颜色映射，具有很好的亮度变化和高对比度。
    #-----------------------
def main():
    feature_mapper = FeatureMapper()

    # ' #可更改为features_class_list：定义了要提取的特征类，提取类下的特征'
    features_class_list = ['glcm'] #features_class_list是一个包含特征类别的列表。
    cur_file_path = Path(__file__).absolute().parent #cur_file_path 变量最终保存了当前脚本文件的绝对路径的父目录路径。

    # 定义了图像和ROI的路径。
    img_path = cur_file_path / 'data' / 'imagesTr1' / 'lung_10.nii.gz'
    roi_path = cur_file_path / 'data' / 'labelsTr1' / 'lung_10.nii.gz'

    # 定义了存储特征图的路径。
    store_path = cur_file_path / 'FeatureMapByClass' / 'FeatureMap_class'

    if not Path(store_path).exists():
        Path(store_path).mkdir()

    '生成特征图：生成上面定义的features_class_list/features_name_list指定的特征map图'
    # 可更改为 features_class_list 替换 features_name_list；
    feature_mapper.generate_feature_map(str(img_path), str(roi_path), 1, features_class_list, str(store_path))

    # 设置保存路径
    cropped_img_path = cur_file_path / 'FeatureMapByClass' / 'FeatureMap_class' / 'original_cropped_img.nii.gz'
    # 设置裁剪后的ROI路径
    cropped_roi_path = cur_file_path / 'FeatureMapByClass' / 'FeatureMap_class' / 'original_cropped_roi.nii.gz'
    # 设置要显示的特征具体名称
    feature_name = 'original_glcm_DifferenceEntropy'  #在保存的nrrd中选择
    # 设置特征图路径
    feature_map = cur_file_path / 'FeatureMapByClass' / 'FeatureMap_class' / str(feature_name + '.nrrd')
    # 设置要保存图像的文件路径
    fig_save_path = cur_file_path / 'FeatureMapByClass' / 'FeatureMap_class' / feature_name
    # 显示特征图，（选取一个特征名称 features_name来直接可视化map图）
    feature_mapper.show_feature_map(str(cropped_img_path), str(cropped_roi_path), str(feature_map),
                                    str(fig_save_path))

if __name__ == '__main__':
    main()

