import SimpleITK as sitk  # 导入SimpleITK库，用于读取医学图像
import six  # 导入six库，提供Python 2和3兼容性
import os  # 导入os库，用于处理文件路径和目录
import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，用于绘图

import time  # 导入time库，用于计时
import tqdm  # 导入tqdm库，用于显示进度条

# 定义FeatureMapVisualizition类，用于特征图的可视化
class FeatureMapVisualizition:
    def __init__(self):
        self.original_image_path = ''  # 原始图像路径
        self.original_roi_path = ''  # 原始ROI（感兴趣区域）路径
        self.feature_map_path = ''  # 特征图路径
        self.feature_name = ''  # 特征名称

    # 加载数据
    def LoadData(self, original_image_path, original_roi_path, feature_map_path):
        self.original_image_path = original_image_path
        self.original_roi_path = original_roi_path
        self.feature_map_path = feature_map_path
        self.feature_name = (os.path.split(self.feature_map_path)[-1]).split('.')[0]  # 从特征图路径中提取特征名称
        _, _, self.original_image_array = self.LoadNiiData(self.original_image_path, is_show_info=False)  # 加载原始图像数据
        _, _, self.original_roi_array = self.LoadNiiData(self.original_roi_path, is_show_info=False)  # 加载原始ROI数据
        self._max_roi_index = np.argmax(np.sum(self.original_roi_array, axis=(0,1)))  # 计算ROI最大值的索引
        _, _, self.original_feature_map_array = self.LoadNiiData(self.feature_map_path, is_show_info=False)  # 加载特征图数据
        if os.path.splitext(self.feature_map_path)[-1] == '.nrrd':  # 如果特征图格式为.nrrd
            self.original_feature_map_array = self.GenerarionFeatureROI()  # 生成特征ROI
        else:
            _, _, self.original_feature_map_array = self.LoadNiiData(self.feature_map_path, is_show_info=False)  # 重新加载特征图数据

    # 获取ROI中目标值的索引范围
    def GetIndexRangeInROI(self, roi_mask, target_value=1):
        # 如果ROI掩码是二维的
        if np.ndim(roi_mask) == 2:
            x, y = np.where(roi_mask == target_value)  # 获取目标值的x,y坐标
            x = np.unique(x)  # 获取唯一的x坐标
            y = np.unique(y)  # 获取唯一的y坐标
            return x, y
        # 如果ROI掩码是三维的
        elif np.ndim(roi_mask) == 3:
            x, y, z = np.where(roi_mask == target_value)  # 获取目标值的x,y,z坐标
            x = np.unique(x)  # 获取唯一的x坐标
            y = np.unique(y)  # 获取唯一的y坐标
            z = np.unique(z)  # 获取唯一的z坐标
            return x, y, z

    # 加载NII格式的数据
    def LoadNiiData(self, file_path, dtype=np.float32, is_show_info=False, is_flip=True, flip_log=[0, 0, 0]):
        image = sitk.ReadImage(file_path)  # 读取图像
        data = np.asarray(sitk.GetArrayFromImage(image), dtype=dtype)  # 将图像转换为numpy数组

        show_data = np.transpose(data)  # 转置数据
        show_data = np.swapaxes(show_data, 0, 1)  # 交换轴

        # 处理翻转情况
        if is_flip:
            direction = image.GetDirection()  # 获取图像方向
            for direct_index in range(3):
                direct_vector = [direction[0 + direct_index * 3], direction[1 + direct_index * 3], direction[2 + direct_index * 3]]
                abs_max_point = np.argmax(np.abs(direct_vector))
                if direct_vector[abs_max_point] < 0:
                    show_data = np.flip(show_data, axis=direct_index)  # 翻转数据
                    flip_log[direct_index] = 1

        # 如果需要显示图像信息
        if is_show_info:
            print('Image size is: ', image.GetSize())
            print('Image resolution is: ', image.GetSpacing())
            print('Image direction is: ', image.GetDirection())
            print('Image Origion is: ', image.GetOrigin())

        return image, data, show_data

    # 生成特征ROI
    def GenerarionFeatureROI(self):
        x, y, z = self.GetIndexRangeInROI(self.original_roi_array)  # 获取ROI中目标值的索引范围
        feature_roi = np.zeros_like(self.original_roi_array, dtype=np.float32)  # 初始化特征ROI数组
        feature_roi[np.min(x):np.min(x) + self.original_feature_map_array.shape[0],
        np.min(y):np.min(y) + self.original_feature_map_array.shape[1],
        :self.original_feature_map_array.shape[2]]= self.original_feature_map_array  # 将特征图数据填充到特征ROI数组中
        return feature_roi

    # 数据归一化到0-1范围
    def Normalize01(self, data, clip=0.0):
        new_data = np.asarray(data, dtype=np.float32)  # 将数据转换为float32类型
        if clip > 1e-6:  # 如果需要裁剪
            data_list = data.flatten().tolist()  # 将数据展平并转换为列表
            data_list.sort()  # 对数据进行排序
            new_data.clip(data_list[int(clip * len(data_list))], data_list[int((1 - clip) * len(data_list))])  # 裁剪数据

        new_data = new_data - np.min(new_data)  # 数据减去最小值
        new_data = new_data / np.max(new_data)  # 数据除以最大值，实现归一化
        return new_data

    # 显示变换后的图像
    def ShowTransforedImage(self, store_path):
        feature_map_show_array = self.original_feature_map_array[:, :, self._max_roi_index]  # 获取特征图数据
        plt.title(self.feature_name)  # 设置标题为特征名称
        plt.imshow(feature_map_show_array, cmap='gray')  # 显示特征图
        plt.contour(self.original_roi_array[:, :, self._max_roi_index], colors='r')  # 绘制ROI轮廓
        plt.axis('off')  # 不显示坐标轴
        if store_path:  # 如果提供了存储路径
            plt.savefig(store_path+'_transformed_image.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)  # 保存为JPG格式
            plt.savefig(store_path+'_transformed_image.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0)  # 保存为EPS格式

        plt.close()  # 显示图像

    # 根据ROI显示颜色
    def ShowColorByROI(self, background_array, fore_array, roi_array, color_map, feature_name, threshold_value=1e-6, size=0, store_path='',
                       is_show=True):
        # 计算ROI中心
        roi_center = [int(np.mean(np.where(roi_array == 1)[0])), int(np.mean(np.where(roi_array == 1)[1]))]

        # 如果指定了大小
        if size:
            roi_array = roi_array[roi_center[0] - size:roi_center[0] + size,
                            roi_center[1] - size:roi_center[1] + size]

            background_array = background_array[roi_center[0] - size:roi_center[0] + size,
                            roi_center[1] - size:roi_center[1] + size]

            fore_array = fore_array[roi_center[0] - size:roi_center[0] + size,
                               roi_center[1] - size:roi_center[1] + size]

        # 确保背景数组和ROI数组形状相同
        if background_array.shape != roi_array.shape:
            print('Array and ROI must have same shape')
            return

        background_array = self.Normalize01(background_array)  # 背景数组归一化
        fore_array = self.Normalize01(fore_array)  # 前景数组归一化
        cmap = plt.get_cmap(color_map)  # 获取颜色映射
        rgba_array = cmap(fore_array)  # 将前景数组映射为RGBA颜色
        rgb_array = np.delete(rgba_array, 3, 2)  # 删除alpha通道，得到RGB颜色

        index_roi_x, index_roi_y = np.where(roi_array < threshold_value)  # 获取ROI中小于阈值的索引

        start_time = time.time()  # 开始计时
        index_list = range(len(index_roi_x))  # 创建索引列表
        for position_index in tqdm.tqdm(index_list):  # 遍历索引列表
            index_x, index_y = index_roi_x[position_index], index_roi_y[position_index]  # 获取索引

            rgb_array[index_x, index_y, :] = background_array[index_x, index_y]  # 将背景颜色填充到RGB数组中
        print(time.time()-start_time)  # 打印耗时
        plt.imshow(rgb_array, cmap=color_map)  # 显示RGB图像
        plt.colorbar()  # 显示颜色条
        plt.axis('off')  # 不显示坐标轴
        plt.gca().set_axis_off()  # 关闭坐标轴
        plt.title(feature_name)  # 设置标题为特征名称

        if store_path:  # 如果提供了存储路径
            plt.savefig(store_path+'.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)  # 保存为JPG格式
            plt.savefig(store_path+'.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0)  # 保存为EPS格式

        # if is_show:  # 如果需要显示图像
            # plt.show()  # 显示图像

        plt.close()  # 关闭图像
        plt.clf()  # 清除图像
        return rgb_array  # 返回RGB数组

    # 显示特征图
    def Show(self, index='', color_map='rainbow', store_path=''):
        if not index:  # 如果没有指定索引
            background_array = self.original_image_array[:, :, self._max_roi_index]  # 获取背景数组
            fore_array = self.original_feature_map_array[:, :, self._max_roi_index]  # 获取前景数组
            roi = self.original_roi_array[:,:, self._max_roi_index]  # 获取ROI

        else:
            background_array = self.original_image_array[:, :, index]  # 获取指定索引的背景数组
            fore_array = self.original_feature_map_array[:, :, index]  # 获取指定索引的前景数组
            roi = self.original_roi_array[:, :, index]  # 获取指定索引的ROI

        self.ShowColorByROI(background_array, fore_array, roi, color_map=color_map, feature_name=self.feature_name,store_path=store_path, is_show=False)  # 根据ROI显示颜色

def main():
    image_path = './FeatureMapByClass/FeatureMap_class/original_cropped_img.nii.gz'  # 图像路径
    roi_path = './FeatureMapByClass/FeatureMap_class/original_cropped_roi.nii.gz' # ROI路径
    feature_map_path = r"D:\zhuomian\pyradiomics\RadiomicsFeatureVisualization-master\FeatureMapByClass\FeatureMap_class\original_firstorder_Entropy.nrrd" # 特征图路径

    # 初始化FeatureMapVisualizition对象
    featuremapvisualization = FeatureMapVisualizition()
    featuremapvisualization.LoadData(image_path, roi_path, feature_map_path)  # 加载数据
    store_path = './FeatureMapByClass/output'  # 存储路径
    store_figure_path = store_path+'\\' +'rainbow_300'+ (os.path.split(feature_map_path)[-1]).split('.')[0]  # 存储图像的路径
    # 使用'seismic'颜色映射显示特征图           ↑↓
    featuremapvisualization.Show(color_map='rainbow', store_path=store_figure_path)
    featuremapvisualization.ShowTransforedImage(store_figure_path)  # 显示变换后的图像
# color_map参数：
# 'jet': 这是一个从蓝色开始，经过青色、黄色，最后到红色的彩虹色映射，对比度较高。
# 'viridis': 这是Matplotlib的默认颜色映射之一，它从黄绿色渐变到深蓝色，视觉上非常鲜明。
# 'plasma': 从紫色到黄色的顺序颜色映射，对比度很高，适合于突出显示。
# 'inferno': 一个从黑色到红色再到黄色的颜色映射，具有很好的亮度变化和高对比度。
# "rainbow":彩虹包含了多种颜色，通常按照光谱顺序排列，从外圈到内圈依次是：红色、橙色、黄色、绿色、蓝色、靛蓝（或称为青色）和紫色。

if __name__ == '__main__':
    main()
