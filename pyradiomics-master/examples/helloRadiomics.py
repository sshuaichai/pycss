# 导入必要的库
from __future__ import print_function
import logging
import os
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import pandas as pd

# 指定图像和掩码的路径
imageName = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\brain1_image.nrrd"
maskName = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\brain1_label.nrrd"

# 设置日志记录
logger = radiomics.logger
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# 定义特征提取的设置
settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkBSpline
}

# 初始化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
'''提取特征的特征里的特定类别：
firstorder是一个特征类别，表示一阶统计特征。['Mean', 'Skewness']是要启用的具体特征列表，包括均值（Mean）和偏度（Skewness）。'''

# 执行特征提取
print("正在计算特征")
featureVector = extractor.execute(imageName, maskName)

# 打印提取的特征
for featureName in featureVector.keys():
    print("计算得到的 %s: %s" % (featureName, featureVector[featureName]))

# 创建一个数据框（DataFrame）以存储特征
feature_df = pd.DataFrame.from_dict(featureVector, orient='index', columns=['Feature Value'])

# 指定保存 Excel 文件的路径
excel_file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\data_new\data_new.xlsx"

# 检查文件路径中的目录是否存在，如果不存在，则创建
excel_dir = os.path.dirname(excel_file_path)
if not os.path.exists(excel_dir):
    os.makedirs(excel_dir)

# 将数据框保存为 Excel 文件
feature_df.to_excel(excel_file_path)
print("特征已保存到 Excel 文件：%s" % excel_file_path)
