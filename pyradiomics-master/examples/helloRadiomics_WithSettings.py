#!/usr/bin/env python

from __future__ import print_function

import logging  # 导入日志记录模块
import os  # 导入操作系统相关模块

import six  # 导入兼容性模块

import radiomics  # 导入放射组学模块
from radiomics import featureextractor, getFeatureClasses  # 导入特征提取器和特征类获取函数

# 获取一些测试数据

# 将测试用例下载到临时文件并返回其位置。如果已经下载，则不会再次下载，但其位置仍会返回。
imageName, maskName = radiomics.getTestCase('brain1')

# 获取示例设置文件的位置
paramsFile = os.path.abspath(os.path.join('exampleSettings', 'Params.yaml'))

if imageName is None or maskName is None:  # 出现问题，此时 PyRadiomics 也会记录错误
  print('获取测试案例时出错！')
  exit()

# 使用 radiomics.verbosity 调整详细程度
# radiomics.setVerbosity(logging.INFO)

# 获取 PyRadiomics 日志记录器（默认日志级别为 INFO）
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # 将级别设置为 DEBUG 以在日志文件中包含调试日志消息

# 将所有日志条目写入文件
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# 使用设置文件初始化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
featureClasses = getFeatureClasses()
'得到指定参数文件的特征，并打印出来查看'
print("激活的特征:")
for cls, features in six.iteritems(extractor.enabledFeatures):
  if features is None or len(features) == 0:
    features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
  for f in features:
    print(f)
    print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

print("计算特征")
featureVector = extractor.execute(imageName, maskName)

for featureName in featureVector.keys():
  print("计算出的 %s: %s" % (featureName, featureVector[featureName]))
