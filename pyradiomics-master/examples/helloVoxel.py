#!/usr/bin/env python
# 基于体素的提取

# 指定脚本运行的解释器
from __future__ import print_function
# 从__future__模块导入print_function，确保print在Python 2.x和3.x中具有相同的功能

import logging
import os
# 导入日志记录和操作系统相关的模块

import SimpleITK as sitk
import six
# 导入SimpleITK处理医学图像，six用于Python 2和3兼容性

import radiomics
from radiomics import featureextractor, getFeatureClasses
# 从radiomics库导入特征提取器和获取特征类的函数

def tqdmProgressbar():
  """
  设置tqdm包提供的进度条。
  进度报告仅在PyRadiomics的GLCM和GLSZM的完全Python模式计算中使用，因此启用GLCM和完全Python模式以展示进度条功能。
  注意：此函数仅在安装了'tqdm'包时有效（不包含在PyRadiomics的要求中）。
  """
  global extractor
  # 声明extractor为全局变量

  radiomics.setVerbosity(logging.INFO)  # 将详细程度至少设置为INFO以启用进度条

  import tqdm
  radiomics.progressReporter = tqdm.tqdm
  # 导入tqdm包，并将其设置为PyRadiomics的进度报告器
def clickProgressbar():
  """
  设置click包提供的进度条。
  进度报告仅在PyRadiomics的GLCM和GLSZM的完全Python模式计算中使用，因此启用GLCM和完全Python模式以展示进度条功能。
  由于实例化click进度条的签名与PyRadiomics期望的不同，我们需要编写一个简单的包装类来使用click进度条。在这种情况下，我们只需要将'desc'关键字参数更改为'label'关键字参数。
  注意：此函数仅在安装了'click'包时有效（不包含在PyRadiomics的要求中）。
  """
  global extractor
  # 声明extractor为全局变量

  # 启用GLCM类以展示进度条
  extractor.enableFeatureClassByName('glcm')

  radiomics.setVerbosity(logging.INFO)  # 将详细程度至少设置为INFO以启用进度条

  import click
  # 导入click包

  class progressWrapper:
    def __init__(self, iterable, desc=''):
      # 对于click进度条，描述必须在'label'关键字参数中提供。
      self.bar = click.progressbar(iterable, label=desc)

    def __iter__(self):
      return self.bar.__iter__()  # 重定向到click进度条的__iter__函数

    def __enter__(self):
      return self.bar.__enter__()  # 重定向到click进度条的__enter__函数

    def __exit__(self, exc_type, exc_value, tb):
      return self.bar.__exit__(exc_type, exc_value, tb)  # 重定向到click进度条的__exit__函数

  radiomics.progressReporter = progressWrapper
  # 将PyRadiomics的进度报告器设置为progressWrapper类的实例


testCase = 'lung2'
# testCase变量指定了要使用的测试案例名称
repositoryRoot = os.path.abspath(os.path.join(os.getcwd(), ".."))
# 计算存储库根目录的绝对路径
imageName, maskName = radiomics.getTestCase(testCase, repositoryRoot)
# 使用radiomics.getTestCase函数获取测试案例的图像和掩码文件路径
paramsFile = os.path.abspath(r'exampleSettings\exampleVoxel.yaml')
# 计算参数文件的绝对路径
if imageName is None or maskName is None:  # 如果获取测试案例时出现问题，PyRadiomics也会记录一个错误
  print('Error getting testcase!')
  exit()
  # 如果无法获取测试案例，则打印错误信息并退出程序


# 通过radiomics.verbosity调节详细程度
# radiomics.setVerbosity(logging.INFO)

# 获取PyRadiomics的日志记录器（默认日志级别=INFO）
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # 将日志级别设置为DEBUG，以在日志文件中包含调试日志消息

handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# 创建一个日志处理器，将日志消息格式化并写入'testLog.txt'文件

extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
featureClasses = getFeatureClasses()
# 实例化特征提取器，并获取特征类

# 取消注释以下其中一个函数，以展示PyRadiomics如何在完全Python模式下运行时使用'tqdm'或'click'包报告进度。假设已安装相应的包（不包含在要求中）
tqdmProgressbar()
# clickProgressbar()

print("Active features:")
for cls, features in six.iteritems(extractor.enabledFeatures):
  if features is None or len(features) == 0:
    features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
  for f in features:
    print(f)
    print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)
# 打印激活的特征，并输出每个特征的文档字符串

print("计算特征")
featureVector = extractor.execute(imageName, maskName, voxelBased=True)
# 计算特征，并将结果存储在featureVector中

for featureName, featureValue in six.iteritems(featureVector):
  if isinstance(featureValue, sitk.Image):
    sitk.WriteImage(featureValue, '%s_%s.nrrd' % (testCase, featureName))
    print('Computed %s, stored as "%s_%s.nrrd"' % (featureName, testCase, featureName))
  else:
    print('%s: %s' % (featureName, featureValue))
# 遍历特征向量中的每个特征，如果特征值是SimpleITK图像，则将其写入NRRD文件。否则，直接打印特征名和特征值。

