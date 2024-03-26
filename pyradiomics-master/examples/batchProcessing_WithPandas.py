#!/usr/bin/env python
''

from __future__ import print_function

import logging
import os

import pandas
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import pandas as pd
import numpy as np

'''pandas批处理，运行不了
设置日志记录，配置参数和输入文件路径。
使用 Pandas 读取输入CSV文件，然后转置数据以便每列代表一个测试案例。
初始化 Radiomics 特征提取器，并设置提取器的参数。
循环遍历每个测试案例，提取特征并将结果存储在 Pandas 数据帧中。
将数据帧转置并将提取的特征保存到CSV文件中。
该脚本使用了 Pandas 库来处理CSV文件和数据帧，使得数据处理更加方便。
'''


def main():
  outPath = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output"  # 输出路径为指定路径下
  paramspath = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings"

  inputCSV = os.path.join(outPath, 'Task02_Heart.csv')  # 输入CSV文件路径
  outputFilepath = os.path.join(outPath, 'radiomics_features_pandas.csv')  # 输出CSV文件路径
  progress_filename = os.path.join(outPath, 'pyrad_log_pandas.txt')  # 进度日志文件路径
  params = os.path.join(paramspath, 'Params.yaml')  # 参数文件路径

  # 配置日志记录
  rLogger = logging.getLogger('radiomics')

  # 设置日志级别
  # rLogger.setLevel(logging.INFO)  # 不需要，默认的日志级别为INFO

  # 创建写入日志文件的处理器
  handler = logging.FileHandler(filename=progress_filename, mode='w')
  handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
  rLogger.addHandler(handler)

  # 初始化批处理日志记录
  logger = rLogger.getChild('batch')

  # 设置输出到标准错误的详细级别（默认级别为WARNING）
  radiomics.setVerbosity(logging.INFO)

  logger.info('pyradiomics版本：%s', radiomics.__version__)
  logger.info('加载CSV')

  # ####### 到此为止，此脚本与常规批处理脚本相同 ########

  try:
    # 使用pandas读取并转置（'.T'）输入数据
    # 需要转置，以便每列代表一个测试案例。这样更容易遍历输入案例
    flists = pandas.read_csv(inputCSV).T
  except Exception:
    logger.error('CSV读取失败', exc_info=True)
    exit(-1)

  logger.info('加载完成')
  logger.info('患者数：%d', len(flists.columns))

  if os.path.isfile(params):
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
  else:  # 找不到参数文件，使用硬编码的设置
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3]
    settings['interpolator'] = sitk.sitkBSpline
    settings['enableCExtensions'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # extractor.enableInputImages(wavelet= {'level': 2})

  logger.info('启用的输入图像类型：%s', extractor.enabledImagetypes)
  logger.info('启用的特征：%s', extractor.enabledFeatures)
  logger.info('当前设置：%s', extractor.settings)

  # 实例化一个pandas数据帧以保存所有患者的结果
  results = pandas.DataFrame()

  for entry in flists:  # 循环遍历所有列（即测试案例）
    logger.info("(%d/%d) 处理患者（图像：%s，掩模：%s）",
                entry + 1,
                len(flists),
                flists[entry]['Image'],
                flists[entry]['Mask'])

    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)

    if str(label).isdigit():
      label = int(label)
    else:
      label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = flists[entry]  # 这是一个pandas Series
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        # PyRadiomics将结果作为有序字典返回，可以很容易地转换为pandas Series
        # 字典中的键将用作索引（行标签），具有特征值的行的值。
        result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
        featureVector = featureVector.append(result)
      except Exception:
        logger.error('特征提取失败：', exc_info=True)

      # 为了将此案例的计算特征添加到我们的数据帧中，系列必须具有名称（将成为列的名称）。
      featureVector.name = entry
      # 通过指定“外部”连接，将所有计算的特征添加到数据帧中，包括以前未计算的特征。这也确保我们不会得到一个空帧，因为对于第一个患者，它与空数据帧“连接”在一起。
      results = results.join(featureVector, how='outer')  # 如果特征提取失败，结果将全部为NaN

  logger.info('提取完成，正在写入CSV')
  # .T转置数据帧，以便每一行表示一个患者，提取的特征作为列
  results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
  logger.info('CSV写入完成')


if __name__ == '__main__':
  main()
