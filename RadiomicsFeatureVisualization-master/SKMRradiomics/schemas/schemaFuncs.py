# 导入必要的库
import pywt  # 导入pywt模块，用于小波变换
import six  # 导入six模块，提供Python 2和3兼容性

# 从SKMRradiomics模块导入获取特征类和图像类型的函数
from SKMRradiomics import getFeatureClasses, getImageTypes

# 获取并初始化特征类和图像类型
featureClasses = getFeatureClasses()  # 获取所有可用的特征类
imageTypes = getImageTypes()  # 获取所有可用的图像类型

def checkWavelet(value, rule_obj, path):
  # 检查给定的小波是否有效
  if not isinstance(value, six.string_types):
    raise TypeError('小波非期望类型(str)')  # 如果value不是字符串类型，则抛出TypeError
  wavelist = pywt.wavelist()  # 获取pywt支持的所有小波列表
  if value not in wavelist:
    raise ValueError('小波"%s"在pywavelet %s中不可用' % (value, wavelist))  # 如果指定的小波不在支持的列表中，则抛出ValueError
  return True  # 如果检查通过，则返回True

def checkInterpolator(value, rule_obj, path):
  # 检查插值方法是否有效
  if value is None:
    return True  # 如果value为None，则直接返回True，表示不需要插值
  if isinstance(value, six.string_types):
    # 定义支持的插值方法
    enum = {'sitkNearestNeighbor', 'sitkLinear', 'sitkBSpline', 'sitkGaussian', 'sitkLabelGaussian',
            'sitkHammingWindowedSinc', 'sitkCosineWindowedSinc', 'sitkWelchWindowedSinc',
            'sitkLanczosWindowedSinc', 'sitkBlackmanWindowedSinc'}
    if value not in enum:
      raise ValueError('插值值“%s”无效，可能的值:%s' % (value, enum))  # 如果指定的插值方法不在支持的列表中，则抛出ValueError
  elif isinstance(value, int):
    if value < 1 or value > 10:
      raise ValueError('插入值%i，必须在[1-10]的范围内' % (value))  # 如果指定的插值方法编号不在1到10之间，则抛出ValueError
  else:
    raise TypeError('插入器不是期望的类型(str或int)')  # 如果value既不是字符串也不是整数，则抛出TypeError
  return True  # 如果检查通过，则返回True

def checkWeighting(value, rule_obj, path):
  # 检查权重是否有效
  if value is None:
    return True  # 如果value为None，则直接返回True，表示不需要权重
  elif isinstance(value, six.string_types):
    # 定义支持的权重类型
    enum = ['euclidean', 'manhattan', 'infinity', 'no_weighting']
    if value not in enum:
      raise ValueError('WeightingNorm 值“%s”无效，可能的值：%s' % (value, enum))  # 如果指定的权重类型不在支持的列表中，则抛出ValueError
  else:
    raise TypeError('WeightingNorm 不是预期类型（str 或 None）')  # 如果value既不是字符串也不是None，则抛出TypeError
  return True  # 如果检查通过，则返回True

def checkFeatureClass(value, rule_obj, path):
  # 检查特征类是否有效
  global featureClasses
  if value is None:
    raise TypeError('featureClass 字典不能为 None 值')  # 如果value为None，则抛出TypeError
  for className, features in six.iteritems(value):
    if className not in featureClasses.keys():
      raise ValueError('要素类 %s 未被识别。可用要素类为 %s' % (className, list(featureClasses.keys())))  # 如果指定的特征类不在支持的列表中，则抛出ValueError
    if features is not None:
      if not isinstance(features, list):
        raise TypeError('特征类%s的值不是期望的类型(列表)' % (className))  # 如果特征值不是列表类型，则抛出TypeError
      unrecognizedFeatures = set(features) - set(featureClasses[className].getFeatureNames())
      if len(unrecognizedFeatures) > 0:
        raise ValueError('功能类%s包含无法识别的功能:%s' % (className, str(unrecognizedFeatures)))  # 如果存在不被识别的特征，则抛出ValueError
  return True  # 如果检查通过，则返回True

def checkImageType(value, rule_obj, path):
  # 检查图像类型是否有效
  global imageTypes
  if value is None:
    raise TypeError('imageType字典不能为None值')  # 如果value为None，则抛出TypeError
  for im_type in value:
    if im_type not in imageTypes:
      raise ValueError('无法识别图像类型%s。可用的映像类型为%s' % (im_type, imageTypes))  # 如果指定的图像类型不在支持的列表中，则抛出ValueError
  return True  # 如果检查通过，则返回True
