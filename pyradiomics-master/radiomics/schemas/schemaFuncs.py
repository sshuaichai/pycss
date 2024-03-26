import pywt  # 导入pywt模块，用于小波变换
import six  # 导入six模块，提供Python 2和3兼容性

from radiomics import getFeatureClasses, getImageTypes  # 从radiomics导入获取特征类和图像类型的函数

# 获取并存储可用的特征类和图像类型
featureClasses = getFeatureClasses()
imageTypes = getImageTypes()

# 检查提供的小波是否有效
def checkWavelet(value, rule_obj, path):
  # 检查value是否为字符串类型
  if not isinstance(value, six.string_types):
    raise TypeError('小波不是预期的类型（字符串）')
  wavelist = pywt.wavelist()  # 获取所有可用的小波列表
  # 检查指定的小波是否在可用小波列表中
  if value not in wavelist:
    raise ValueError('小波“%s”在pyWavelets %s中不可用' % (value, wavelist))
  return True

# 检查提供的插值器是否有效
def checkInterpolator(value, rule_obj, path):
  # 如果value为None，直接返回True
  if value is None:
    return True
  # 检查value是否为字符串类型
  if isinstance(value, six.string_types):
    # 定义有效的插值器选项
    enum = {'sitkNearestNeighbor',
            'sitkLinear',
            'sitkBSpline',
            'sitkGaussian',
            'sitkLabelGaussian',
            'sitkHammingWindowedSinc',
            'sitkCosineWindowedSinc',
            'sitkWelchWindowedSinc',
            'sitkLanczosWindowedSinc',
            'sitkBlackmanWindowedSinc'}
    # 检查指定的插值器是否在有效选项中
    if value not in enum:
      raise ValueError('插值器值“%s”无效，可能的值为：%s' % (value, enum))
  elif isinstance(value, int):
    # 如果value为整数，检查其是否在有效范围内
    if value < 1 or value > 10:
      raise ValueError('插值器值%i必须在[1-10]范围内' % (value))
  else:
    raise TypeError('插值器不是预期的类型（字符串或整数）')
  return True

# 检查提供的权重是否有效
def checkWeighting(value, rule_obj, path):
  # 如果value为None，直接返回True
  if value is None:
    return True
  # 检查value是否为字符串类型
  elif isinstance(value, six.string_types):
    # 定义有效的权重选项
    enum = ['euclidean', 'manhattan', 'infinity', 'no_weighting']
    # 检查指定的权重是否在有效选项中
    if value not in enum:
      raise ValueError('WeightingNorm值“%s”无效，可能的值为：%s' % (value, enum))
  else:
    raise TypeError('WeightingNorm不是预期的类型（字符串或None）')
  return True

# 检查提供的特征类是否有效
def checkFeatureClass(value, rule_obj, path):
  global featureClasses
  # 检查value是否为None
  if value is None:
    raise TypeError('featureClass字典的值不能为None')
  # 遍历value中的每个特征类
  for className, features in six.iteritems(value):
    # 检查特征类是否在有效的特征类列表中
    if className not in featureClasses.keys():
      raise ValueError(
        '特征类%s无法识别。可用的特征类有%s' % (className, list(featureClasses.keys())))
    # 如果特征列表不为None，检查其是否为列表类型
    if features is not None:
      if not isinstance(features, list):
        raise TypeError('特征类%s的值不是预期的类型（列表）' % (className))
      # 检查特征列表中是否有无法识别的特征
      unrecognizedFeatures = set(features) - set(featureClasses[className].getFeatureNames())
      if len(unrecognizedFeatures) > 0:
        raise ValueError('特征类%s包含无法识别的特征：%s' % (className, str(unrecognizedFeatures)))

  return True

# 检查提供的图像类型是否有效
def checkImageType(value, rule_obj, path):
  global imageTypes
  # 检查value是否为None
  if value is None:
    raise TypeError('imageType字典的值不能为None')
  # 遍历value中的每个图像类型
  for im_type in value:
    # 检查图像类型是否在有效的图像类型列表中
    if im_type not in imageTypes:
      raise ValueError('图像类型%s无法识别。可用的图像类型有%s' %
                       (im_type, imageTypes))

  return True
