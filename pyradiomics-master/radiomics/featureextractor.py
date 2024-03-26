# -*- coding: utf-8 -*-  # 指定文件编码为utf-8
from __future__ import print_function  # 从__future__模块导入print_function特性，使得print作为函数使用

import collections  # 导入collections模块，提供了许多有用的集合类
from itertools import chain  # 从itertools模块导入chain函数，用于对多个迭代器进行串联
import json  # 导入json模块，用于处理JSON数据
import logging  # 导入logging模块，用于记录日志
import os  # 导入os模块，提供了与操作系统交互的功能
import pathlib  # 导入pathlib模块，提供面向对象的文件系统路径操作

import pykwalify.core  # 导入pykwalify.core模块，用于基于schema验证YAML和JSON数据
import SimpleITK as sitk  # 导入SimpleITK模块并重命名为sitk，用于医学图像处理
import six  # 导入six模块，提供了Python 2和3之间的兼容性函数

from radiomics import generalinfo, getFeatureClasses, getImageTypes, getParameterValidationFiles, imageoperations

# 从radiomics包导入多个模块和函数，用于放射组学特征提取

logger = logging.getLogger(__name__)  # 创建或获取一个名为__name__的logger
geometryTolerance = None  # 初始化geometryTolerance变量为None


class RadiomicsFeatureExtractor:  # 定义RadiomicsFeatureExtractor类
  r"""
  用于计算放射组学签名的包装类。
  在初始化及之后，可以使用各种设置来自定义结果签名。
  这包括要使用的类和特征，以及在预处理图像方面应该做什么
  以及应作为输入使用哪些图像（原始和/或过滤）。

  然后调用 :py:func:`execute` 为传递的图像和标签图组合生成由这些设置指定的放射组学
  签名。这个函数可以在批处理过程中重复调用，以计算所有图像和标签图组合的放射组学签名。

  在初始化时，可以提供一个参数文件（指向yaml或json结构化文件的字符串）或字典
  包含所有必要的设置（顶层包含键 "setting", "imageType" 和/或 "featureClass"）。这是
  通过将其作为第一个位置参数传递完成的。如果没有提供位置参数，或者参数不是
  字典或指向有效文件的字符串，则将应用默认值。
  此外，在初始化时，可以提供自定义设置（*不启用图像类型和/或特征类*）
  作为关键字参数，设置名称作为键，其值作为参数值（例如 ``binWidth=25``）。
  这里指定的设置将覆盖参数文件/字典/默认设置中的设置。
  有关可能的设置和自定义的更多信息，请参见
  :ref:`Customizing the Extraction <radiomics-customization-label>`。

  默认情况下，所有特征类中的所有特征都被启用。
  默认情况下，只启用了“Original”输入图像（未应用过滤器）。
  """

  def __init__(self, *args, **kwargs):  # 初始化函数，接受任意数量的位置参数和关键字参数
    global logger  # 使用global声明，表明函数内部将使用全局变量logger

    self.settings = {}  # 初始化settings属性为一个空字典
    self.enabledImagetypes = {}  # 初始化enabledImagetypes属性为一个空字典
    self.enabledFeatures = {}  # 初始化enabledFeatures属性为一个空字典

    self.featureClassNames = list(getFeatureClasses().keys())  # 获取并存储所有可用的特征类名称

    if len(args) == 1 and isinstance(args[0], dict):  # 如果提供了一个位置参数且该参数是字典
      logger.info("Loading parameter dictionary")  # 记录日志
      self._applyParams(paramsDict=args[0])  # 调用_applyParams函数，传入参数字典
    elif len(args) == 1 and (isinstance(args[0], six.string_types) or isinstance(args[0],
                                                                                 pathlib.PurePath)):  # 如果提供了一个位置参数且该参数是字符串或pathlib.PurePath实例
      if not os.path.isfile(args[0]):  # 检查参数指定的文件是否存在
        raise IOError("Parameter file %s does not exist." % args[0])  # 如果文件不存在，抛出IOError异常
      logger.info("Loading parameter file %s", str(args[0]))  # 记录日志
      self._applyParams(paramsFile=args[0])  # 调用_applyParams函数，传入参数文件路径
    else:  # 如果没有提供有效的位置参数
      # 设置默认设置并使用kwargs中包含的更改设置进行更新
      self.settings = self._getDefaultSettings()  # 获取并设置默认配置
      logger.info('No valid config parameter, using defaults: %s', self.settings)  # 记录日志

      self.enabledImagetypes = {'Original': {}}  # 默认启用“Original”图像类型
      logger.info('Enabled image types: %s', self.enabledImagetypes)  # 记录日志

      for featureClassName in self.featureClassNames:  # 遍历所有特征类名称
        if featureClassName == 'shape2D':  # 不默认启用shape2D特征类
          continue
        self.enabledFeatures[featureClassName] = []  # 默认启用其他所有特征类
      logger.info('Enabled features: %s', self.enabledFeatures)  # 记录日志

    if len(kwargs) > 0:  # 如果提供了关键字参数
      logger.info('Applying custom setting overrides: %s', kwargs)  # 记录日志
      self.settings.update(kwargs)  # 使用关键字参数更新设置
      logger.debug("Settings: %s", self.settings)  # 记录详细日志

    if self.settings.get('binCount', None) is not None:  # 如果设置了binCount
      logger.warning('Fixed bin Count enabled! However, we recommend using a fixed bin Width. See '
                     'http://pyradiomics.readthedocs.io/en/latest/faq.html#radiomics-fixed-bin-width for more '
                     'details')  # 记录警告日志

    self._setTolerance()  # 调用_setTolerance函数，设置几何公差

  def _setTolerance(self):
    global geometryTolerance, logger
    geometryTolerance = self.settings.get('geometryTolerance')
    if geometryTolerance is not None:
      logger.debug('Setting SimpleITK tolerance to %s', geometryTolerance)
      sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(geometryTolerance)
      sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(geometryTolerance)

  def addProvenance(self, provenance_on=True):
    """
    启用或禁用提取过程中额外信息的报告。这些信息包括工具箱版本、
    启用的输入图像和应用的设置。此外，还提供了图像和感兴趣区域（ROI）
    的额外信息，包括原始图像间距、ROI中的体素总数和ROI中完全连接体积的总数。

    要禁用此功能，请调用 ``addProvenance(False)``。
    """
    self.settings['additionalInfo'] = provenance_on

  @staticmethod
  def _getDefaultSettings():
    """
    返回此类中指定的默认设置的字典。这些设置覆盖全局设置，
    如 ``additionalInfo``，以及图像预处理设置（例如，重采样）。特定于特征类的
    设置在各自的特征类中定义，不包括在此处。类似地，特定于过滤器的设置在
    ``imageoperations.py``中定义，也不包括在此处。
    """
    return {'minimumROIDimensions': 2,
            'minimumROISize': None,  # 默认情况下跳过测试ROI大小
            'normalize': False,
            'normalizeScale': 1,
            'removeOutliers': None,
            'resampledPixelSpacing': None,  # 默认不进行重采样
            'interpolator': 'sitkBSpline',  # 替代方案: sitk.sitkBSpline
            'preCrop': False,
            'padDistance': 5,
            'distances': [1],
            'force2D': False,
            'force2Ddimension': 0,
            'resegmentRange': None,  # 默认不进行重新分割
            'label': 1,
            'additionalInfo': True}

  def loadParams(self, paramsFile):
    """
    解析指定的参数文件，并使用它来更新设置、启用的特征（类）和图像类型。有关参数文件结构的更多信息，请参见
    :ref:`Customizing the extraction <radiomics-customization-label>`。

    如果提供的文件不符合要求（即，未识别的名称或设置的无效值），将引发pykwalify错误。
    """
    self._applyParams(paramsFile=paramsFile)

  def loadJSONParams(self, JSON_configuration):
    """
    解析JSON结构化配置字符串，并使用它来更新设置、启用的特征（类）和图像类型。
    有关参数文件结构的更多信息，请参见
    :ref:`Customizing the extraction <radiomics-customization-label>`。

    如果提供的字符串不符合要求（即，未识别的名称或设置的无效值），将引发pykwalify错误。
    """
    parameter_data = json.loads(JSON_configuration)
    self._applyParams(paramsDict=parameter_data)

  def _applyParams(self, paramsFile=None, paramsDict=None):
    """
    验证并应用参数字典。有关更多信息，请参见 :py:func:`loadParams` 和 :py:func:`loadJSONParams`。
    """
    global logger

    # 确保pykwalify.core有一个日志处理程序（在参数验证失败时需要）
    if len(pykwalify.core.log.handlers) == 0 and len(logging.getLogger().handlers) == 0:
      # 对于pykwalify或根记录器都没有可用的处理程序，提供第一个radiomics处理程序（输出到stderr）
      pykwalify.core.log.addHandler(logging.getLogger('radiomics').handlers[0])

    schemaFile, schemaFuncs = getParameterValidationFiles()
    c = pykwalify.core.Core(source_file=paramsFile, source_data=paramsDict,
                            schema_files=[schemaFile], extensions=[schemaFuncs])
    params = c.validate()
    logger.debug('Parameters parsed, input is valid.')

    enabledImageTypes = params.get('imageType', {})
    enabledFeatures = params.get('featureClass', {})
    settings = params.get('setting', {})
    voxelSettings = params.get('voxelSetting', {})

    logger.debug("Applying settings")

    if len(enabledImageTypes) == 0:
      self.enabledImagetypes = {'Original': {}}
    else:
      self.enabledImagetypes = enabledImageTypes

    logger.debug("Enabled image types: %s", self.enabledImagetypes)

    if len(enabledFeatures) == 0:
      self.enabledFeatures = {}
      for featureClassName in self.featureClassNames:
        self.enabledFeatures[featureClassName] = []
    else:
      self.enabledFeatures = enabledFeatures

    logger.debug("Enabled features: %s", self.enabledFeatures)

    # 设置默认设置并使用kwargs中包含的更改设置进行更新
    self.settings = self._getDefaultSettings()
    self.settings.update(settings)
    self.settings.update(voxelSettings)

    logger.debug("Settings: %s", settings)


"""
      计算提供的图像和掩码组合的放射组学签名。它包括以下步骤：

      1. 加载并必要时对图像和掩码进行标准化/重采样。
      2. 使用 :py:func:`~imageoperations.checkMask` 检查ROI的有效性，并计算并返回边界框。
      3. 如果启用，计算并作为结果的一部分存储提取信息。（在基于体素的提取中不可用）
      4. 在原始图像的裁剪（无填充）版本上计算形状特征。（在基于体素的提取中不可用）
      5. 如果启用，根据 ``resegmentRange`` 指定的范围重新分割掩码（默认为None：禁用重新分割）。
      6. 使用 ``_enabledImageTypes`` 中指定的所有图像类型计算其他启用的特征类。在应用任何过滤器并在传递给特征类之前，图像被裁剪到肿瘤掩码（无填充）。
      7. 返回计算的特征作为 ``collections.OrderedDict``。

      :param imageFilepath: SimpleITK 图像，或指向图像文件位置的字符串
      :param maskFilepath: SimpleITK 图像，或指向标签图文件位置的字符串
      :param label: 整数，要提取特征的标签值。如果未指定，使用最后指定的标签。默认标签为1。
      :param label_channel: 整数，当maskFilepath产生一个具有向量像素类型的SimpleITK.Image时使用的通道索引。默认索引为0。
      :param voxelBased: 布尔值，默认为False。如果设置为true，则执行基于体素的提取，否则执行基于段的提取。
      :returns: 包含计算签名的字典（"<imageType>_<featureClass>_<featureName>":value）。
          在基于段的提取中，特征的值类型为float，如果是基于体素的，类型为SimpleITK.Image。
          诊断特征的类型不同，但总是可以表示为字符串。
"""
def execute(self, imageFilepath, maskFilepath, label=None, label_channel=None, voxelBased=False):
    global geometryTolerance, logger
    _settings = self.settings.copy()

    tolerance = _settings.get('geometryTolerance')
    additionalInfo = _settings.get('additionalInfo', False)
    resegmentShape = _settings.get('resegmentShape', False)

    if label is not None:
      _settings['label'] = label
    else:
      label = _settings.get('label', 1)

    if label_channel is not None:
      _settings['label_channel'] = label_channel

    if geometryTolerance != tolerance:
      self._setTolerance()

    if additionalInfo:
      generalInfo = generalinfo.GeneralInfo()
      generalInfo.addGeneralSettings(_settings)
      generalInfo.addEnabledImageTypes(self.enabledImagetypes)
    else:
      generalInfo = None

    if voxelBased:
      _settings['voxelBased'] = True
      kernelRadius = _settings.get('kernelRadius', 1)
      logger.info('开始基于体素的提取')
    else:
      kernelRadius = 0

    logger.info('计算标签为: %d 的特征', label)
    logger.debug('启用的图像类型: %s', self.enabledImagetypes)
    logger.debug('启用的特征: %s', self.enabledFeatures)
    logger.debug('当前设置: %s', _settings)

    # 1. 加载图像和掩码
    featureVector = collections.OrderedDict()
    image, mask = self.loadImage(imageFilepath, maskFilepath, generalInfo, **_settings)

    # 2. 检查加载的掩码是否包含有效的ROI以进行特征提取并获取边界框
    # 如果ROI无效，则引发ValueError
    boundingBox, correctedMask = imageoperations.checkMask(image, mask, **_settings)

    # 如果需要对掩码进行重采样，则更新掩码
    if correctedMask is not None:
      if generalInfo is not None:
        generalInfo.addMaskElements(image, correctedMask, label, 'corrected')
      mask = correctedMask

    logger.debug('图像和掩码加载且有效，开始提取')

    # 5. 如果启用（参数resegmentMask不为None），则重新分割掩码
    resegmentedMask = None
    if _settings.get('resegmentRange', None) is not None:
      resegmentedMask = imageoperations.resegmentMask(image, mask, **_settings)

      # 重新检查以确认掩码仍然有效，如果不是，则引发ValueError
      boundingBox, correctedMask = imageoperations.checkMask(image, resegmentedMask, **_settings)

      if generalInfo is not None:
        generalInfo.addMaskElements(image, resegmentedMask, label, 'resegmented')

    # 3. 如果启用，则添加额外的信息
    if generalInfo is not None:
      featureVector.update(generalInfo.getGeneralInfo())

    # 如果resegmentShape为True且启用了重新分割，则在此处更新掩码，也用重新分割的掩码计算形状（例如，PET重新分割）
    if resegmentShape and resegmentedMask is not None:
      mask = resegmentedMask

    if not voxelBased:
      # 4. 如果应计算形状描述符，则在此处单独处理
      featureVector.update(self.computeShape(image, mask, boundingBox, **_settings))

    # （默认）仅对形状以外的特征类使用重新分割的掩码
    # 可以通过指定`resegmentShape` = True来覆盖
    if not resegmentShape and resegmentedMask is not None:
      mask = resegmentedMask

    # 6. 使用启用的图像类型计算其他启用的特征类
    # 为所有启用的图像类型创建生成器
    logger.debug('创建图像类型迭代器')
    imageGenerators = []
    for imageType, customKwargs in six.iteritems(self.enabledImagetypes):
      args = _settings.copy()
      args.update(customKwargs)
      logger.info('添加图像类型 "%s" 与自定义设置: %s' % (imageType, str(customKwargs)))
      imageGenerators = chain(imageGenerators, getattr(imageoperations, 'get%sImage' % imageType)(image, mask, **args))

    logger.debug('提取特征')
    # 计算生成器中所有（过滤的）图像的特征
    for inputImage, imageTypeName, inputKwargs in imageGenerators:
      logger.info('计算 %s 图像的特征', imageTypeName)
      inputImage, inputMask = imageoperations.cropToTumorMask(inputImage, mask, boundingBox, padDistance=kernelRadius)
      featureVector.update(self.computeFeatures(inputImage, inputMask, imageTypeName, **inputKwargs))

    logger.debug('特征提取完成')

    return featureVector


    """
    加载并预处理图像和标签图。
    如果ImageFilePath是字符串，则作为SimpleITK图像加载并赋值给``image``，
    如果它已经是SimpleITK图像，则直接赋值给``image``。
    其他情况将被忽略（不进行计算）。
    使用MaskFilePath为``mask``赋值采用相同的方法。如有必要，将分割对象（即具有向量图像类型的掩码体积）
    转换为标签图（=标量图像类型）。数据类型强制为UInt32。
    参见 :py:func:`~imageoperations.getMask()`。

    如果启用了标准化，则在应用任何重采样之前首先对图像进行标准化。

    如果启用了重采样，则在分配图像和掩码之后，对图像和掩码进行重采样并裁剪到肿瘤掩码（额外填充由padDistance指定）。

    :param ImageFilePath: SimpleITK.Image对象或指向SimpleITK可读文件的字符串，表示要使用的图像。
    :param MaskFilePath: SimpleITK.Image对象或指向SimpleITK可读文件的字符串，表示要使用的掩码。
    :param generalInfo: GeneralInfo对象。如果提供，用于存储预处理的诊断信息。
    :param kwargs: 字典，包含要用于此特定图像类型的设置。
    :return: 分别代表加载的图像和掩码的2个SimpleITK.Image对象。
    """

    # @staticmethod
def loadImage(ImageFilePath, MaskFilePath, generalInfo=None, **kwargs):
    global logger
    normalize = kwargs.get('normalize', False)
    interpolator = kwargs.get('interpolator')
    resampledPixelSpacing = kwargs.get('resampledPixelSpacing')
    preCrop = kwargs.get('preCrop', False)
    label = kwargs.get('label', 1)

    logger.info('加载图像和掩码')
    if isinstance(ImageFilePath, six.string_types) and os.path.isfile(ImageFilePath):
      image = sitk.ReadImage(ImageFilePath)
    elif isinstance(ImageFilePath, sitk.SimpleITK.Image):
      image = ImageFilePath
    else:
      raise ValueError('读取图像文件路径或SimpleITK对象时出错')

    if isinstance(MaskFilePath, six.string_types) and os.path.isfile(MaskFilePath):
      mask = sitk.ReadImage(MaskFilePath)
    elif isinstance(MaskFilePath, sitk.SimpleITK.Image):
      mask = MaskFilePath
    else:
      raise ValueError('读取掩码文件路径或SimpleITK对象时出错')

    # 处理掩码
    mask = imageoperations.getMask(mask, **kwargs)

    if generalInfo is not None:
      generalInfo.addImageElements(image)
      # 不在此处包含图像，因为尚未检查图像和掩码之间的重叠
      # 因此，图像和掩码可能不对齐，甚至可能有不同的大小。
      generalInfo.addMaskElements(None, mask, label)

    # 只有在图像和掩码正确加载的情况下才会到达这一点
    if normalize:
      image = imageoperations.normalizeImage(image, **kwargs)

    if interpolator is not None and resampledPixelSpacing is not None:
      image, mask = imageoperations.resampleImage(image, mask, **kwargs)
      if generalInfo is not None:
        generalInfo.addImageElements(image, '插值后')
        generalInfo.addMaskElements(image, mask, label, '插值后')

    elif preCrop:
      bb, correctedMask = imageoperations.checkMask(image, mask, **kwargs)
      if correctedMask is not None:
        # 如果需要对掩码进行重采样，则更新掩码
        mask = correctedMask
      if bb is None:
        # 掩码检查失败
        raise ValueError('预裁剪期间掩码检查失败')

      image, mask = imageoperations.cropToTumorMask(image, mask, bb, **kwargs)

    return image, mask

    """
    计算传递的图像和掩码的形状（2D和/或3D）特征。

    :param image: 表示使用的图像的SimpleITK.Image对象
    :param mask: 表示使用的掩码的SimpleITK.Image对象
    :param boundingBox: 由 :py:func:`~imageoperations.checkMask()` 计算的边界框，即每个维度的下界（偶数索引）和上界（奇数索引）的元组。
    :param kwargs: 包含要使用的设置的字典。
    :return: 包含计算的形状特征的collections.OrderedDict。如果没有计算特征，则返回一个空的OrderedDict。
    """
def computeShape(self, image, mask, boundingBox, **kwargs):
    global logger
    featureVector = collections.OrderedDict()

    enabledFeatures = self.enabledFeatures

    croppedImage, croppedMask = imageoperations.cropToTumorMask(image, mask, boundingBox)

    # 定义临时函数来计算形状特征
    def compute(shape_type):
      logger.info('计算 %s', shape_type)
      featureNames = enabledFeatures[shape_type]
      shapeClass = getFeatureClasses()[shape_type](croppedImage, croppedMask, **kwargs)

      if featureNames is not None:
        for feature in featureNames:
          shapeClass.enableFeatureByName(feature)

      for (featureName, featureValue) in six.iteritems(shapeClass.execute()):
        newFeatureName = 'original_%s_%s' % (shape_type, featureName)
        featureVector[newFeatureName] = featureValue

    Nd = mask.GetDimension()
    if 'shape' in enabledFeatures.keys():
      if Nd == 3:
        compute('shape')
      else:
        logger.warning('形状特征仅适用于3D输入（对于2D输入，请使用shape2D）。发现%iD输入', Nd)

    if 'shape2D' in enabledFeatures.keys():
      if Nd == 3:
        force2D = kwargs.get('force2D', False)
        force2Ddimension = kwargs.get('force2Ddimension', 0)
        if not force2D:
          logger.warning('参数force2D必须设置为True以启用shape2D提取')
        elif not (boundingBox[1::2] - boundingBox[0::2] + 1)[force2Ddimension] > 1:
          logger.warning('指定的2D维度(%i)大小大于1，无法计算2D形状', force2Ddimension)
        else:
          compute('shape2D')
      elif Nd == 2:
        compute('shape2D')
      else:
        logger.warning('Shape2D特征仅适用于2D和3D（force2D=True）输入。发现%iD输入', Nd)

    return featureVector

    r"""
    使用图像、掩码和\*\*kwargs设置计算签名。

    此函数仅计算传递的图像（原始或派生）的签名，不对传递的图像进行预处理或应用滤镜。用于计算签名的特征/类在``self.enabledFeatures``中定义。另见 :py:func:`enableFeaturesByName`。

    :param image: 裁剪（并可选过滤）的SimpleITK.Image对象，表示使用的图像
    :param mask: 裁剪的SimpleITK.Image对象，表示使用的掩码
    :param imageTypeName: 指定应用于图像的滤镜的字符串，或者如果没有应用滤镜则为"original"。
    :param kwargs: 包含此特定图像类型要使用的设置的字典。
    :return: 包含所有启用类计算的特征的collections.OrderedDict。如果没有计算特征，则返回一个空的OrderedDict。

    .. note::

      形状描述符与灰度级无关，因此单独计算（在`execute`中处理）。在此函数中，不计算形状特征。
    """
def computeFeatures(self, image, mask, imageTypeName, **kwargs):
    global logger
    featureVector = collections.OrderedDict()
    featureClasses = getFeatureClasses()

    enabledFeatures = self.enabledFeatures

    # 计算特征类
    for featureClassName, featureNames in six.iteritems(enabledFeatures):
      # 单独处理形状特征的计算
      if featureClassName.startswith('shape'):
        continue

      if featureClassName in featureClasses:
        logger.info('计算 %s', featureClassName)

        featureClass = featureClasses[featureClassName](image, mask, **kwargs)

        if featureNames is not None:
          for feature in featureNames:
            featureClass.enableFeatureByName(feature)

        for (featureName, featureValue) in six.iteritems(featureClass.execute()):
          newFeatureName = '%s_%s_%s' % (imageTypeName, featureClassName, featureName)
          featureVector[newFeatureName] = featureValue

    return featureVector

    """
    启用所有可能的图像类型，不带任何自定义设置。
    """
def enableAllImageTypes(self):
    global logger

    logger.debug('启用所有图像类型')
    for imageType in getImageTypes():
      self.enabledImagetypes[imageType] = {}
    logger.debug('启用的图像类型: %s', self.enabledImagetypes)

    """
    禁用所有图像类型。
    """
def disableAllImageTypes(self):
    global logger

    logger.debug('禁用所有图像类型')
    self.enabledImagetypes = {}

    r"""
    启用或禁用指定的图像类型。如果启用图像类型，可以在customArgs中指定可选的自定义设置。

    当前可能的图像类型包括：

    - Original: 未应用滤镜
    - Wavelet: 小波滤波，产生每级8个分解（在三个维度中应用高通或低通滤波的所有可能组合）。
      另见 :py:func:`~radiomics.imageoperations.getWaveletImage`
    - LoG: 高斯拉普拉斯滤波器，边缘增强滤波器。强调灰度级变化区域，其中sigma定义了应强调的纹理的粗糙程度。低sigma值强调细纹理（短距离内的变化），高sigma值强调粗纹理（大距离内的灰度级变化）。
      另见 :py:func:`~radiomics.imageoperations.getLoGImage`
    - Square: 对图像强度取平方并线性缩放回原始范围。原始图像中的负值在应用滤波后再次变为负值。
    - SquareRoot: 对绝对图像强度取平方根并缩放回原始范围。原始图像中的负值在应用滤波后再次变为负值。
    - Logarithm: 对绝对强度+1取对数。值缩放到原始范围，原始图像中的负值在应用滤波后再次变为负值。
    - Exponential: 取指数，其中过滤强度为e^(绝对强度)。值缩放到原始范围，原始图像中的负值在应用滤波后再次变为负值。
    - Gradient: 返回梯度幅度。
    - LBP2D: 计算并返回2D中应用的局部二值模式。
    - LBP3D: 使用球面谐波在3D中计算并返回局部二值模式图。最后返回的图像是相应的峭度图。

    关于square, squareroot, logarithm和exponential的数学公式，参见它们在 :ref:`imageoperations<radiomics-imageoperations-label>` 中的相应函数
    (:py:func:`~radiomics.imageoperations.getSquareImage`,
    :py:func:`~radiomics.imageoperations.getSquareRootImage`,
    :py:func:`~radiomics.imageoperations.getLogarithmImage`,
    :py:func:`~radiomics.imageoperations.getExponentialImage`,
    :py:func:`~radiomics.imageoperations.getGradientImage`,
    :py:func:`~radiomics.imageoperations.getLBP2DImage` 和
    :py:func:`~radiomics.imageoperations.getLBP3DImage`,
    分别)。
    """
def enableImageTypeByName(self, imageType, enabled=True, customArgs=None):
    global logger

    if imageType not in getImageTypes():
      logger.warning('图像类型 %s 无法识别', imageType)
      return

    if enabled:
      if customArgs is None:
        customArgs = {}
        logger.debug('启用图像类型 %s（无额外自定义设置）', imageType)
      else:
        logger.debug('启用图像类型 %s（额外自定义设置：%s）', imageType, customArgs)
      self.enabledImagetypes[imageType] = customArgs
    elif imageType in self.enabledImagetypes:
      logger.debug('禁用图像类型 %s', imageType)
      del self.enabledImagetypes[imageType]
    logger.debug('启用的图像类型：%s', self.enabledImagetypes)

def enableImageTypes(self, **enabledImagetypes):
    """
    启用输入图像，并可选地应用自定义设置，这些设置将应用于相应的输入图像。
    这里指定的设置将覆盖kwargs中的设置。
    以下设置不可自定义：

    - interpolator
    - resampledPixelSpacing
    - padDistance

    更新当前设置：如有必要，启用输入图像。始终覆盖在inputImages中传递的输入图像的自定义设置。
    要禁用输入图像，请使用 :py:func:`enableInputImageByName` 或 :py:func:`disableAllInputImages`。

    :param enabledImagetypes: 字典，键是图像类型（original, wavelet 或 log）和值是自定义设置（字典）
    """
    global logger

    logger.debug('使用 %s 更新启用的图像类型', enabledImagetypes)
    self.enabledImagetypes.update(enabledImagetypes)
    logger.debug('启用的图像类型: %s', self.enabledImagetypes)


    """
    启用所有类和所有特征。

    .. note::
      通过此函数不启用已标记为"deprecated"的单个特征。它们仍然可以通过调用 :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`、
      :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()` 或在参数文件中（通过名称指定特征，而不是启用所有特征时）手动启用。
      但在大多数情况下，这仍然只会导致弃用警告。
    """
def enableAllFeatures(self):
    global logger

    logger.debug('在所有特征类中启用所有特征')
    for featureClassName in self.featureClassNames:
      self.enabledFeatures[featureClassName] = []
    logger.debug('启用的特征: %s', self.enabledFeatures)


    """
    禁用所有类。
    """
def disableAllFeatures(self):
    global logger

    logger.debug('禁用所有特征类')
    self.enabledFeatures = {}


"""
    启用或禁用给定类中的所有特征。

    .. note::
      通过此函数不启用已标记为"deprecated"的单个特征。它们仍然可以通过调用 :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`、
      :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()` 或在参数文件中（通过名称指定特征，而不是启用所有特征时）手动启用。
      但在大多数情况下，这仍然只会导致弃用警告。
"""
def enableFeatureClassByName(self, featureClass, enabled=True):
    global logger

    if featureClass not in self.featureClassNames:
      logger.warning('特征类 %s 无法识别', featureClass)
      return

    if enabled:
      logger.debug('在类 %s 中启用所有特征', featureClass)
      self.enabledFeatures[featureClass] = []
    elif featureClass in self.enabledFeatures:
      logger.debug('禁用特征类 %s', featureClass)
      del self.enabledFeatures[featureClass]
    logger.debug('启用的特征: %s', self.enabledFeatures)


"""
    指定要启用的特征。键是特征类名称，值是启用的特征名称列表。

    要为一个类启用所有特征，请为该类名提供一个空列表或None作为值。
    在enabledFeatures.keys中指定的特征类的设置被更新，未在enabledFeatures.keys中出现的特征类的设置被添加。
    要禁用整个类，请使用 :py:func:`disableAllFeatures` 或 :py:func:`enableFeatureClassByName`。
    """
def enableFeaturesByName(self, **enabledFeatures):
    global logger

    logger.debug('使用 %s 更新启用的特征', enabledFeatures)
    self.enabledFeatures.update(enabledFeatures)
    logger.debug('启用的特征: %s', self.enabledFeatures)



