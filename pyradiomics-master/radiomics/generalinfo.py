import collections
import logging
import sys

import numpy
import pywt
import SimpleITK as sitk

import radiomics

class GeneralInfo:
  def __init__(self):
    self.logger = logging.getLogger(self.__module__)  # 初始化日志记录器

    self.generalInfo_prefix = 'diagnostics_'  # 通用信息前缀

    self.generalInfo = collections.OrderedDict()  # 使用有序字典存储通用信息
    self.addStaticElements()  # 添加静态元素

  def getGeneralInfo(self):
    """
    返回包含所有通用信息项的字典。格式为 <info_item>:<value>，保留值的类型。
    对于 CSV 格式，这将导致转换为字符串并在必要时加引号，对于 JSON，值将被解释并存储为 JSON 字符串。
    """
    return self.generalInfo

  def addStaticElements(self):
    """
    向通用信息中添加以下元素：

    - Version: PyRadiomics 的当前版本
    - NumpyVersion: 使用的 numpy 版本
    - SimpleITKVersion: 使用的 SimpleITK 版本
    - PyWaveletVersion: 使用的 PyWavelet 版本
    - PythonVersion: 运行 PyRadiomics 的 python 解释器版本
    """

    self.generalInfo[self.generalInfo_prefix + 'Versions_PyRadiomics'] = radiomics.__version__
    self.generalInfo[self.generalInfo_prefix + 'Versions_Numpy'] = numpy.__version__
    self.generalInfo[self.generalInfo_prefix + 'Versions_SimpleITK'] = sitk.Version().VersionString()
    self.generalInfo[self.generalInfo_prefix + 'Versions_PyWavelet'] = pywt.__version__
    self.generalInfo[self.generalInfo_prefix + 'Versions_Python'] = '%i.%i.%i' % sys.version_info[:3]

  def addImageElements(self, image, prefix='original'):
    """
    计算图像的出处信息

    添加以下内容：

    - Hash: 掩码的 sha1 哈希，可用于检查在可重复性测试中是否使用了相同的掩码。（仅当前缀为 "original" 时添加）
    - Dimensionality: 图像的维度数（例如 2D, 3D）。（仅当前缀为 "original" 时添加）
    - Spacing: 像素间距（x, y, z）以毫米为单位。
    - Size: 图像的尺寸（x, y, z）以体素数为单位。
    - Mean: 图像中所有体素的平均强度值。
    - Minimum: 图像中所有体素的最小强度值。
    - Maximum: 图像中所有体素的最大强度值。

    添加前缀以指示描述的图像类型：

    - original: 加载的图像，无预处理。
    - interpolated: 图像经过重新采样到新间距后的结果（包括裁剪）。
    """
    if prefix == 'original':
      self.generalInfo[self.generalInfo_prefix + 'Image-original_Hash'] = sitk.Hash(image)
      self.generalInfo[self.generalInfo_prefix + 'Image-original_Dimensionality'] = '%iD' % image.GetDimension()

    self.generalInfo[self.generalInfo_prefix + 'Image-' + prefix + '_Spacing'] = image.GetSpacing()
    self.generalInfo[self.generalInfo_prefix + 'Image-' + prefix + '_Size'] = image.GetSize()
    im_arr = sitk.GetArrayFromImage(image).astype('float')
    self.generalInfo[self.generalInfo_prefix + 'Image-' + prefix + '_Mean'] = numpy.mean(im_arr)
    self.generalInfo[self.generalInfo_prefix + 'Image-' + prefix + '_Minimum'] = numpy.min(im_arr)
    self.generalInfo[self.generalInfo_prefix + 'Image-' + prefix + '_Maximum'] = numpy.max(im_arr)

  def addMaskElements(self, image, mask, label, prefix='original'):
    """
    计算掩码的出处信息

    添加以下内容：

    - MaskHash: 掩码的 sha1 哈希，可用于检查在可重复性测试中是否使用了相同的掩码。（仅当前缀为 "original" 时添加）
    - BoundingBox: 由指定标签定义的 ROI 的边界框：
      元素 0, 1 和 2 分别是 x, y 和 z 坐标的下界。
      元素 3, 4 和 5 分别是边界框在 x, y 和 z 方向的大小。
    - VoxelNum: 由指定标签定义的 ROI 中包含的体素数。
    - VolumeNum: 由指定标签定义的 ROI 中完全连接（26-连通性）体积的数量。
    - CenterOfMassIndex: ROI 质心的 x, y 和 z 坐标，以图像坐标空间（连续索引）表示。
    - CenterOfMass: ROI 质心的实际世界 x, y 和 z 坐标
    - ROIMean: 由指定标签定义的 ROI 中所有体素的平均强度值。
    - ROIMinimum: 由指定标签定义的 ROI 中所有体素的最小强度值。
    - ROIMaximum: 由指定标签定义的 ROI 中所有体素的最大强度值。

    添加前缀以指示描述的掩码类型：

    - original: 加载的掩码，无预处理。
    - corrected: 掩码经过 :py:func:`imageoperations.checkMask` 修正后的结果。
    - interpolated: 掩码经过重新采样到新间距后的结果（包括裁剪）。
    - resegmented: 应用重新分割后的掩码。
    """
    if mask is None:
      return

    if prefix == 'original':
      self.generalInfo[self.generalInfo_prefix + 'Mask-original_Hash'] = sitk.Hash(mask)

    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_Spacing'] = mask.GetSpacing()
    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_Size'] = mask.GetSize()

    lssif = sitk.LabelShapeStatisticsImageFilter()
    lssif.Execute(mask)

    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_BoundingBox'] = lssif.GetBoundingBox(int(label))
    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_VoxelNum'] = lssif.GetNumberOfPixels(int(label))

    labelMap = (mask == label)
    ccif = sitk.ConnectedComponentImageFilter()
    ccif.FullyConnectedOn()
    ccif.Execute(labelMap)
    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_VolumeNum'] = ccif.GetObjectCount()

    ma_arr = sitk.GetArrayFromImage(labelMap) == 1
    maskCoordinates = numpy.array(numpy.where(ma_arr))
    center_index = tuple(numpy.mean(maskCoordinates, axis=1)[::-1])  # 同时将 z, y, x 转换为 x, y, z 顺序

    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_CenterOfMassIndex'] = center_index

    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_CenterOfMass'] = mask.TransformContinuousIndexToPhysicalPoint(center_index)

    if image is None:
      return

    im_arr = sitk.GetArrayFromImage(image)
    targetvoxels = im_arr[ma_arr].astype('float')
    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_Mean'] = numpy.mean(targetvoxels)
    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_Minimum'] = numpy.min(targetvoxels)
    self.generalInfo[self.generalInfo_prefix + 'Mask-' + prefix + '_Maximum'] = numpy.max(targetvoxels)

  def addGeneralSettings(self, settings):
    """
    添加通用设置的字符串表示。
    格式为 {<settings_name>:<value>, ...}。
    """
    self.generalInfo[self.generalInfo_prefix + 'Configuration_Settings'] = settings

  def addEnabledImageTypes(self, enabledImageTypes):
    """
    添加启用的图像类型及每种图像类型的任何自定义设置的字符串表示。
    格式为 {<imageType_name>:{<setting_name>:<value>, ...}, ...}。
    """
    self.generalInfo[self.generalInfo_prefix + 'Configuration_EnabledImageTypes'] = enabledImageTypes
