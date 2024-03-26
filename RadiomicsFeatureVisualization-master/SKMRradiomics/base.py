import inspect
import logging
import traceback

import numpy
import SimpleITK as sitk
import six

from . import cMatrices, deprecated, getProgressReporter, imageoperations


class RadiomicsFeaturesBase(object):
  """
  这是一个抽象类，定义了特征类的共同接口。所有特征类都继承（直接或间接）自这个类。

  在初始化时，作为 SimpleITK 图像对象传递图像和标签图像（分别为``inputImage``和``inputMask``）。
  使用 SimpleITK 图像作为输入的动机是为了保持将来在 SimpleITK 中实现的优化特征计算器的可重用性。
  如果图像或掩膜为 None，则初始化失败并记录警告（不会引发错误）。

  使用父 'radiomics' 记录器设置日志记录。这样生成的日志会保留工具箱的结构。
  子记录器的名称采用包含特征类的模块的名称（例如 'radiomics.glcm'）。

  可通过重写``_initSegmentBasedCalculation``函数添加在调用特征函数之前需要的任何预计算，
  该函数为特征提取准备输入。如果需要图像离散化，则可以通过在初始化函数中添加对 ``_applyBinning`` 的调用来实现，
  该函数还实例化了包含 ROI 内最大（'Ng'）和唯一（'GrayLevels'）灰度级的系数。此函数还实例化了 `matrix` 变量，
  该变量保存离散化图像（`imageArray` 变量仅保存原始灰度级）。

  初始化时实例化以下变量：

  - kwargs: 包含传递给此特征类的所有自定义设置的字典。
  - label: 标签图像（ROI）中的标签值。如果键不存在，则使用默认值 1。
  - featureNames: 包含特征类中定义的特征名称的列表。参见 :py:func:`getFeatureNames`
  - inputImage: 输入图像的 SimpleITK 图像对象（维度 x、y、z）

  ``_initSegmentBasedCalculation``函数实例化以下变量：

  - inputMask: 输入标签图像的 SimpleITK 图像对象（维度 x、y、z）
  - imageArray: 输入图像中灰度值的 numpy 数组（维度 z、y、x）
  - maskArray: 布尔类型的 numpy 数组，其中元素设置为 ``True``（标签图像 = 标签），否则为 ``False``（维度 z、y、x）。
  - labelledVoxelCoordinates: 包含 ROI 内体素的 z、x 和 y 坐标的 3 个 numpy 数组的元组。每个数组的长度等于 ROI 内的总体素数。
  - boundingBoxSize: 包含 ROI 边界框的 z、x 和 y 尺寸的 3 个整数的元组。
  - matrix: imageArray 变量的副本，其中 ROI 内的灰度值使用指定的 binWidth 进行离散化。
    仅在特征类中的 ``_initSegmentBasedCalculation`` 的重写中添加了对 ``_applyBinning`` 的调用时才实例化此变量。

  .. 注意::
    尽管此处列出的一些变量与定制设置具有相似的名称，但它们*不*代表特征类级别上的所有可能设置。
    这些变量在此处列出，以帮助开发人员开发新的特征类，这些特征类利用了这些变量。有关定制的更多信息，请参见
    :ref:`radiomics-customization-label`，其中包括所有可能设置的全面列表，包括默认值和使用说明。
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    self.logger = logging.getLogger(self.__module__)
    self.logger.debug('Initializing feature class')

    if inputImage is None or inputMask is None:
      raise ValueError('Missing input image or mask')

    self.progressReporter = getProgressReporter

    self.settings = kwargs

    self.label = kwargs.get('label', 1)
    self.voxelBased = kwargs.get('voxelBased', False)

    self.coefficients = {}

    # all features are disabled by default
    self.enabledFeatures = {}
    self.featureValues = {}

    self.featureNames = self.getFeatureNames()

    self.inputImage = inputImage
    self.inputMask = inputMask

    if self.voxelBased:
      self._initVoxelBasedCalculation()
    else:
      self._initSegmentBasedCalculation()

  def _initSegmentBasedCalculation(self):
    self.imageArray = sitk.GetArrayFromImage(self.inputImage)
    self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)  # boolean array

    self.labelledVoxelCoordinates = numpy.where(self.maskArray)
    self.boundingBoxSize = numpy.max(self.labelledVoxelCoordinates, 1) - numpy.min(self.labelledVoxelCoordinates, 1) + 1

  def _initVoxelBasedCalculation(self):
    self.masked = self.settings.get('maskedKernel', True)

    self.imageArray = sitk.GetArrayFromImage(self.inputImage)

    # Set up the mask array for the gray value discretization
    if self.masked:
      self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)  # boolean array
    else:
      self.maskArray = None  # This will cause the discretization to use the entire image

    # Prepare the kernels (1 per voxel in the ROI)
    self.kernels = self._getKernelGenerator()

  def _getKernelGenerator(self):
    kernelRadius = self.settings.get('kernelRadius', 1)

    ROI_mask = sitk.GetArrayFromImage(self.inputMask) == self.label
    ROI_indices = numpy.array(numpy.where(ROI_mask))

    # Get the size of the input, which depends on whether it is in masked mode or not
    if self.masked:
      size = numpy.max(ROI_indices, 1) - numpy.min(ROI_indices, 1) + 1
    else:
      size = numpy.array(self.imageArray.shape)

    # Take the minimum size along each x, y and z dimension from either the size of the ROI or the kernel
    # First add the kernel radius to the size, yielding shape (2, 3), then take the minimum along axis 0, getting back
    # to shape (3,)
    self.boundingBoxSize = numpy.min(numpy.insert([size], 1, kernelRadius * 2 + 1, axis=0), axis=0)

    # Calculate the offsets, which are used to generate a list of kernel Coordinates
    kernelOffsets = cMatrices.generate_angles(self.boundingBoxSize,
                                              numpy.array(six.moves.range(1, kernelRadius + 1)),
                                              True,  # Bi-directional
                                              self.settings.get('force2D', False),
                                              self.settings.get('force2Ddimension', 0))

    # Generator loop that yields a kernel mask: a boolean array that defines the voxels included in the kernel
    kernelMask = numpy.zeros(self.imageArray.shape, dtype='bool')  # Boolean array to hold mask defining current kernel

    for idx in ROI_indices.T:  # Flip axes to get sets of 3 elements (z, y and x) for each voxel
      kernelMask[:] = False  # Reset kernel mask

      # Get coordinates for all potential voxels in this kernel
      kernelCoordinates = kernelOffsets + idx

      # Exclude voxels outside image bounds
      kernelCoordinates = numpy.delete(kernelCoordinates, numpy.where(numpy.any(kernelCoordinates < 0, axis=1)), axis=0)
      kernelCoordinates = numpy.delete(kernelCoordinates,
                                       numpy.where(numpy.any(kernelCoordinates >= self.imageArray.shape, axis=1)), axis=0)

      idx = tuple(idx)

      # Transform indices to boolean mask array
      kernelMask[tuple(kernelCoordinates.T)] = True
      kernelMask[idx] = True  # Also include center voxel

      if self.masked:
        # Exclude voxels outside ROI
        kernelMask = numpy.logical_and(kernelMask, ROI_mask)

        # check if there are enough voxels to calculate texture, skip voxel if this is not the case.
        if numpy.sum(kernelMask) <= 1:
          continue

      # Also yield the index, identifying which voxel this kernel belongs to
      yield idx, kernelMask

  def _initCalculation(self):
    """
    Last steps to prepare the class for extraction. This function calculates the texture matrices and coefficients in
    the respective feature classes
    """
    pass

  def _applyBinning(self):
    self.matrix, _ = imageoperations.binImage(self.imageArray, self.maskArray, **self.settings)
    self.coefficients['grayLevels'] = numpy.unique(self.matrix[self.maskArray])
    self.coefficients['Ng'] = int(numpy.max(self.coefficients['grayLevels']))  # max gray level in the ROI

  def enableFeatureByName(self, featureName, enable=True):
    """
    Enables or disables feature specified by ``featureName``. If feature is not present in this class, a lookup error is
    raised. ``enable`` specifies whether to enable or disable the feature.
    """
    if featureName not in self.featureNames:
      raise LookupError('Feature not found: ' + featureName)
    if self.featureNames[featureName]:
      self.logger.warning('Feature %s is deprecated, use with caution!', featureName)
    self.enabledFeatures[featureName] = enable

  def enableAllFeatures(self):
    """
    Enables all features found in this class for calculation.

    .. note::
      Features that have been marked "deprecated" are not enabled by this function. They can still be enabled manually by
      a call to :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`,
      :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()`
      or in the parameter file (by specifying the feature by name, not when enabling all features).
      However, in most cases this will still result only in a deprecation warning.
    """
    for featureName, is_deprecated in six.iteritems(self.featureNames):
      # only enable non-deprecated features here
      if not is_deprecated:
        self.enableFeatureByName(featureName, True)

  def disableAllFeatures(self):
    """
    Disables all features. Additionally resets any calculated features.
    """
    self.enabledFeatures = {}
    self.featureValues = {}

  @classmethod
  def getFeatureNames(cls):
    """
    Dynamically enumerates features defined in the feature class. Features are identified by the
    ``get<Feature>FeatureValue`` signature, where <Feature> is the name of the feature (unique on the class level).

    Found features are returned as a dictionary of the feature names, where the value ``True`` if the
    feature is deprecated, ``False`` otherwise (``{<Feature1>:<deprecated>, <Feature2>:<deprecated>, ...}``).

    This function is called at initialization, found features are stored in the ``featureNames`` variable.
    """
    attributes = inspect.getmembers(cls)
    features = {a[0][3:-12]: getattr(a[1], '_is_deprecated', False) for a in attributes
                if a[0].startswith('get') and a[0].endswith('FeatureValue')}
    return features

  def execute(self):
    """
    Calculates all features enabled in  ``enabledFeatures``. A feature is enabled if it's key is present in this
    dictionary and it's value is True.

    Calculated values are stored in the ``featureValues`` dictionary, with feature name as key and the calculated
    feature value as value. If an exception is thrown during calculation, the error is logged, and the value is set to
    NaN.
    """
    if self.voxelBased:
      self._calculateVoxels()
    else:
      self._calculateSegment()

    return self.featureValues

  @deprecated
  def calculateFeatures(self):
    self.logger.warning('calculateFeatures() is deprecated, use execute() instead.')
    self.execute()

  def _calculateVoxels(self):
    initValue = self.settings.get('initValue', 0)
    # Initialize the output with empty numpy arrays
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = numpy.full(self.imageArray.shape, initValue, dtype='float')

    # Calculate the feature values for all enabled features
    with self.progressReporter(self.kernels, 'Calculating voxels') as bar:
      for vox_idx, kernelMask in bar:
        self.maskArray = kernelMask
        self.labelledVoxelCoordinates = numpy.where(self.maskArray)

        # Calculate the feature values for the current kernel
        for success, featureName, featureValue in self._calculateFeatures():
          if success:  # Do not store results in case of an error
            self.featureValues[featureName][vox_idx] = featureValue

    # Convert the output to simple ITK image objects
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = sitk.GetImageFromArray(self.featureValues[feature])
        self.featureValues[feature].CopyInformation(self.inputImage)

  def _calculateSegment(self):
    # Get the feature values using the current segment.
    for success, featureName, featureValue in self._calculateFeatures():
      # Always store the result. In case of an error, featureValue will be NaN
      self.featureValues[featureName] = featureValue

  def _calculateFeatures(self):
    # Initialize the calculation
    # This function serves to calculate the texture matrices where applicable
    self._initCalculation()

    self.logger.debug('Calculating features')
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        try:
          # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
          yield True, feature, getattr(self, 'get%sFeatureValue' % feature)()
        except DeprecationWarning as deprecatedFeature:
          # Add a debug log message, as a warning is usually shown and would entail a too verbose output
          self.logger.debug('Feature %s is deprecated: %s', feature, deprecatedFeature.message)
        except Exception:
          self.logger.error('FAILED: %s', traceback.format_exc())
          yield False, feature, numpy.nan
