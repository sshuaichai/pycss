import inspect  # 导入inspect模块，用于获取对象信息
import logging  # 导入logging模块，用于日志记录
import traceback  # 导入traceback模块，用于获取异常信息

import numpy  # 导入numpy库，用于数值计算
import SimpleITK as sitk  # 导入SimpleITK库，用于处理医学影像数据
import six  # 导入six库，用于处理Python 2和Python 3兼容性

from radiomics import getProgressReporter, imageoperations  # 导入radiomics库中的相关模块

class RadiomicsFeaturesBase(object):
    """
    这是一个抽象类，定义了特征类的共同接口。所有特征类都继承（直接或间接）自这个类。

    在初始化时，图像和标签地图作为SimpleITK图像对象传递（分别为“inputImage”和“inputMask”）。
    使用SimpleITK图像作为输入的原因是保留了在未来可以重用SimpleITK中实现的优化特征计算器的可能性。
    如果图像或掩码为空，则初始化失败并记录警告（不会引发错误）。

    使用从父“radiomics”记录器派生的子记录器设置日志记录。这保留了生成的日志中的工具箱结构。
    子记录器的名称取决于包含特征类的模块（例如，'radiomics.glcm'）。

    任何在调用特征函数之前需要的预计算都可以通过覆盖“_initSegmentBasedCalculation”函数来添加，
    该函数准备特征提取的输入。如果需要图像离散化，可以通过在初始化函数中添加对“_applyBinning”的调用来实现，
    该函数还实例化了在离散化后在ROI内可以找到的最大（'Ng'）和唯一（'GrayLevels'）系数。
    此函数还实例化“matrix”变量，该变量保存了离散化的图像（“imageArray”变量仅保存原始灰度级）。

    在初始化时会实例化以下变量：

    - kwargs：包含传递给此特征类的所有自定义设置的字典。
    - label：标签图中感兴趣区域（ROI）的标签值。如果键不存在，则使用默认值1。
    - featureNames：包含特征类中定义的特征名称的列表。参见：py:func:`getFeatureNames`
    - inputImage：输入图像的SimpleITK图像对象（维度x、y、z）

    “_initSegmentBasedCalculation”函数会实例化以下变量：

    - inputMask：输入标签图的SimpleITK图像对象（维度x、y、z）
    - imageArray：输入图像中灰度值的numpy数组（维度z、y、x）
    - maskArray：布尔numpy数组，其中元素设置为“True”，表示标签图=标签，否则为“False”（维度z、y、x）。
    - labelledVoxelCoordinates：包含ROI内的体素的z、x和y坐标的三个numpy数组的元组，长度等于ROI内的总体素数。
    - matrix：图像Array的副本，使用指定的binWidth离散化ROI内的灰度值。仅当特征类的覆盖函数“_initSegmentBasedCalculation”中添加对“_applyBinning”的调用时才实例化此变量。

    .. 注意::
        虽然这里列出的一些变量与自定义设置具有相似的名称，但它们不表示特征类级别上的所有可能设置。
        这些变量在这里列出，以帮助开发人员开发使用这些变量的新特征类。有关自定义的更多信息，请参阅:ref:`radiomics-customization-label`，
        其中包括所有可能设置的综合列表，包括默认值和用法说明。
    """

    def __init__(self, inputImage, inputMask, **kwargs):
        self.logger = logging.getLogger(self.__module__)
        self.logger.debug('Initializing feature class')  # 记录初始化过程

        if inputImage is None or inputMask is None:
            raise ValueError('缺少输入图像或掩码')  # 如果输入图像或掩码为空，则引发错误

        self.progressReporter = getProgressReporter

        self.settings = kwargs  # 自定义设置存储在kwargs字典中

        self.label = kwargs.get('label', 1)  # 获取标签值，如果不存在则使用默认值1
        self.voxelBased = kwargs.get('voxelBased', False)  # 获取voxelBased设置，如果不存在则使用默认值False

        self.coefficients = {}  # 初始化系数字典

        # 默认情况下，所有特征都禁用
        self.enabledFeatures = {}
        self.featureValues = {}

        self.featureNames = self.getFeatureNames()  # 获取特征名称列表

        self.inputImage = inputImage  # 输入图像
        self.inputMask = inputMask  # 输入标签图

        self.imageArray = sitk.GetArrayFromImage(self.inputImage)  # 从SimpleITK图像中获取灰度值数组

        if self.voxelBased:
            self._initVoxelBasedCalculation()  # 初始化voxelBased计算
        else:
            self._initSegmentBasedCalculation()  # 初始化segmentBased计算

    def _initSegmentBasedCalculation(self):
        self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)  # 创建掩码数组

    def _initVoxelBasedCalculation(self):
        self.masked = self.settings.get('maskedKernel', True)  # 获取maskedKernel设置，如果不存在则使用默认值True

        maskArray = sitk.GetArrayFromImage(self.inputMask) == self.label  # 创建掩码数组
        self.labelledVoxelCoordinates = numpy.array(numpy.where(maskArray))  # 获取ROI内的体素坐标

        # 设置用于灰度值离散化的掩码数组
        if self.masked:
            self.maskArray = maskArray
        else:
            # 这将导致离散化使用整个图像
            self.maskArray = numpy.ones(self.imageArray.shape, dtype='bool')

    def _initCalculation(self, voxelCoordinates=None):
        """
        准备执行特征提取的最后步骤，此函数计算特征类中的纹理矩阵和系数。
        """
        pass

    def _applyBinning(self, matrix):
        matrix, _ = imageoperations.binImage(matrix, self.maskArray, **self.settings)  # 应用灰度离散化
        self.coefficients['grayLevels'] = numpy.unique(matrix[self.maskArray])
        self.coefficients['Ng'] = int(numpy.max(self.coefficients['grayLevels']))  # ROI中的最大灰度级
        return matrix

    def enableFeatureByName(self, featureName, enable=True):
        """
        启用或禁用指定名称的特征。如果特征不在此类中存在，则引发查找错误。
        参数“enable”指定是否启用或禁用该特征。
        """
        if featureName not in self.featureNames:
            raise LookupError('未找到特征：' + featureName)
        if self.featureNames[featureName]:
            self.logger.warning('特征 %s 已弃用，请谨慎使用！', featureName)
        self.enabledFeatures[featureName] = enable

    def enableAllFeatures(self):
        """
        启用此类中找到的所有特征进行计算。

        .. 注意::
            此函数不会启用已标记为“已弃用”的特征。它们仍然可以通过手动调用：py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`，
            :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()` 或参数文件（通过名称指定特征，而不是在启用所有特征时指定）来手动启用。
            但是，在大多数情况下，这仍然只会导致出现已弃用的警告。
        """
        for featureName, is_deprecated in six.iteritems(self.featureNames):
            # 仅在此处启用非已弃用的特征
            if not is_deprecated:
                self.enableFeatureByName(featureName, True)

    def disableAllFeatures(self):
        """
        禁用所有特征，同时重置任何已计算的特征。
        """
        self.enabledFeatures = {}
        self.featureValues = {}

    @classmethod
    def getFeatureNames(cls):
        """
        动态列举特征类中定义的特征。特征通过其名称为“get<Feature>FeatureValue”的签名进行标识，其中<Feature>是特征的名称（在类级别上是唯一的）。
        找到的特征作为特征名称的字典返回，其中如果特征已弃用则值为“True”，否则为“False”（{<Feature1>:<已弃用>, <Feature2>:<已弃用>, ...}）。

        此函数在初始化时调用，找到的特征存储在“featureNames”变量中。
        """
        attributes = inspect.getmembers(cls)
        features = {a[0][3:-12]: getattr(a[1], '_is_deprecated', False) for a in attributes
                    if a[0].startswith('get') and a[0].endswith('FeatureValue')}
        return features

    def execute(self):
        """
        计算“enabledFeatures”中启用的所有特征。如果特征的键存在于此字典中且其值为True，则特征已启用。

        计算后的值存储在“featureValues”字典中，特征名称作为键，计算后的特征值作为值。
        如果在计算过程中引发异常，则记录错误并将值设置为NaN。
        """
        if len(self.enabledFeatures) == 0:
            self.enableAllFeatures()

        if self.voxelBased:
            self._calculateVoxels()
        else:
            self._calculateSegment()

        return self.featureValues

    def _calculateVoxels(self):
        initValue = self.settings.get('initValue', 0)  # 获取initValue设置，如果不存在则使用默认值0
        voxelBatch = self.settings.get('voxelBatch', -1)  # 获取voxelBatch设置，如果不存在则使用默认值-1

        # 使用空的numpy数组初始化输出
        for feature, enabled in six.iteritems(self.enabledFeatures):
            if enabled:
                self.featureValues[feature] = numpy.full(list(self.inputImage.GetSize())[::-1], initValue, dtype='float')

        # 计算所有已启用特征的特征值
        voxel_count = self.labelledVoxelCoordinates.shape[1]
        voxel_batch_idx = 0
        if voxelBatch < 0:
            voxelBatch = voxel_count
        n_batches = numpy.ceil(float(voxel_count) / voxelBatch)
        with self.progressReporter(total=n_batches, desc='batch') as pbar:
            while voxel_batch_idx < voxel_count:
                self.logger.debug('正在计算体素批次 %i/%i', int(voxel_batch_idx / voxelBatch) + 1, n_batches)
                voxelCoords = self.labelledVoxelCoordinates[:, voxel_batch_idx:voxel_batch_idx + voxelBatch]
                # 计算当前核的特征值
                for success, featureName, featureValue in self._calculateFeatures(voxelCoords):
                    if success:
                        self.featureValues[featureName][tuple(voxelCoords)] = featureValue

                voxel_batch_idx += voxelBatch
                pbar.update(1)  # 更新进度条

        # 将输出转换为SimpleITK图像对象
        for feature, enabled in six.iteritems(self.enabledFeatures):
            if enabled:
                self.featureValues[feature] = sitk.GetImageFromArray(self.featureValues[feature])
                self.featureValues[feature].CopyInformation(self.inputImage)

    def _calculateSegment(self):
        # 使用当前段获取特征值。
        for success, featureName, featureValue in self._calculateFeatures():
            # 始终存储结果。在出现错误的情况下，featureValue将为NaN
            self.featureValues[featureName] = numpy.squeeze(featureValue)

    def _calculateFeatures(self, voxelCoordinates=None):
        # 初始化计算
        # 此函数用于计算纹理矩阵（如适用）
        self._initCalculation(voxelCoordinates)

        self.logger.debug('正在计算特征')
        for feature, enabled in six.iteritems(self.enabledFeatures):
            if enabled:
                try:
                    # 使用getattr获取特征计算方法，然后使用'()'来评估这些方法
                    yield True, feature, getattr(self, 'get%sFeatureValue' % feature)()
                except DeprecationWarning as deprecatedFeature:
                    # 添加调试日志消息，因为通常会显示警告并导致输出过于冗长
                    self.logger.debug('特征 %s 已弃用：%s', feature, deprecatedFeature.args[0])
                except Exception:
                    self.logger.error('失败：%s', traceback.format_exc())
                    yield False, feature, numpy.nan
