import numpy  # 导入numpy库
from six.moves import range  # 从six.moves导入range，提供Python2和Python3间的兼容性

from radiomics import base, cMatrices, deprecated  # 从radiomics包导入base, cMatrices, deprecated模块


class RadiomicsFirstOrder(base.RadiomicsFeaturesBase):
  r"""
  一阶统计描述了由掩码定义的图像区域内体素强度的分布，通过常用和基本的度量。

  设：

  - :math:`\textbf{X}` 是ROI中包含的 :math:`N_p` 个体素的集合
  - :math:`\textbf{P}(i)` 是具有 :math:`N_g` 个离散强度级别的一阶直方图，
    其中 :math:`N_g` 是非零箱数的数量，从0开始等间距排列，宽度由``binWidth``参数定义。
  - :math:`p(i)` 是归一化的一阶直方图，等于 :math:`\frac{\textbf{P}(i)}{N_p}`

  可以进行以下额外设置：

  - voxelArrayShift [0]: 整数，此值加到特征能量、总能量和RMS的灰度级强度中，这是为了防止负值。*如果使用CT数据，或以均值0归一化的数据，考虑将此参数设置为确保图像中非负数的固定值（例如，2000）。但请记住，值越大，体积混淆效应将越大。*

  .. note::
    在IBSI特征定义中，没有实现对负灰度值的校正。要在PyRadiomics中实现类似行为，将``voxelArrayShift``设置为0。
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsFirstOrder, self).__init__(inputImage, inputMask, **kwargs)  # 调用基类的构造函数

    self.pixelSpacing = inputImage.GetSpacing()  # 获取图像的像素间距
    self.voxelArrayShift = kwargs.get('voxelArrayShift', 0)  # 获取体素数组偏移量
    self.discretizedImageArray = self._applyBinning(self.imageArray.copy())  # 应用分箱处理并复制图像数组

  def _initVoxelBasedCalculation(self):
    super(RadiomicsFirstOrder, self)._initVoxelBasedCalculation()  # 调用基类的_voxelBasedCalculation初始化方法

    kernelRadius = self.settings.get('kernelRadius', 1)  # 获取核半径

    # 获取输入的大小，这取决于是否处于掩码模式
    if self.masked:
      size = numpy.max(self.labelledVoxelCoordinates, 1) - numpy.min(self.labelledVoxelCoordinates, 1) + 1
    else:
      size = numpy.array(self.imageArray.shape)

    # 从ROI的大小或核的大小中取每个维度的最小尺寸
    boundingBoxSize = numpy.minimum(size, kernelRadius * 2 + 1)

    # 计算偏移量，可用于生成核坐标列表。形状 (Nd, Nk)
    self.kernelOffsets = cMatrices.generate_angles(boundingBoxSize,
                                                   numpy.array(range(1, kernelRadius + 1)),
                                                   True,  # 双向
                                                   self.settings.get('force2D', False),
                                                   self.settings.get('force2Ddimension', 0))
    self.kernelOffsets = numpy.append(self.kernelOffsets, [[0, 0, 0]], axis=0)  # 添加中心体素
    self.kernelOffsets = self.kernelOffsets.transpose((1, 0))

    self.imageArray = self.imageArray.astype('float')
    self.imageArray[~self.maskArray] = numpy.nan
    self.imageArray = numpy.pad(self.imageArray,
                                pad_width=self.settings.get('kernelRadius', 1),
                                mode='constant', constant_values=numpy.nan)
    self.maskArray = numpy.pad(self.maskArray,
                               pad_width=self.settings.get('kernelRadius', 1),
                               mode='constant', constant_values=False)

  def _initCalculation(self, voxelCoordinates=None):

    if voxelCoordinates is None:
      self.targetVoxelArray = self.imageArray[self.maskArray].astype('float').reshape((1, -1))
      _, p_i = numpy.unique(self.discretizedImageArray[self.maskArray], return_counts=True)
      p_i = p_i.reshape((1, -1))
    else:
      # voxelCoordinates 形状 (Nd, Nvox)
      voxelCoordinates = voxelCoordinates.copy() + self.settings.get('kernelRadius', 1)  # 调整填充
      kernelCoords = self.kernelOffsets[:, None, :] + voxelCoordinates[:, :, None]  # 形状 (Nd, Nvox, Nk)
      kernelCoords = tuple(kernelCoords)  # 形状 (Nd, (Nvox, Nk))

      self.targetVoxelArray = self.imageArray[kernelCoords]  # 形状 (Nvox, Nk)

      p_i = numpy.empty((voxelCoordinates.shape[1], len(self.coefficients['grayLevels'])))  # 形状 (Nvox, Ng)
      for gl_idx, gl in enumerate(self.coefficients['grayLevels']):
        p_i[:, gl_idx] = numpy.nansum(self.discretizedImageArray[kernelCoords] == gl, 1)

    sumBins = numpy.sum(p_i, 1, keepdims=True).astype('float')
    sumBins[sumBins == 0] = 1  # 防止除以0错误
    p_i = p_i.astype('float') / sumBins
    self.coefficients['p_i'] = p_i

    self.logger.debug('一阶特征类初始化完成')

  @staticmethod
  def _moment(a, moment=1):
    r"""
    计算给定轴上数组的n阶矩
    """

    if moment == 1:
      return numpy.float(0.0)
    else:
      mn = numpy.nanmean(a, 1, keepdims=True)
      s = numpy.power((a - mn), moment)
      return numpy.nanmean(s, 1)



  def getEnergyFeatureValue(self):
    r"""
    **1. 能量**

    .. math::
      \textit{energy} = \displaystyle\sum^{N_p}_{i=1}{(\textbf{X}(i) + c)^2}

    这里，:math:`c` 是一个可选值，由 ``voxelArrayShift`` 定义，它将强度加到 :math:`\textbf{X}` 中以防止负值。这确保了灰度值最低的体素对能量的贡献最小，而不是接近0的灰度级强度的体素。

    能量是图像中体素值大小的度量。较大的值意味着这些值的平方和较大。

    .. note::
      此特征与体积混淆相关，:math:`c` 的较大值会增加体积混淆的效应。
    """

    shiftedParameterArray = self.targetVoxelArray + self.voxelArrayShift

    return numpy.nansum(shiftedParameterArray ** 2, 1)

  def getTotalEnergyFeatureValue(self):
    r"""
    **2. 总能量**

    .. math::
      \textit{total energy} = V_{voxel}\displaystyle\sum^{N_p}_{i=1}{(\textbf{X}(i) + c)^2}

    这里，:math:`c` 是一个可选值，由 ``voxelArrayShift`` 定义，它将强度加到 :math:`\textbf{X}` 中以防止负值。这确保了灰度值最低的体素对能量的贡献最小，而不是接近0的灰度级强度的体素。

    总能量是能量特征的值，按体素的体积（立方毫米）缩放。

    .. note::
      此特征与体积混淆相关，:math:`c` 的较大值会增加体积混淆的效应。

    .. note::
      在IBSI特征定义中不出现
    """

    cubicMMPerVoxel = numpy.multiply.reduce(self.pixelSpacing)

    return self.getEnergyFeatureValue() * cubicMMPerVoxel

  def getEntropyFeatureValue(self):
    r"""
    **3. 熵**

    .. math::
      \textit{entropy} = -\displaystyle\sum^{N_g}_{i=1}{p(i)\log_2\big(p(i)+\epsilon\big)}

    这里，:math:`\epsilon` 是一个任意小的正数 (:math:`\approx 2.2\times10^{-16}`)。

    熵指定了图像值的不确定性/随机性。它衡量编码图像值所需的平均信息量。

    .. note::
      由IBSI定义为强度直方图熵。
    """
    p_i = self.coefficients['p_i']

    eps = numpy.spacing(1)
    return -1.0 * numpy.sum(p_i * numpy.log2(p_i + eps), 1)

  def getMinimumFeatureValue(self):
    r"""
    **4. 最小值**

    .. math::
      \textit{minimum} = \min(\textbf{X})
    """

    return numpy.nanmin(self.targetVoxelArray, 1)

  def get10PercentileFeatureValue(self):
    r"""
    **5. 第10百分位数**

    :math:`\textbf{X}` 的第10百分位数
    """
    return numpy.nanpercentile(self.targetVoxelArray, 10, axis=1)

  def get90PercentileFeatureValue(self):
    r"""
    **6. 第90百分位数**

    :math:`\textbf{X}` 的第90百分位数
    """

    return numpy.nanpercentile(self.targetVoxelArray, 90, axis=1)

  def getMaximumFeatureValue(self):
    r"""
    **7. 最大值**

    .. math::
      \textit{maximum} = \max(\textbf{X})

    ROI内的最大灰度级强度。
    """

    return numpy.nanmax(self.targetVoxelArray, 1)

  def getMeanFeatureValue(self):
    r"""
    **8. 平均值**

    .. math::
      \textit{mean} = \frac{1}{N_p}\displaystyle\sum^{N_p}_{i=1}{\textbf{X}(i)}

    ROI内的平均灰度级强度。
    """

    return numpy.nanmean(self.targetVoxelArray, 1)

  def getMedianFeatureValue(self):
    r"""
    **9. 中位数**

    ROI内的中位灰度级强度。
    """

    return numpy.nanmedian(self.targetVoxelArray, 1)

  def getInterquartileRangeFeatureValue(self):
    r"""
    **10. 四分位数范围**

    .. math::
      \textit{interquartile range} = \textbf{P}_{75} - \textbf{P}_{25}

    这里 :math:`\textbf{P}_{25}` 和 :math:`\textbf{P}_{75}` 分别是图像数组的第25和第75百分位数。
    """

    return numpy.nanpercentile(self.targetVoxelArray, 75, axis=1) - numpy.nanpercentile(self.targetVoxelArray, 25, axis=1)

  def getRangeFeatureValue(self):
    r"""
    **11. 范围**

    .. math::
      \textit{range} = \max(\textbf{X}) - \min(\textbf{X})

    ROI内的灰度值范围。
    """

    return numpy.nanmax(self.targetVoxelArray, axis=1) - numpy.nanmin(self.targetVoxelArray, axis=1)

  def getMeanAbsoluteDeviationFeatureValue(self):
    r"""
    **12. 平均绝对偏差 (MAD)**

    .. math::
      \textit{MAD} = \frac{1}{N_p}\displaystyle\sum^{N_p}_{i=1}{|\textbf{X}(i)-\bar{X}|}

    平均绝对偏差是图像数组中所有强度值与图像数组的平均值的平均距离。
    """

    u_x = numpy.nanmean(self.targetVoxelArray, axis=1, keepdims=True)
    return numpy.nanmean(numpy.absolute(self.targetVoxelArray - u_x), axis=1)

  def getRobustMeanAbsoluteDeviationFeatureValue(self):
    r"""
    **13. 稳健平均绝对偏差 (rMAD)**

    .. math::
      \textit{rMAD} = \frac{1}{N_{10-90}}\displaystyle\sum^{N_{10-90}}_{i=1}
      {|\textbf{X}_{10-90}(i)-\bar{X}_{10-90}|}

    稳健平均绝对偏差是所有强度值与在第10至第90百分位数之间或等于第10至第90百分位数的图像数组子集上计算的平均值的平均距离。
    """

    prcnt10 = self.get10PercentileFeatureValue()
    prcnt90 = self.get90PercentileFeatureValue()
    percentileArray = self.targetVoxelArray.copy()

    # 首先获取所有有效体素的掩码
    msk = ~numpy.isnan(percentileArray)
    # 然后，更新掩码以反映所有有效体素，这些体素位于封闭的第10-90百分位范围之外
    msk[msk] = ((percentileArray - prcnt10[:, None])[msk] < 0) | ((percentileArray - prcnt90[:, None])[msk] > 0)
    # 最后，通过设置它们为numpy.nan来排除无效体素。
    percentileArray[msk] = numpy.nan

    return numpy.nanmean(numpy.absolute(percentileArray - numpy.nanmean(percentileArray, axis=1, keepdims=True)), axis=1)

  def getRootMeanSquaredFeatureValue(self):
    r"""
    **14. 均方根 (RMS)**

    .. math::
      \textit{RMS} = \sqrt{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i) + c)^2}}

    这里，:math:`c` 是一个可选值，由 ``voxelArrayShift`` 定义，它将强度加到 :math:`\textbf{X}` 中以防止负值。这确保了灰度值最低的体素对RMS的贡献最小，而不是接近0的灰度级强度的体素。

    RMS是所有平方强度值的平均值的平方根。它是图像值大小的另一种度量。此特征与体积混淆相关，:math:`c` 的较大值会增加体积混淆的效应。
    """

    # 如果没有分割体素，防止除以0并返回0
    if self.targetVoxelArray.size == 0:
      return 0

    shiftedParameterArray = self.targetVoxelArray + self.voxelArrayShift
    Nvox = numpy.sum(~numpy.isnan(self.targetVoxelArray), axis=1).astype('float')
    return numpy.sqrt(numpy.nansum(shiftedParameterArray ** 2, axis=1) / Nvox)

  @deprecated
  def getStandardDeviationFeatureValue(self):
    r"""
    **15. 标准偏差**

    .. math::
      \textit{standard deviation} = \sqrt{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^2}}

    标准偏差衡量每个强度值与平均值的偏差或分散量。按定义，:math:`\textit{standard deviation} = \sqrt{\textit{variance}}`

    .. note::
      由于此特征与方差相关，因此默认不启用。要在提取中包含此特征，请在启用的特征中按名称指定它（即，如果未指定单个特征（启用“所有”特征），则不会启用此特征，但是当指定单个特征时，包括此特征，将被启用）。在IBSI特征定义中不出现（与方差相关）
    """

    return numpy.nanstd(self.targetVoxelArray, axis=1)

  def getSkewnessFeatureValue(self):
    r"""
    **16. 偏度**

    .. math::
      \textit{skewness} = \displaystyle\frac{\mu_3}{\sigma^3} =
      \frac{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^3}}
      {\left(\sqrt{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^2}}\right)^3}

    其中 :math:`\mu_3` 是第3中心矩。

    偏度衡量值分布关于平均值的不对称性。根据尾部延伸和分布质量集中的位置，这个值可以是正的或负的。

    相关链接:

    https://en.wikipedia.org/wiki/Skewness

    .. note::
      在平坦区域的情况下，标准偏差和第4中心矩都是0。在这种情况下，返回值0。
    """

    m2 = self._moment(self.targetVoxelArray, 2)
    m3 = self._moment(self.targetVoxelArray, 3)

    m2[m2 == 0] = 1  # 平坦区域，防止除以0错误
    m3[m2 == 0] = 0  # 确保平坦区域返回为0

    return m3 / m2 ** 1.5

  def getKurtosisFeatureValue(self):
    r"""
    **17. 峰度**

    .. math::
      \textit{kurtosis} = \displaystyle\frac{\mu_4}{\sigma^4} =
      \frac{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^4}}
      {\left(\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X}})^2\right)^2}

    其中 :math:`\mu_4` 是第4中心矩。

    峰度是图像ROI中值分布“尖峰性”的度量。较高的峰度意味着分布的质量集中于尾部而不是平均值附近。较低的峰度意味着相反：分布的质量集中于接近平均值的尖峰。

    相关链接:

    https://en.wikipedia.org/wiki/Kurtosis

    .. note::
      在平坦区域的情况下，标准偏差和第4中心矩都是0。在这种情况下，返回值0。

    .. note::
      IBSI特征定义实现了超额峰度，其中峰度通过-3进行了校正，对于正态分布产生0。PyRadiomics的峰度没有校正，产生的值比IBSI峰度高3。
    """

    m2 = self._moment(self.targetVoxelArray, 2)
    m4 = self._moment(self.targetVoxelArray, 4)

    m2[m2 == 0] = 1  # 平坦区域，防止除以0错误
    m4[m2 == 0] = 0  # 确保平坦区域返回为0

    return m4 / m2 ** 2.0

  def getVarianceFeatureValue(self):
    r"""
    **18. 方差**

    .. math::
      \textit{variance} = \frac{1}{N_p}\displaystyle\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^2}

    方差是每个强度值与平均值的平方距离的平均值。这是衡量分布围绕平均值的分散程度的度量。按定义，:math:`\textit{variance} = \sigma^2`
    """

    return numpy.nanstd(self.targetVoxelArray, axis=1) ** 2

  def getUniformityFeatureValue(self):
    r"""
    **19. 均匀性**

    .. math::
      \textit{uniformity} = \displaystyle\sum^{N_g}_{i=1}{p(i)^2}

    均匀性是每个强度值的平方和的度量。这是图像数组的均匀性的度量，其中较大的均匀性意味着较大的均匀性或较小的离散强度值范围。

    .. note::
      由IBSI定义为强度直方图均匀性。
    """
    p_i = self.coefficients['p_i']
    return numpy.nansum(p_i ** 2, axis=1)




