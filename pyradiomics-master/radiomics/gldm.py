import numpy

from radiomics import base, cMatrices, deprecated

class RadiomicsGLDM(base.RadiomicsFeaturesBase):
  r"""
  灰度级依赖矩阵（GLDM）量化图像中的灰度级依赖性。
  灰度级依赖性定义为与中心体素距离 :math:`\delta` 内的连接体素的数量，这些体素依赖于中心体素。
  如果 :math:`|i-j|\le\alpha`，则具有灰度级 :math:`j` 的邻近体素被认为依赖于具有灰度级 :math:`i` 的中心体素。
  在灰度级依赖矩阵 :math:`\textbf{P}(i,j)` 中，:math:`(i,j)`\ :sup:`th` 元素描述了图像中以 :math:`i` 灰度级的体素出现的次数，其邻域内有 :math:`j` 个依赖体素。

  作为二维示例，考虑以下具有5个离散灰度级的5x5图像：

  .. math::
    \textbf{I} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}

  对于 :math:`\alpha=0` 和 :math:`\delta = 1`，则GLDM变为：

  .. math::
    \textbf{P} = \begin{bmatrix}
    0 & 1 & 2 & 1 \\
    1 & 2 & 3 & 0 \\
    1 & 4 & 4 & 0 \\
    1 & 2 & 0 & 0 \\
    3 & 0 & 0 & 0 \end{bmatrix}

  让：

  - :math:`N_g` 是图像中离散强度值的数量
  - :math:`N_d` 是图像中离散依赖大小的数量
  - :math:`N_z` 是图像中依赖区域的数量，等于 :math:`\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)}`
  - :math:`\textbf{P}(i,j)` 是依赖矩阵
  - :math:`p(i,j)` 是归一化的依赖矩阵，定义为 :math:`p(i,j) = \frac{\textbf{P}(i,j)}{N_z}`

  .. note::
    因为允许不完整的区域，所以ROI中的每个体素都有一个依赖区域。因此，:math:`N_z = N_p`，其中 :math:`N_p` 是图像中体素的数量。
    由于 :math:`Nz = N_p`，依赖百分比和灰度级非均匀性归一化（GLNN）已被移除。第一个因为它总是计算为1，后者因为它在数学上等于一阶-均匀性（见 :py:func:`~radiomics.firstorder.RadiomicsFirstOrder.getUniformityFeatureValue()`）。数学证明见 :ref:`这里 <radiomics-excluded-gldm-label>`。

  以下是类特定的设置：

  - distances [[1]]: 整数列表。这指定了中心体素和邻居之间的距离，应该为其生成角度。
  - gldm_a [0]: float, 依赖的 :math:`\alpha` 截止值。如果 :math:`|i-j|\le\alpha`，则认为具有灰度级 :math:`j` 的邻近体素依赖于具有灰度级 :math:`i` 的中心体素

  参考文献：

  - Sun C, Wee WG. 邻近灰度级依赖矩阵用于纹理分类。计算视觉，图形图像处理。1983;23:341-352
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsGLDM, self).__init__(inputImage, inputMask, **kwargs)

    self.gldm_a = kwargs.get('gldm_a', 0)  # 获取GLDM的alpha值，默认为0

    self.P_gldm = None  # 初始化GLDM矩阵为None
    self.imageArray = self._applyBinning(self.imageArray)  # 对图像进行分箱处理

  def _initCalculation(self, voxelCoordinates=None):
    self.P_gldm = self._calculateMatrix(voxelCoordinates)  # 计算GLDM矩阵

    self.logger.debug('特征类初始化完成，计算得到的GLDM矩阵形状为 %s', self.P_gldm.shape)

  def _calculateMatrix(self, voxelCoordinates=None):
    self.logger.debug('在C中计算GLDM矩阵')

    Ng = self.coefficients['Ng']  # 灰度级数量

    matrix_args = [
      self.imageArray,
      self.maskArray,
      numpy.array(self.settings.get('distances', [1])),  # 距离设置
      Ng,
      self.gldm_a,  # alpha值
      self.settings.get('force2D', False),  # 是否强制2D处理
      self.settings.get('force2Ddimension', 0)  # 2D处理的维度
    ]
    if self.voxelBased:
      matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]  # 体素基础设置

    P_gldm = cMatrices.calculate_gldm(*matrix_args)  # 计算GLDM矩阵，形状 (Nv, Ng, Nd)

    # 删除未出现在ROI中的灰度级对应的行
    NgVector = range(1, Ng + 1)  # 所有可能的灰度值
    GrayLevels = self.coefficients['grayLevels']  # ROI中存在的灰度值
    emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)), dtype=int)  # ROI中不存在的灰度值

    P_gldm = numpy.delete(P_gldm, emptyGrayLevels - 1, 1)

    jvector = numpy.arange(1, P_gldm.shape[2] + 1, dtype='float64')

    # 形状 (Nv, Nd)
    pd = numpy.sum(P_gldm, 1)
    # 形状 (Nv, Ng)
    pg = numpy.sum(P_gldm, 2)

    # 删除不存在于ROI中的依赖大小对应的列
    empty_sizes = numpy.sum(pd, 0)
    P_gldm = numpy.delete(P_gldm, numpy.where(empty_sizes == 0), 2)
    jvector = numpy.delete(jvector, numpy.where(empty_sizes == 0))
    pd = numpy.delete(pd, numpy.where(empty_sizes == 0), 1)

    Nz = numpy.sum(pd, 1)  # 每个核的Nz，形状 (Nv, )
    Nz[Nz == 0] = 1  # 如果和为0，则设置为1？

    self.coefficients['Nz'] = Nz

    self.coefficients['pd'] = pd
    self.coefficients['pg'] = pg

    self.coefficients['ivector'] = self.coefficients['grayLevels'].astype(float)
    self.coefficients['jvector'] = jvector

    return P_gldm


  def getSmallDependenceEmphasisFeatureValue(self):
    r"""
    **1. 小依赖强调 (SDE)**

    .. math::
      SDE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}

    衡量小依赖分布的指标，较大的值表明更小的依赖和较不均匀的纹理。
    """
    pd = self.coefficients['pd']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']  # Nz = Np, 见类文档字符串

    sde = numpy.sum(pd / (jvector[None, :] ** 2), 1) / Nz
    return sde

  def getLargeDependenceEmphasisFeatureValue(self):
    r"""
    **2. 大依赖强调 (LDE)**

    .. math::
      LDE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)j^2}}{N_z}

    衡量大依赖分布的指标，较大的值表明更大的依赖和更均匀的纹理。
    """
    pd = self.coefficients['pd']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    lre = numpy.sum(pd * (jvector[None, :] ** 2), 1) / Nz
    return lre

  def getGrayLevelNonUniformityFeatureValue(self):
    r"""
    **3. 灰度级非均匀性 (GLN)**

    .. math::
      GLN = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_d}_{j=1}{\textbf{P}(i,j)}\right)^2}{N_z}

    衡量图像中灰度级强度值的相似性，较低的GLN值与强度值的更大相似性相关。
    """
    pg = self.coefficients['pg']
    Nz = self.coefficients['Nz']

    gln = numpy.sum(pg ** 2, 1) / Nz
    return gln

  def getDependenceNonUniformityFeatureValue(self):
    r"""
    **4. 依赖性非均匀性 (DN)**

    .. math::
      DN = \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z}

    衡量图像中依赖性的相似性，较低的值指示图像中依赖性的更大均匀性。
    """
    pd = self.coefficients['pd']
    Nz = self.coefficients['Nz']

    dn = numpy.sum(pd ** 2, 1) / Nz
    return dn

  def getDependenceNonUniformityNormalizedFeatureValue(self):
    r"""
    **5. 依赖性非均匀性归一化 (DNN)**

    .. math::
      DNN = \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z^2}

    衡量图像中依赖性的相似性，较低的值指示图像中依赖性的更大均匀性。这是DN公式的归一化版本。
    """
    pd = self.coefficients['pd']
    Nz = self.coefficients['Nz']

    dnn = numpy.sum(pd ** 2, 1) / Nz ** 2
    return dnn

  def getGrayLevelVarianceFeatureValue(self):
    r"""
    **6. 灰度级方差 (GLV)**

    .. math::
      GLV = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{p(i,j)(i - \mu)^2} \text{，其中}
      \mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{ip(i,j)}

    衡量图像中灰度级的方差。
    """
    ivector = self.coefficients['ivector']
    Nz = self.coefficients['Nz']
    pg = self.coefficients['pg'] / Nz[:, None]  # 除以Nz得到归一化矩阵

    u_i = numpy.sum(pg * ivector[None, :], 1, keepdims=True)
    glv = numpy.sum(pg * (ivector[None, :] - u_i) ** 2, 1)
    return glv

  def getDependenceVarianceFeatureValue(self):
    r"""
    **7. 依赖性方差 (DV)**

    .. math::
      DV = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{p(i,j)(j - \mu)^2} \text{，其中}
      \mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{jp(i,j)}

    衡量图像中依赖大小的方差。
    """
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']
    pd = self.coefficients['pd'] / Nz[:, None]  # 除以Nz得到归一化矩阵

    u_j = numpy.sum(pd * jvector[None, :], 1, keepdims=True)
    dv = numpy.sum(pd * (jvector[None, :] - u_j) ** 2, 1)
    return dv

  def getDependenceEntropyFeatureValue(self):
    r"""
    **8. 依赖性熵 (DE)**

    .. math::
      依赖性熵 = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{p(i,j)\log_{2}(p(i,j)+\epsilon)}
    """
    eps = numpy.spacing(1)
    Nz = self.coefficients['Nz']
    p_gldm = self.P_gldm / Nz[:, None, None]  # 除以Nz得到归一化矩阵

    return -numpy.sum(p_gldm * numpy.log2(p_gldm + eps), (1, 2))

  def getLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **9. 低灰度级强调 (LGLE)**

    .. math::
      LGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}

    衡量低灰度级值分布的指标，较高的值表明图像中低灰度级值的更大集中。
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nz = self.coefficients['Nz']

    lgle = numpy.sum(pg / (ivector[None, :] ** 2), 1) / Nz
    return lgle

  def getHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **10. 高灰度级强调 (HGLE)**

    .. math::
      HGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2}}{N_z}

    衡量高灰度级值分布的指标，较高的值表明图像中高灰度级值的更大集中。
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nz = self.coefficients['Nz']

    hgle = numpy.sum(pg * (ivector[None, :] ** 2), 1) / Nz
    return hgle

  def getSmallDependenceLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **11. 小依赖性低灰度级强调 (SDLGLE)**

    .. math::
      SDLGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2j^2}}}{N_z}

    衡量小依赖性与低灰度级值联合分布的指标。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    sdlgle = numpy.sum(self.P_gldm / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)), (1, 2)) / Nz
    return sdlgle

  def getSmallDependenceHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **12. 小依赖性高灰度级强调 (SDHGLE)**

    .. math::
      SDHGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)i^2}{j^2}}}{N_z}

    衡量小依赖性与高灰度级值联合分布的指标。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    sdhgle = numpy.sum(self.P_gldm * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2), (1, 2)) / Nz
    return sdhgle

  def getLargeDependenceLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **13. 大依赖性低灰度级强调 (LDLGLE)**

    .. math::
      LDLGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)j^2}{i^2}}}{N_z}

    衡量大依赖性与低灰度级值联合分布的指标。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    ldlgle = numpy.sum(self.P_gldm * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2), (1, 2)) / Nz
    return ldlgle

  def getLargeDependenceHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **14. 大依赖性高灰度级强调 (LDHGLE)**

    .. math::
      LDHGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2j^2}}{N_z}

    衡量大依赖性与高灰度级值联合分布的指标。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    ldhgle = numpy.sum(self.P_gldm * ((jvector[None, None, :] ** 2) * (ivector[None, :, None] ** 2)), (1, 2)) / Nz
    return ldhgle

