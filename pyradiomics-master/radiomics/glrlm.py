import numpy

from radiomics import base, cMatrices

class RadiomicsGLRLM(base.RadiomicsFeaturesBase):
  r"""
  灰度级长度矩阵 (GLRLM) 量化了灰度级的运行长度，这些运行长度定义为具有相同灰度级值的连续像素的长度（以像素数计）。
  在灰度级长度矩阵 :math:`\textbf{P}(i,j|\theta)` 中，:math:`(i,j)^{\text{th}}` 元素描述了图像 (ROI) 沿角度 :math:`\theta`
  出现的灰度级 :math:`i` 和长度 :math:`j` 的运行次数。

  作为一个二维示例，考虑以下具有5个离散灰度级的5x5图像：

  .. math::
    \textbf{I} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}

  对于 :math:`\theta = 0`，其中0度是水平方向，GLRLM变为：

  .. math::
    \textbf{P} = \begin{bmatrix}
    1 & 0 & 1 & 0 & 0\\
    3 & 0 & 1 & 0 & 0\\
    4 & 1 & 1 & 0 & 0\\
    1 & 1 & 0 & 0 & 0\\
    3 & 0 & 0 & 0 & 0 \end{bmatrix}

  让：

  - :math:`N_g` 是图像中离散强度值的数量
  - :math:`N_r` 是图像中离散运行长度的数量
  - :math:`N_p` 是图像中体素的数量
  - :math:`N_r(\theta)` 是图像沿角度 :math:`\theta` 的运行次数，等于
    :math:`\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}` 且 :math:`1 \leq N_r(\theta) \leq N_p`
  - :math:`\textbf{P}(i,j|\theta)` 是任意方向 :math:`\theta` 的运行长度矩阵
  - :math:`p(i,j|\theta)` 是标准化的运行长度矩阵，定义为 :math:`p(i,j|\theta) =
    \frac{\textbf{P}(i,j|\theta)}{N_r(\theta)}`

  默认情况下，特征值是在每个角度的GLRLM上单独计算的，之后返回这些值的平均值。如果启用了距离加权，GLRLMs通过邻近体素之间的距离加权后求和并归一化。然后在结果矩阵上计算特征。使用'weightingNorm'指定的范数计算每个角度的邻近体素之间的距离。

  以下类特定设置是可能的：

  - weightingNorm [None]: 字符串，指示应用距离加权时应使用哪种范数。
    枚举设置，可能的值：

    - 'manhattan': 一阶范数
    - 'euclidean': 二阶范数
    - 'infinity': 无穷范数。
    - 'no_weighting': GLCMs通过因子1加权并求和
    - None: 不应用加权，返回单独矩阵上计算的值的平均值。

    如果是其他值，将记录一个警告并使用选项'no_weighting'。

  参考文献

  - Galloway MM. 1975. 使用灰度级运行长度的纹理分析。计算机图形学和图像处理，
    4(2):172-179。
  - Chu A., Sehgal C.M., Greenleaf J. F. 1990. 使用运行长度的灰度值分布进行纹理分析。
    模式识别信件，11(6):415-419
  - Xu D., Kurani A., Furst J., Raicu D. 2004. 体积纹理的运行长度编码。国际可视化、成像和图像处理会议 (VIIP)，第452-458页
  - Tang X. 1998. 运行长度矩阵中的纹理信息。IEEE图像处理事务7(11):1602-1609。
  - `Tustison N., Gee J. 纹理分析的运行长度矩阵。Insight Journal 2008年1月 - 6月。
    <http://www.insight-journal.org/browse/publication/231>`_
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsGLRLM, self).__init__(inputImage, inputMask, **kwargs)

    self.weightingNorm = kwargs.get('weightingNorm', None)  # 曼哈顿，欧几里得，无穷大

    self.P_glrlm = None
    self.imageArray = self._applyBinning(self.imageArray)

  def _initCalculation(self, voxelCoordinates=None):
    self.P_glrlm = self._calculateMatrix(voxelCoordinates)

    self._calculateCoefficients()

    self.logger.debug('GLRLM 特征类初始化，计算得到的 GLRLM 形状为 %s', self.P_glrlm.shape)

  def _calculateMatrix(self, voxelCoordinates=None):
    self.logger.debug('用 C 语言计算 GLRLM 矩阵')

    Ng = self.coefficients['Ng']
    Nr = numpy.max(self.imageArray.shape)

    matrix_args = [
      self.imageArray,
      self.maskArray,
      Ng,
      Nr,
      self.settings.get('force2D', False),
      self.settings.get('force2Ddimension', 0)
    ]
    if self.voxelBased:
      matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

    P_glrlm, angles = cMatrices.calculate_glrlm(*matrix_args)  # 形状 (Nvox, Ng, Nr, Na)

    self.logger.debug('处理计算得到的矩阵')

    # 删除指定 ROI 中不存在的灰度级的行
    NgVector = range(1, Ng + 1)  # 所有可能的灰度值
    GrayLevels = self.coefficients['grayLevels']  # ROI 中存在的灰度值
    emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)), dtype=int)  # ROI 中不存在的灰度值

    P_glrlm = numpy.delete(P_glrlm, emptyGrayLevels - 1, 1)

    # 可选地应用加权因子
    if self.weightingNorm is not None:
      self.logger.debug('应用加权 (%s)', self.weightingNorm)

      pixelSpacing = self.inputImage.GetSpacing()[::-1]
      weights = numpy.empty(len(angles))
      for a_idx, a in enumerate(angles):
        if self.weightingNorm == 'infinity':
          weights[a_idx] = max(numpy.abs(a) * pixelSpacing)
        elif self.weightingNorm == 'euclidean':
          weights[a_idx] = numpy.sqrt(numpy.sum((numpy.abs(a) * pixelSpacing) ** 2))
        elif self.weightingNorm == 'manhattan':
          weights[a_idx] = numpy.sum(numpy.abs(a) * pixelSpacing)
        elif self.weightingNorm == 'no_weighting':
          weights[a_idx] = 1
        else:
          self.logger.warning('未知的加权范数 "%s"，加权因子设置为 1', self.weightingNorm)
          weights[a_idx] = 1

      P_glrlm = numpy.sum(P_glrlm * weights[None, None, None, :], 3, keepdims=True)

    Nr = numpy.sum(P_glrlm, (1, 2))

    # 如果没有应用加权，则删除空角度
    if P_glrlm.shape[3] > 1:
      emptyAngles = numpy.where(numpy.sum(Nr, 0) == 0)
      if len(emptyAngles[0]) > 0:  # 一个或多个角度是“空的”
        self.logger.debug('删除 %d 个空角度:\n%s', len(emptyAngles[0]), angles[emptyAngles])
        P_glrlm = numpy.delete(P_glrlm, emptyAngles, 3)
        Nr = numpy.delete(Nr, emptyAngles, 1)
      else:
        self.logger.debug('没有空角度')

    Nr[Nr == 0] = numpy.nan  # 如果和为0，则设置为 numpy.spacing(1)？
    self.coefficients['Nr'] = Nr

    return P_glrlm

  def _calculateCoefficients(self):
    self.logger.debug('计算 GLRLM 系数')

    pr = numpy.sum(self.P_glrlm, 1)  # 形状 (Nvox, Nr, Na)
    pg = numpy.sum(self.P_glrlm, 2)  # 形状 (Nvox, Ng, Na)

    ivector = self.coefficients['grayLevels'].astype(float)  # 形状 (Ng,)
    jvector = numpy.arange(1, self.P_glrlm.shape[2] + 1, dtype=numpy.float64)  # 形状 (Nr,)

    # 删除 ROI 中不存在的运行长度的列
    emptyRunLenghts = numpy.where(numpy.sum(pr, (0, 2)) == 0)
    self.P_glrlm = numpy.delete(self.P_glrlm, emptyRunLenghts, 2)
    jvector = numpy.delete(jvector, emptyRunLenghts)
    pr = numpy.delete(pr, emptyRunLenghts, 1)

    self.coefficients['pr'] = pr
    self.coefficients['pg'] = pg
    self.coefficients['ivector'] = ivector
    self.coefficients['jvector'] = jvector


  def getShortRunEmphasisFeatureValue(self):
    r"""
    **1. 短跑长度强调 (SRE)**

    .. math::
      \textit{SRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)}{j^2}}}{N_r(\theta)}

    SRE 是衡量短跑长度分布的指标，值越大表示跑长度越短，纹理越细腻。
    """
    pr = self.coefficients['pr']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    sre = numpy.sum((pr / (jvector[None, :, None] ** 2)), 1) / Nr
    return numpy.nanmean(sre, 1)

  def getLongRunEmphasisFeatureValue(self):
    r"""
    **2. 长跑长度强调 (LRE)**

    .. math::
      \textit{LRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)j^2}}{N_r(\theta)}

    LRE 是衡量长跑长度分布的指标，值越大表示跑长度越长，结构纹理越粗糙。
    """
    pr = self.coefficients['pr']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    lre = numpy.sum((pr * (jvector[None, :, None] ** 2)), 1) / Nr
    return numpy.nanmean(lre, 1)

  def getGrayLevelNonUniformityFeatureValue(self):
    r"""
    **3. 灰度级非均匀性 (GLN)**

    .. math::
      \textit{GLN} = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)}

    GLN 衡量图像中灰度级强度值的相似性，GLN 值较低表示强度值更相似。
    """
    pg = self.coefficients['pg']
    Nr = self.coefficients['Nr']

    gln = numpy.sum((pg ** 2), 1) / Nr
    return numpy.nanmean(gln, 1)

  def getGrayLevelNonUniformityNormalizedFeatureValue(self):
    r"""
    **4. 灰度级非均匀性归一化 (GLNN)**

    .. math::
      \textit{GLNN} = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)^2}

    GLNN 衡量图像中灰度级强度值的相似性，GLNN 值较低表示强度值更相似。这是 GLN 公式的归一化版本。
    """
    pg = self.coefficients['pg']
    Nr = self.coefficients['Nr']

    glnn = numpy.sum(pg ** 2, 1) / (Nr ** 2)
    return numpy.nanmean(glnn, 1)

  def getRunLengthNonUniformityFeatureValue(self):
    r"""
    **5. 跑长度非均匀性 (RLN)**

    .. math::
      \textit{RLN} = \frac{\sum^{N_r}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)}

    RLN 衡量图像中跑长度的相似性，RLN 值较低表示跑长度在图像中更加均匀。
    """
    pr = self.coefficients['pr']
    Nr = self.coefficients['Nr']

    rln = numpy.sum((pr ** 2), 1) / Nr
    return numpy.nanmean(rln, 1)

  def getRunLengthNonUniformityNormalizedFeatureValue(self):
    r"""
    **6. 跑长度非均匀性归一化 (RLNN)**

    .. math::
      \textit{RLNN} = \frac{\sum^{N_r}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j|\theta)}\right)^2}{N_r(\theta)^2}

    RLNN 衡量图像中跑长度的相似性，RLNN 值较低表示跑长度在图像中更加均匀。这是 RLN 公式的归一化版本。
    """
    pr = self.coefficients['pr']
    Nr = self.coefficients['Nr']

    rlnn = numpy.sum((pr ** 2), 1) / Nr ** 2
    return numpy.nanmean(rlnn, 1)


  def getRunPercentageFeatureValue(self):
    r"""
    **7. 跑长度百分比 (RP)**

    .. math::
      \textit{RP} = {\frac{N_r(\theta)}{N_p}}

    RP 通过计算跑长度数与ROI中体素数的比率来衡量纹理的粗糙度。

    值范围为 :math:`\frac{1}{N_p} \leq RP \leq 1`，较高的值表示ROI由较短的跑长度组成（表示更细的纹理）。

    .. note::
      注意，当应用加权并在计算前合并矩阵时，:math:`N_p` 会乘以合并矩阵的数量 `n`，以确保正确的归一化（因为每个体素被考虑了 `n` 次）。
    """
    pr = self.coefficients['pr']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    Np = numpy.sum(pr * jvector[None, :, None], 1)  # shape (Nvox, Na)

    rp = Nr / Np
    return numpy.nanmean(rp, 1)

  def getGrayLevelVarianceFeatureValue(self):
    r"""
    **8. 灰度级方差 (GLV)**

    .. math::
      \textit{GLV} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)(i - \mu)^2}

    其中, :math:`\mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)i}`

    GLV 衡量跑长度中灰度级强度的方差。
    """
    ivector = self.coefficients['ivector']
    Nr = self.coefficients['Nr']
    pg = self.coefficients['pg'] / Nr[:, None, :]  # 除以 Nr 得到归一化矩阵

    u_i = numpy.sum(pg * ivector[None, :, None], 1, keepdims=True)
    glv = numpy.sum(pg * (ivector[None, :, None] - u_i) ** 2, 1)
    return numpy.nanmean(glv, 1)

  def getRunVarianceFeatureValue(self):
    r"""
    **9. 跑长度方差 (RV)**

    .. math::
      \textit{RV} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)(j - \mu)^2}

    其中, :math:`\mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}{p(i,j|\theta)j}`

    RV 是衡量跑长度方差的指标。
    """
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']
    pr = self.coefficients['pr'] / Nr[:, None, :]   # 除以 Nr 得到归一化矩阵

    u_j = numpy.sum(pr * jvector[None, :, None], 1, keepdims=True)
    rv = numpy.sum(pr * (jvector[None, :, None] - u_j) ** 2, 1)
    return numpy.nanmean(rv, 1)

  def getRunEntropyFeatureValue(self):
    r"""
    **10. 跑长度熵 (RE)**

    .. math::
      \textit{RE} = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_r}_{j=1}
      {p(i,j|\theta)\log_{2}(p(i,j|\theta)+\epsilon)}

    其中, :math:`\epsilon` 是一个极小的正数 (:math:`\approx 2.2\times10^{-16}`)。

    RE 衡量跑长度和灰度级分布的不确定性/随机性。较高的值表示纹理模式中的异质性更大。
    """
    eps = numpy.spacing(1)
    Nr = self.coefficients['Nr']
    p_glrlm = self.P_glrlm / Nr[:, None, None, :]  # 除以 Nr 得到归一化矩阵

    re = -numpy.sum(p_glrlm * numpy.log2(p_glrlm + eps), (1, 2))
    return numpy.nanmean(re, 1)


  def getLowGrayLevelRunEmphasisFeatureValue(self):
    r"""
    **11. 低灰度级跑长度强调 (LGLRE)**

    .. math::
      \textit{LGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)}{i^2}}}{N_r(\theta)}

    LGLRE 衡量低灰度级值的分布，较高的值表示图像中低灰度级值的集中度更高。
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nr = self.coefficients['Nr']

    lglre = numpy.sum((pg / (ivector[None, :, None] ** 2)), 1) / Nr
    return numpy.nanmean(lglre, 1)

  def getHighGrayLevelRunEmphasisFeatureValue(self):
    r"""
    **12. 高灰度级跑长度强调 (HGLRE)**

    .. math::
      \textit{HGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)i^2}}{N_r(\theta)}

    HGLRE 衡量较高灰度级值的分布，较高的值表示图像中高灰度级值的集中度更高。
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nr = self.coefficients['Nr']

    hglre = numpy.sum((pg * (ivector[None, :, None] ** 2)), 1) / Nr
    return numpy.nanmean(hglre, 1)

  def getShortRunLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **13. 短跑长度低灰度级强调 (SRLGLE)**

    .. math::
      \textit{SRLGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)}{i^2j^2}}}{N_r(\theta)}

    SRLGLE 衡量短跑长度与低灰度级值的联合分布。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    srlgle = numpy.sum((self.P_glrlm / ((ivector[None, :, None, None] ** 2) * (jvector[None, None, :, None] ** 2))),
                       (1, 2)) / Nr
    return numpy.nanmean(srlgle, 1)

  def getShortRunHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **14. 短跑长度高灰度级强调 (SRHGLE)**

    .. math::
      \textit{SRHGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)i^2}{j^2}}}{N_r(\theta)}

    SRHGLE 衡量短跑长度与高灰度级值的联合分布。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    srhgle = numpy.sum((self.P_glrlm * (ivector[None, :, None, None] ** 2) / (jvector[None, None, :, None] ** 2)),
                       (1, 2)) / Nr
    return numpy.nanmean(srhgle, 1)

  def getLongRunLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **15. 长跑长度低灰度级强调 (LRLGLE)**

    .. math::
      \textit{LRLGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\frac{\textbf{P}(i,j|\theta)j^2}{i^2}}}{N_r(\theta)}

    LRLGLRE 衡量长跑长度与低灰度级值的联合分布。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    lrlgle = numpy.sum((self.P_glrlm * (jvector[None, None, :, None] ** 2) / (ivector[None, :, None, None] ** 2)),
                       (1, 2)) / Nr
    return numpy.nanmean(lrlgle, 1)

  def getLongRunHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **16. 长跑长度高灰度级强调 (LRHGLE)**

    .. math::
      \textit{LRHGLRE} = \frac{\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)i^2j^2}}{N_r(\theta)}

    LRHGLRE 衡量长跑长度与高灰度级值的联合分布。
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nr = self.coefficients['Nr']

    lrhgle = numpy.sum((self.P_glrlm * ((jvector[None, None, :, None] ** 2) * (ivector[None, :, None, None] ** 2))),
                       (1, 2)) / Nr
    return numpy.nanmean(lrhgle, 1)



















