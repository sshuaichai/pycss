import numpy
from six.moves import range

from radiomics import base, cMatrices, deprecated

class RadiomicsGLCM(base.RadiomicsFeaturesBase):
  r"""
  灰度共生矩阵（GLCM）大小为 :math:`N_g \times N_g`，描述了由掩码限制的图像区域的二阶联合概率函数，
  定义为 :math:`\textbf{P}(i,j|\delta,\theta)`。此矩阵的 :math:`(i,j)^{\text{th}}` 元素表示图像中两个像素间
  灰度级 :math:`i` 和 :math:`j` 组合出现的次数，这两个像素沿角度 :math:`\theta` 方向相隔 :math:`\delta` 个像素距离。
  距离 :math:`\delta` 是根据无穷范数定义的中心体素的距离。对于 :math:`\delta=1`，这将在 3D 中为每个角度的 2 个邻居
  产生 13 个角度（26-连通性），对于 :math:`\delta=2` 则为 98-连通性（49 个唯一角度）。

  注意 pyradiomics 默认计算对称 GLCM！

  作为二维示例，让下面的矩阵 :math:`\textbf{I}` 代表一个 5x5 图像，具有 5 个离散的灰度级：

  .. math::
    \textbf{I} = \begin{bmatrix}
    1 & 2 & 5 & 2 & 3\\
    3 & 2 & 1 & 3 & 1\\
    1 & 3 & 5 & 5 & 2\\
    1 & 1 & 1 & 1 & 2\\
    1 & 2 & 4 & 3 & 5 \end{bmatrix}

  对于距离 :math:`\delta = 1`（考虑与每个像素相距 1 个像素的像素）和角度 :math:`\theta=0^\circ`（水平面，即中心体素左右的体素），
  得到以下对称 GLCM：

  .. math::
    \textbf{P} = \begin{bmatrix}
    6 & 4 & 3 & 0 & 0\\
    4 & 0 & 2 & 1 & 3\\
    3 & 2 & 0 & 1 & 2\\
    0 & 1 & 1 & 0 & 0\\
    0 & 3 & 2 & 0 & 2 \end{bmatrix}

  让：

  - :math:`\epsilon` 是一个任意小的正数 (:math:`\approx 2.2\times10^{-16}`)
  - :math:`\textbf{P}(i,j)` 是任意 :math:`\delta` 和 :math:`\theta` 的共生矩阵
  - :math:`p(i,j)` 是归一化的共生矩阵，等于 :math:`\frac{\textbf{P}(i,j)}{\sum{\textbf{P}(i,j)}}`
  - :math:`N_g` 是图像中离散强度级别的数量
  - :math:`p_x(i) = \sum^{N_g}_{j=1}{p(i,j)}` 是边际行概率
  - :math:`p_y(j) = \sum^{N_g}_{i=1}{p(i,j)}` 是边际列概率
  - :math:`\mu_x` 是 :math:`p_x` 的平均灰度强度，定义为 :math:`\mu_x = \displaystyle\sum^{N_g}_{i=1}{p_x(i)i}`
  - :math:`\mu_y` 是 :math:`p_y` 的平均灰度强度，定义为 :math:`\mu_y = \displaystyle\sum^{N_g}_{j=1}{p_y(j)j}`
  - :math:`\sigma_x` 是 :math:`p_x` 的标准差
  - :math:`\sigma_y` 是 :math:`p_y` 的标准差
  - :math:`p_{x+y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)},\text{ 其中 }i+j=k,\text{ 且 }k=2,3,\dots,2N_g`
  - :math:`p_{x-y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)},\text{ 其中 }|i-j|=k,\text{ 且 }k=0,1,\dots,N_g-1`
  - :math:`HX =  -\sum^{N_g}_{i=1}{p_x(i)\log_2\big(p_x(i)+\epsilon\big)}` 是 :math:`p_x` 的熵
  - :math:`HY =  -\sum^{N_g}_{j=1}{p_y(j)\log_2\big(p_y(j)+\epsilon\big)}` 是 :math:`p_y` 的熵
  - :math:`HXY =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)\log_2\big(p(i,j)+\epsilon\big)}` 是 :math:`p(i,j)` 的熵
  - :math:`HXY1 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)\log_2\big(p_x(i)p_y(j)+\epsilon\big)}`
  - :math:`HXY2 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_x(i)p_y(j)\log_2\big(p_x(i)p_y(j)+\epsilon\big)}`

  默认情况下，特征值是在每个角度的 GLCM 上单独计算的，之后返回这些值的平均值。如果启用了距离加权，GLCM 矩阵通过加权因子 W 加权，
  然后求和并归一化。然后在结果矩阵上计算特征。加权因子 W 是通过以下方式计算相邻体素之间的距离：

  :math:`W = e^{-\|d\|^2}`，其中 d 是根据设置 'weightingNorm' 中指定的范数计算的相关角度的距离。

  以下类特定设置是可能的：

  - distances [[1]]: 整数列表。这指定了中心体素和邻居之间的距离，应为其生成角度。
  - symmetricalGLCM [True]: 布尔值，指示是否应该按每个角度的两个方向评估共生现象，这将导致对称矩阵，对于 :math:`i` 和 :math:`j` 有相等的分布。对称矩阵对应于 Haralick 等人定义的 GLCM。
  - weightingNorm [None]: 字符串，指示在应用距离加权时应使用哪种范数。枚举设置，可能的值：

    - 'manhattan': 一阶范数
    - 'euclidean': 二阶范数
    - 'infinity': 无穷范数。
    - 'no_weighting': GLCM 通过因子 1 加权并求和
    - None: 不应用加权，返回单独矩阵上计算的值的平均值。

    如果是其他值，将记录警告并使用选项 'no_weighting'。

  参考文献

  - Haralick, R., Shanmugan, K., Dinstein, I; 图像分类的纹理特征; IEEE 系统、人与控制论交易; 1973(3), p610-621
  - `<https://en.wikipedia.org/wiki/Co-occurrence_matrix>`_
  - `<http://www.fp.ucalgary.ca/mhallbey/the_glcm.htm>`_
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsGLCM, self).__init__(inputImage, inputMask, **kwargs)

    self.symmetricalGLCM = kwargs.get('symmetricalGLCM', True)
    self.weightingNorm = kwargs.get('weightingNorm', None)  # 'manhattan', 'euclidean', 'infinity'

    self.P_glcm = None
    self.imageArray = self._applyBinning(self.imageArray)

  def _initCalculation(self, voxelCoordinates=None):
    self.P_glcm = self._calculateMatrix(voxelCoordinates)

    self._calculateCoefficients()

    self.logger.debug('GLCM 特征类初始化，计算得到的 GLCM 形状为 %s', self.P_glcm.shape)

  def _calculateMatrix(self, voxelCoordinates=None):
    r"""
    对于每个方向在 3D 中计算输入图像的 GLCMs。
    计算的 GLCMs 放置在数组 P_glcm 中，形状为 (i/j, a)
    i/j = 图像数组的总灰度级别，
    a = 3D 中的方向（由 imageoperations.generateAngles 生成）
    """
    self.logger.debug('在 C 中计算 GLCM 矩阵')

    Ng = self.coefficients['Ng']

    matrix_args = [
      self.imageArray,
      self.maskArray,
      numpy.array(self.settings.get('distances', [1])),
      Ng,
      self.settings.get('force2D', False),
      self.settings.get('force2Ddimension', 0)
    ]
    if self.voxelBased:
      matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

    P_glcm, angles = cMatrices.calculate_glcm(*matrix_args)

    self.logger.debug('处理计算得到的矩阵')

    # 删除指定 ROI 中不存在的灰度级的行和列
    NgVector = range(1, Ng + 1)  # 所有可能的灰度值
    GrayLevels = self.coefficients['grayLevels']  # ROI 中存在的灰度值
    emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)), dtype=int)  # ROI 中不存在的灰度值

    P_glcm = numpy.delete(P_glcm, emptyGrayLevels - 1, 1)
    P_glcm = numpy.delete(P_glcm, emptyGrayLevels - 1, 2)

    # 可选地使每个角度的 GLCMs 对称
    if self.symmetricalGLCM:
      self.logger.debug('创建对称矩阵')
      # 转置并复制 GLCM 并将其添加到 P_glcm。Numpy.transpose 返回一个视图（如果可能），使用 .copy() 确保
      # 使用数组的副本而不仅仅是视图（否则可能会发生错误的添加）
      P_glcm += numpy.transpose(P_glcm, (0, 2, 1, 3)).copy()

    # 可选地应用加权因子
    if self.weightingNorm is not None:
      self.logger.debug('应用加权 (%s)', self.weightingNorm)
      pixelSpacing = self.inputImage.GetSpacing()[::-1]
      weights = numpy.empty(len(angles))
      for a_idx, a in enumerate(angles):
        if self.weightingNorm == 'infinity':
          weights[a_idx] = numpy.exp(-max(numpy.abs(a) * pixelSpacing) ** 2)
        elif self.weightingNorm == 'euclidean':
          weights[a_idx] = numpy.exp(-numpy.sum((numpy.abs(a) * pixelSpacing) ** 2))  # sqrt ^ 2 = 1
        elif self.weightingNorm == 'manhattan':
          weights[a_idx] = numpy.exp(-numpy.sum(numpy.abs(a) * pixelSpacing) ** 2)
        elif self.weightingNorm == 'no_weighting':
          weights[a_idx] = 1
        else:
          self.logger.warning('未知的加权范数 "%s"，W 设置为 1', self.weightingNorm)
          weights[a_idx] = 1

      P_glcm = numpy.sum(P_glcm * weights[None, None, None, :], 3, keepdims=True)

    sumP_glcm = numpy.sum(P_glcm, (1, 2))

    # 如果没有应用加权，则删除空角度
    if P_glcm.shape[3] > 1:
      emptyAngles = numpy.where(numpy.sum(sumP_glcm, 0) == 0)
      if len(emptyAngles[0]) > 0:  # 一个或多个角度是“空”的
        self.logger.debug('删除 %d 个空角度:\n%s', len(emptyAngles[0]), angles[emptyAngles])
        P_glcm = numpy.delete(P_glcm, emptyAngles, 3)
        sumP_glcm = numpy.delete(sumP_glcm, emptyAngles, 1)
      else:
        self.logger.debug('没有空角度')

    # 用 NaN 标记空角度，允许在特征计算中忽略它们
    sumP_glcm[sumP_glcm == 0] = numpy.nan
    # 归一化每个 glcm
    P_glcm /= sumP_glcm[:, None, None, :]

    return P_glcm

  # 检查 ivector 和 jvector 是否可以被替换
  def _calculateCoefficients(self):
    r"""
    计算并填充系数字典。
    """
    self.logger.debug('计算 GLCM 系数')

    Ng = self.coefficients['Ng']
    eps = numpy.spacing(1)

    NgVector = self.coefficients['grayLevels'].astype('float')
    # 形状 = (Ng, Ng)
    i, j = numpy.meshgrid(NgVector, NgVector, indexing='ij', sparse=True)

    # 形状 = (2*Ng-1)
    kValuesSum = numpy.arange(2, (Ng * 2) + 1, dtype='float')
    # 形状 = (Ng-1)
    kValuesDiff = numpy.arange(0, Ng, dtype='float')

    # 边缘行概率 #形状 = (Nv, Ng, 1, angles)
    px = self.P_glcm.sum(2, keepdims=True)
    # 边缘列概率 #形状 = (Nv, 1, Ng, angles)
    py = self.P_glcm.sum(1, keepdims=True)

    # 形状 = (Nv, 1, 1, angles)
    ux = numpy.sum(i[None, :, :, None] * self.P_glcm, (1, 2), keepdims=True)
    uy = numpy.sum(j[None, :, :, None] * self.P_glcm, (1, 2), keepdims=True)

    # 形状 = (Nv, 2*Ng-1, angles)
    pxAddy = numpy.array([numpy.sum(self.P_glcm[:, i + j == k, :], 1) for k in kValuesSum]).transpose((1, 0, 2))
    # 形状 = (Nv, Ng, angles)
    pxSuby = numpy.array([numpy.sum(self.P_glcm[:, numpy.abs(i - j) == k, :], 1) for k in kValuesDiff]).transpose((1, 0, 2))

    # 形状 = (Nv, angles)
    HXY = (-1) * numpy.sum((self.P_glcm * numpy.log2(self.P_glcm + eps)), (1, 2))

    self.coefficients['eps'] = eps
    self.coefficients['i'] = i
    self.coefficients['j'] = j
    self.coefficients['kValuesSum'] = kValuesSum
    self.coefficients['kValuesDiff'] = kValuesDiff
    self.coefficients['px'] = px
    self.coefficients['py'] = py
    self.coefficients['ux'] = ux
    self.coefficients['uy'] = uy
    self.coefficients['pxAddy'] = pxAddy
    self.coefficients['pxSuby'] = pxSuby
    self.coefficients['HXY'] = HXY

  def getAutocorrelationFeatureValue(self):
    r"""
    **1. 自相关**

    .. math::
      \textit{autocorrelation} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{p(i,j)ij}

    自相关是衡量纹理的细腻度和粗糙度的大小的指标。
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ac = numpy.sum(self.P_glcm * (i * j)[None, :, :, None], (1, 2))
    return numpy.nanmean(ac, 1)

  def getJointAverageFeatureValue(self):
    r"""
    **2. 联合平均**

    .. math::
      \textit{joint average} = \mu_x = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{p(i,j)i}

    返回分布 :math:`i` 的平均灰度强度。

    .. warning::
      由于这个公式代表了 :math:`i` 的分布的平均值，它与 :math:`j` 的分布无关。因此，只有当 GLCM 是对称的时候才使用这个公式，
      其中 :math:`p_x(i) = p_y(j) \text{, 其中 } i = j`。
    """
    if not self.symmetricalGLCM:
      self.logger.warning('GLCM - 联合平均的公式假设 GLCM 是对称的，但实际上并非如此。')
    return self.coefficients['ux'].mean((1, 2, 3))

  def getClusterProminenceFeatureValue(self):
    r"""
    **3. 簇突出度**

    .. math::
      \textit{cluster prominence} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {\big( i+j-\mu_x-\mu_y\big)^4p(i,j)}

    簇突出度是 GLCM 的偏斜度和不对称性的度量。较高的值意味着关于平均值的更大不对称性，较低的值表示在平均值附近有一个峰值，关于平均值的变化较小。
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    cp = numpy.sum((self.P_glcm * (((i + j)[None, :, :, None] - ux - uy) ** 4)), (1, 2))
    return numpy.nanmean(cp, 1)

  def getClusterShadeFeatureValue(self):
    r"""
    **4. 簇阴影**

    .. math::
      \textit{cluster shade} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {\big(i+j-\mu_x-\mu_y\big)^3p(i,j)}

    簇阴影是 GLCM 的偏斜度和均匀性的度量。较高的簇阴影意味着关于平均值的更大不对称性。
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    cs = numpy.sum((self.P_glcm * (((i + j)[None, :, :, None] - ux - uy) ** 3)), (1, 2))
    return numpy.nanmean(cs, 1)

  def getClusterTendencyFeatureValue(self):
    r"""
    **5. 簇倾向**

    .. math::
      \textit{cluster tendency} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {\big(i+j-\mu_x-\mu_y\big)^2p(i,j)}

    簇倾向是衡量具有相似灰度值的体素的分组的度量。
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    ct = numpy.sum((self.P_glcm * (((i + j)[None, :, :, None] - ux - uy) ** 2)), (1, 2))
    return numpy.nanmean(ct, 1)

  def getContrastFeatureValue(self):
    r"""
    **6. 对比度**

    .. math::
      \textit{contrast} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{(i-j)^2p(i,j)}

    对比度是局部强度变化的度量，偏好远离对角线 :math:`(i = j)` 的值。较大的值与相邻体素间的强度值差异较大相关联。
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    cont = numpy.sum((self.P_glcm * ((numpy.abs(i - j))[None, :, :, None] ** 2)), (1, 2))
    return numpy.nanmean(cont, 1)

  def getCorrelationFeatureValue(self):
    r"""
    **7. 相关性**

    .. math::
      \textit{correlation} = \frac{\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)ij-\mu_x\mu_y}}{\sigma_x(i)\sigma_y(j)}

    相关性是一个介于 0（无关联）和 1（完全相关）之间的值，显示了灰度级值与其在 GLCM 中的相应体素的线性依赖性。

    .. note::
      当 ROI 中只有一个离散的灰度值（平坦区域）时，:math:`\sigma_x` 和 :math:`\sigma_y` 将为 0。在这种情况下，返回一个任意值 1。这是基于每个角度分别评估的。
    """
    eps = self.coefficients['eps']
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']

    # 形状 = (Nv, 1, 1, angles)
    sigx = numpy.sum(self.P_glcm * ((i[None, :, :, None] - ux) ** 2), (1, 2), keepdims=True) ** 0.5
    # 形状 = (Nv, 1, 1, angles)
    sigy = numpy.sum(self.P_glcm * ((j[None, :, :, None] - uy) ** 2), (1, 2), keepdims=True) ** 0.5

    corm = numpy.sum(self.P_glcm * (i[None, :, :, None] - ux) * (j[None, :, :, None] - uy), (1, 2), keepdims=True)
    corr = corm / (sigx * sigy + eps)
    corr[sigx * sigy == 0] = 1  # 将会被 0 除的元素设置为 1。
    return numpy.nanmean(corr, (1, 2, 3))

  def getDifferenceAverageFeatureValue(self):
    r"""
    **8. 差异平均值**

    .. math::
      \textit{difference average} = \displaystyle\sum^{N_g-1}_{k=0}{kp_{x-y}(k)}

    差异平均值衡量了具有相似强度值的对出现的关系与具有不同强度值的对出现的关系。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    diffavg = numpy.sum((kValuesDiff[None, :, None] * pxSuby), 1)
    return numpy.nanmean(diffavg, 1)

  def getDifferenceEntropyFeatureValue(self):
    r"""
    **9. 差异熵**

    .. math::
      \textit{difference entropy} = \displaystyle\sum^{N_g-1}_{k=0}{p_{x-y}(k)\log_2\big(p_{x-y}(k)+\epsilon\big)}

    差异熵是邻域强度值差异的随机性/变异性的度量。
    """
    pxSuby = self.coefficients['pxSuby']
    eps = self.coefficients['eps']
    difent = (-1) * numpy.sum((pxSuby * numpy.log2(pxSuby + eps)), 1)
    return numpy.nanmean(difent, 1)

  def getDifferenceVarianceFeatureValue(self):
    r"""
    **10. 差异方差**

    .. math::
      \textit{difference variance} = \displaystyle\sum^{N_g-1}_{k=0}{(k-DA)^2p_{x-y}(k)}

    差异方差是一种异质性度量，对偏离平均值更多的不同强度级别对赋予更高的权重。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    diffavg = numpy.sum((kValuesDiff[None, :, None] * pxSuby), 1, keepdims=True)
    diffvar = numpy.sum((pxSuby * ((kValuesDiff[None, :, None] - diffavg) ** 2)), 1)
    return numpy.nanmean(diffvar, 1)

  def getJointEnergyFeatureValue(self):
    r"""
    **11. 联合能量**

    .. math::
      \textit{joint energy} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{\big(p(i,j)\big)^2}

    能量是图像中均匀模式的度量。更大的能量意味着图像中存在更多的相邻强度值对，它们以更高的频率相邻。

    .. note::
      由 IBSI 定义为角二阶矩。
    """
    ene = numpy.sum((self.P_glcm ** 2), (1, 2))
    return numpy.nanmean(ene, 1)

  def getJointEntropyFeatureValue(self):
    r"""
    **12. 联合熵**

    .. math::
      \textit{joint entropy} = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {p(i,j)\log_2\big(p(i,j)+\epsilon\big)}

    联合熵是邻域强度值的随机性/变异性的度量。

    .. note::
      由 IBSI 定义为联合熵。
    """
    ent = self.coefficients['HXY']
    return numpy.nanmean(ent, 1)

  def getIdmFeatureValue(self):
    r"""
    **15. 逆差矩 (IDM)**

    .. math::
      \textit{IDM} = \displaystyle\sum^{N_g-1}_{k=0}{\frac{p_{x-y}(k)}{1+k^2}}

    IDM（也称为均匀性2）是图像局部均匀性的度量。IDM的权重是对比度权重的倒数（从GLCM的对角线 :math:`i=j` 开始指数级递减）。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    idm = numpy.sum(pxSuby / (1 + (kValuesDiff[None, :, None] ** 2)), 1)
    return numpy.nanmean(idm, 1)

  def getMCCFeatureValue(self):
    r"""
    **16. 最大相关系数 (MCC)**

    .. math::
      \textit{MCC} = \sqrt{\text{Q的第二大特征值}}

      Q(i, j) = \displaystyle\sum^{N_g}_{k=0}{\frac{p(i,k)p(j, k)}{p_x(i)p_y(k)}}

    最大相关系数是纹理复杂性的度量，范围为 :math:`0 \leq MCC \leq 1`。

    在平坦区域的情况下，每个GLCM矩阵的形状为 (1, 1)，因此只有1个特征值。在这种情况下，返回一个任意值1。
    """
    px = self.coefficients['px']
    py = self.coefficients['py']
    eps = self.coefficients['eps']

    # 计算 Q (形状 (i, i, d))。为了防止除以0，添加 epsilon（当ROI中沿某个角度的体素在灰度级i没有邻居时，可能发生这种除法）
    Q = ((self.P_glcm[:, :, None, 0, :] * self.P_glcm[:, None, :, 0, :]) /  # 切片: v, i, j, k, d
         (px[:, :, None, 0, :] * py[:, None, :, 0, :] + eps))  # 在k（第4轴-->索引3）上求和

    for gl in range(1, self.P_glcm.shape[1]):
      Q += ((self.P_glcm[:, :, None, gl, :] * self.P_glcm[:, None, :, gl, :]) /  # 切片: v, i, j, k, d
            (px[:, :, None, 0, :] * py[:, None, :, gl, :] + eps))  # 在k（第4轴-->索引3）上求和

    # 如果在最后2个维度上执行特征值计算，则将角度维度（d）向前移动
    Q_eigenValue = numpy.linalg.eigvals(Q.transpose((0, 3, 1, 2)))
    Q_eigenValue.sort()  # 沿最后一个轴排序 --> 特征值，从低到高

    if Q_eigenValue.shape[2] < 2:
      return 1  # 平坦区域

    MCC = numpy.sqrt(Q_eigenValue[:, :, -2])  # 第二高的特征值

    return numpy.nanmean(MCC, 1).real

  def getIdmnFeatureValue(self):
    r"""
    **17. 逆差矩归一化 (IDMN)**

    .. math::
      \textit{IDMN} = \displaystyle\sum^{N_g-1}_{k=0}{ \frac{p_{x-y}(k)}{1+\left(\frac{k^2}{N_g^2}\right)} }

    IDMN（逆差矩归一化）是图像局部均匀性的度量。与Homogeneity2不同，IDMN通过除以离散强度值总数的平方来归一化相邻强度值之间的差的平方。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    Ng = self.coefficients['Ng']
    idmn = numpy.sum(pxSuby / (1 + ((kValuesDiff[None, :, None] ** 2) / (Ng ** 2))), 1)
    return numpy.nanmean(idmn, 1)

  def getIdFeatureValue(self):
    r"""
    **18. 逆差 (ID)**

    .. math::
      \textit{ID} = \displaystyle\sum^{N_g-1}_{k=0}{\frac{p_{x-y}(k)}{1+k}}

    ID（也称为均匀性1）是图像局部均匀性的另一种度量。在灰度级更均匀的情况下，分母将保持较低，导致整体值较高。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    invDiff = numpy.sum(pxSuby / (1 + kValuesDiff[None, :, None]), 1)
    return numpy.nanmean(invDiff, 1)

  def getIdnFeatureValue(self):
    r"""
    **19. 逆差归一化 (IDN)**

    .. math::
      \textit{IDN} = \displaystyle\sum^{N_g-1}_{k=0}{ \frac{p_{x-y}(k)}{1+\left(\frac{k}{N_g}\right)} }

    IDN（逆差归一化）是图像局部均匀性的另一种度量。与Homogeneity1不同，IDN通过除以离散强度值总数来归一化相邻强度值之间的差。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    Ng = self.coefficients['Ng']
    idn = numpy.sum(pxSuby / (1 + (kValuesDiff[None, :, None] / Ng)), 1)
    return numpy.nanmean(idn, 1)

  def getInverseVarianceFeatureValue(self):
    r"""
    **20. 逆方差**

    .. math::
      \textit{inverse variance} = \displaystyle\sum^{N_g-1}_{k=1}{\frac{p_{x-y}(k)}{k^2}}

    注意 :math:`k=0` 被跳过，因为这将导致除以0的情况。
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    inv = numpy.sum(pxSuby[:, 1:, :] / kValuesDiff[None, 1:, None] ** 2, 1)  # 跳过 k = 0 (除以0)
    return numpy.nanmean(inv, 1)

  def getMaximumProbabilityFeatureValue(self):
    r"""
    **21. 最大概率**

    .. math::

      \textit{maximum probability} = \max\big(p(i,j)\big)

    最大概率是图像中最主要的相邻强度值对出现的次数。

    .. note::
      由 IBSI 定义为联合最大值。
    """
    maxprob = numpy.amax(self.P_glcm, (1, 2))
    return numpy.nanmean(maxprob, 1)

  def getSumAverageFeatureValue(self):
    r"""
    **22. 和平均 (Sum Average)**

    .. math::

      \textit{sum average} = \displaystyle\sum^{2N_g}_{k=2}{p_{x+y}(k)k}

    和平均衡量了低强度值对出现的关系与高强度值对出现的关系。

    .. warning::
      当GLCM是对称的时，:math:`\mu_x = \mu_y`，因此 :math:`\text{Sum Average} = \mu_x + \mu_y =
      2 \mu_x = 2 * 联合平均`。请参阅 :ref:`这里 <radiomics-excluded-sumvariance-label>` 的公式 (4.)、(5.) 和 (6.) 证明 :math:`\text{Sum Average} = \mu_x + \mu_y`。
      在 ``examples/exampleSettings`` 中提供的默认参数文件中，此特征已被禁用。
    """
    # 如果GLCM是对称的并且计算了这个特征，警告用户（因为它与联合平均线性相关）
    if self.symmetricalGLCM:
      self.logger.warning('GLCM是对称的，因此和平均 = 2 * 联合平均，只需要计算一个')

    pxAddy = self.coefficients['pxAddy']
    kValuesSum = self.coefficients['kValuesSum']
    sumavg = numpy.sum((kValuesSum[None, :, None] * pxAddy), 1)
    return numpy.nanmean(sumavg, 1)

  def getSumEntropyFeatureValue(self):
    r"""
    **23. 和熵 (Sum Entropy)**

    .. math::

      \textit{sum entropy} = \displaystyle\sum^{2N_g}_{k=2}{p_{x+y}(k)\log_2\big(p_{x+y}(k)+\epsilon\big)}

    和熵是邻域强度值差异的总和。
    """
    pxAddy = self.coefficients['pxAddy']
    eps = self.coefficients['eps']
    sumentr = (-1) * numpy.sum((pxAddy * numpy.log2(pxAddy + eps)), 1)
    return numpy.nanmean(sumentr, 1)

  def getSumSquaresFeatureValue(self):
    r"""
    **24. 平方和 (Sum of Squares)**

    .. math::

      \textit{sum squares} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{(i-\mu_x)^2p(i,j)}

    平方和或方差是GLCM中邻近强度级对的分布关于平均强度级的度量。

    .. warning::

      这个公式代表了 :math:`i` 的分布的方差，与 :math:`j` 的分布无关。因此，只有当GLCM是对称的时，:math:`p_x(i) = p_y(j) \text{, 其中 } i = j` 才使用这个公式

    .. note::
      由 IBSI 定义为联合方差。
    """
    if not self.symmetricalGLCM:
      self.logger.warning('GLCM - 平方和的公式假设GLCM是对称的，但实际并非如此。')
    i = self.coefficients['i']
    ux = self.coefficients['ux']
    # 也称为方差
    ss = numpy.sum((self.P_glcm * ((i[None, :, :, None] - ux) ** 2)), (1, 2))
    return numpy.nanmean(ss, 1)


