
import numpy

from radiomics import base, cMatrices

# 定义一个类，用于计算放射组学的邻近灰度差异矩阵（NGTDM）
class RadiomicsNGTDM(base.RadiomicsFeaturesBase):
  """
  邻近灰度差异矩阵（NGTDM）量化了一个灰度值与其邻居的平均灰度值之间的差异，距离为 :math:`\delta`。
  矩阵中存储了灰度级 :math:`i` 的绝对差异之和。
  设 :math:`\textbf{X}_{gl}` 为一组分割的体素，:math:`x_{gl}(j_x,j_y,j_z) \in \textbf{X}_{gl}` 为位置
  :math:`(j_x,j_y,j_z)` 处体素的灰度级，则邻域的平均灰度级为：

  .. math::

    \bar{A}_i &= \bar{A}(j_x, j_y, j_z) \\
    &= \displaystyle\frac{1}{W} \displaystyle\sum_{k_x=-\delta}^{\delta}\displaystyle\sum_{k_y=-\delta}^{\delta}
    \displaystyle\sum_{k_z=-\delta}^{\delta}{x_{gl}(j_x+k_x, j_y+k_y, j_z+k_z)}, \\
    &\mbox{其中 }(k_x,k_y,k_z)\neq(0,0,0)\mbox{ 且 } x_{gl}(j_x+k_x, j_y+k_y, j_z+k_z) \in \textbf{X}_{gl}

  这里，:math:`W` 是邻域内也在 :math:`\textbf{X}_{gl}` 中的体素数量。

  作为一个二维示例，设以下矩阵 :math:`\textbf{I}` 代表一个4x4图像，
  有5个离散的灰度级，但没有灰度级为4的体素：

  .. math::
    \textbf{I} = \begin{bmatrix}
    1 & 2 & 5 & 2\\
    3 & 5 & 1 & 3\\
    1 & 3 & 5 & 5\\
    3 & 1 & 1 & 1\end{bmatrix}

  可以得到以下的NGTDM：

  .. math::
    \begin{array}{cccc}
    i & n_i & p_i & s_i\\
    \hline
    1 & 6 & 0.375 & 13.35\\
    2 & 2 & 0.125 & 2.00\\
    3 & 4 & 0.25  & 2.63\\
    4 & 0 & 0.00  & 0.00\\
    5 & 4 & 0.25  & 10.075\end{array}

  6个像素的灰度级为1，因此：

  :math:`s_1 = |1-10/3| + |1-30/8| + |1-15/5| + |1-13/5| + |1-15/5| + |1-11/3| = 13.35`

  对于灰度级2，有2个像素，因此：

  :math:`s_2 = |2-15/5| + |2-9/3| = 2`

  对于灰度值3和5也是类似的：

  :math:`s_3 = |3-12/5| + |3-18/5| + |3-20/8| + |3-5/3| = 3.03 \\
  s_5 = |5-14/5| + |5-18/5| + |5-20/8| + |5-11/5| = 10.075`

  设：

  :math:`n_i` 为 :math:`X_{gl}` 中灰度级为 :math:`i` 的体素数量

  :math:`N_{v,p}` 为 :math:`X_{gl}` 中体素的总数量，等于 :math:`\sum{n_i}`（即，有至少一个邻居的有效区域的体素数量）。:math:`N_{v,p} \leq N_p`，其中 :math:`N_p` 是ROI中体素的总数量。

  :math:`p_i` 为灰度级概率，等于 :math:`n_i/N_v`

  :math:`s_i = \left\{ {\begin{array} {rcl}
  \sum^{n_i}{|i-\bar{A}_i|} & \mbox{对于} & n_i \neq 0 \\
  0 & \mbox{对于} & n_i = 0 \end{array}}\right.`
  为灰度级 :math:`i` 的绝对差异之和

  :math:`N_g` 为离散灰度级的数量

  :math:`N_{g,p}` 为 :math:`p_i \neq 0` 的灰度级数量

  可以设置以下类特定的设置：

  - distances [[1]]: 整数列表。这指定了中心体素与邻居之间的距离，应该生成哪些角度。

  参考文献

  - Amadasun M, King R; Textural features corresponding to textural properties;
    Systems, Man and Cybernetics, IEEE Transactions on 19:1264-1274 (1989). doi: 10.1109/21.44046
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsNGTDM, self).__init__(inputImage, inputMask, **kwargs)

    self.P_ngtdm = None
    self.imageArray = self._applyBinning(self.imageArray)

  def _initCalculation(self, voxelCoordinates=None):
    self.P_ngtdm = self._calculateMatrix(voxelCoordinates)
    self._calculateCoefficients()

  def _calculateMatrix(self, voxelCoordinates=None):
    matrix_args = [
      self.imageArray,
      self.maskArray,
      numpy.array(self.settings.get('distances', [1])),
      self.coefficients['Ng'],
      self.settings.get('force2D', False),
      self.settings.get('force2Ddimension', 0)
    ]
    if self.voxelBased:
      matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

    P_ngtdm = cMatrices.calculate_ngtdm(*matrix_args)  # shape (Nvox, Ng, 3)

    # 删除空的灰度级
    emptyGrayLevels = numpy.where(numpy.sum(P_ngtdm[:, :, 0], 0) == 0)
    P_ngtdm = numpy.delete(P_ngtdm, emptyGrayLevels, 1)

    return P_ngtdm

  def _calculateCoefficients(self):
    # 有有效区域的体素数量，小于等于Np
    Nvp = numpy.sum(self.P_ngtdm[:, :, 0], 1)  # shape (Nvox,)
    self.coefficients['Nvp'] = Nvp  # shape (Nv,)

    # 标准化 P_ngtdm[:, 0] (= n_i) 以获得 p_i
    self.coefficients['p_i'] = self.P_ngtdm[:, :, 0] / Nvp[:, None]

    self.coefficients['s_i'] = self.P_ngtdm[:, :, 1]
    self.coefficients['ivector'] = self.P_ngtdm[:, :, 2]

    # Ngp = p_i > 0 的灰度级数量
    self.coefficients['Ngp'] = numpy.sum(self.P_ngtdm[:, :, 0] > 0, 1)

    p_zero = numpy.where(self.coefficients['p_i'] == 0)
    self.coefficients['p_zero'] = p_zero

  def getCoarsenessFeatureValue(self):
    """
    计算并返回粗糙度。

    :math:`Coarseness = \frac{1}{\sum^{N_g}_{i=1}{p_{i}s_{i}}}`

    粗糙度是中心体素与其邻域之间平均差异的度量，是空间变化率的指示。较高的值表示较低的空间变化率和局部更均匀的纹理。

    注意 :math:`\sum^{N_g}_{i=1}{p_{i}s_{i}}` 可能评估为0（在完全均匀的图像中）。如果是这种情况，返回一个任意值 :math:`10^6`。
    """
    p_i = self.coefficients['p_i']
    s_i = self.coefficients['s_i']
    sum_coarse = numpy.sum(p_i * s_i, 1)

    sum_coarse[sum_coarse != 0] = 1 / sum_coarse[sum_coarse != 0]
    sum_coarse[sum_coarse == 0] = 1e6
    return sum_coarse

  def getContrastFeatureValue(self):
    """
    计算并返回对比度。

    :math:`Contrast = \left(\frac{1}{N_{g,p}(N_{g,p}-1)}\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{p_{i}p_{j}(i-j)^2}\right)
    \left(\frac{1}{N_{v,p}}\displaystyle\sum^{N_g}_{i=1}{s_i}\right)\text{, 其中 }p_i \neq 0, p_j \neq 0`

    对比度是空间强度变化的度量，但也依赖于整体灰度级动态范围。当动态范围和空间变化率都高时，对比度高，即具有大范围灰度级的图像，体素及其邻域之间有大的变化。

    注意 在完全均匀的图像中，:math:`N_{g,p} = 1`，这将导致除以0。在这种情况下，返回一个任意值0。
    """
    Ngp = self.coefficients['Ngp']  # shape (Nvox,)
    Nvp = self.coefficients['Nvp']  # shape (Nvox,)
    p_i = self.coefficients['p_i']  # shape (Nvox, Ng)
    s_i = self.coefficients['s_i']  # shape (Nvox, Ng)
    i = self.coefficients['ivector']  # shape (Ng,)

    div = Ngp * (Ngp - 1)

    # p_i = 0 或 p_j = 0 的项将计算为0，因此不影响总和
    contrast = (numpy.sum(p_i[:, :, None] * p_i[:, None, :] * (i[:, :, None] - i[:, None, :]) ** 2, (1, 2)) *
                numpy.sum(s_i, 1) / Nvp)

    contrast[div != 0] /= div[div != 0]
    contrast[div == 0] = 0

    return contrast

  def getBusynessFeatureValue(self):
    """
    计算并返回忙碌度。

    :math:`Busyness = \frac{\sum^{N_g}_{i = 1}{p_{i}s_{i}}}{\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{|ip_i - jp_j|}}\text{, 其中 }p_i \neq 0, p_j \neq 0`

    忙碌度是从一个像素到其邻居的变化的度量。忙碌度的高值表示“忙碌”的图像，具有体素及其邻域之间的快速强度变化。

    注意 如果 :math:`N_{g,p} = 1`，则 :math:`busyness = \frac{0}{0}`。如果是这种情况，返回0，因为它涉及到一个完全均匀的区域。
    """
    p_i = self.coefficients['p_i']  # shape (Nv, Ngp)
    s_i = self.coefficients['s_i']  # shape (Nv, Ngp)
    i = self.coefficients['ivector']  # shape (Nv, Ngp)
    p_zero = self.coefficients['p_zero']  # shape (2, z)

    i_pi = i * p_i
    absdiff = numpy.abs(i_pi[:, :, None] - i_pi[:, None, :])

    # 从总和中移除 p_i = 0 或 p_j = 0 的项
    absdiff[p_zero[0], :, p_zero[1]] = 0
    absdiff[p_zero[0], p_zero[1], :] = 0

    absdiff = numpy.sum(absdiff, (1, 2))

    busyness = numpy.sum(p_i * s_i, 1)
    busyness[absdiff != 0] = busyness[absdiff != 0] / absdiff[absdiff != 0]
    busyness[absdiff == 0] = 0
    return busyness

  def getComplexityFeatureValue(self):
    """
    计算并返回复杂度。

    :math:`Complexity = \frac{1}{N_{v,p}}\displaystyle\sum^{N_g}_{i = 1}\displaystyle\sum^{N_g}_{j = 1}{|i - j|
    \frac{p_{i}s_{i} + p_{j}s_{j}}{p_i + p_j}}\text{, 其中 }p_i \neq 0, p_j \neq 0`

    当图像中有许多基本组件时，认为图像是复杂的，即图像是非均匀的，并且灰度级强度的变化很快。
    """
    Nvp = self.coefficients['Nvp']  # shape (Nv,)
    p_i = self.coefficients['p_i']  # shape (Nv, Ngp)
    s_i = self.coefficients['s_i']  # shape (Nv, Ngp)
    i = self.coefficients['ivector']  # shape (Nv, Ngp)
    p_zero = self.coefficients['p_zero']  # shape (2, z)

    pi_si = p_i * s_i
    numerator = pi_si[:, :, None] + pi_si[:, None, :]

    # 从总和中移除 p_i = 0 或 p_j = 0 的项
    numerator[p_zero[0], :, p_zero[1]] = 0
    numerator[p_zero[0], p_zero[1], :] = 0

    divisor = p_i[:, :, None] + p_i[:, None, :]
    divisor[divisor == 0] = 1  # 防止除以0错误。（这些索引处的分子也是0）

    complexity = numpy.sum(numpy.abs(i[:, :, None] - i[:, None, :]) * numerator / divisor, (1, 2)) / Nvp

    return complexity

  def getStrengthFeatureValue(self):
    """
    计算并返回强度。

    :math:`Strength = \frac{\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{(p_i + p_j)(i-j)^2}}{\sum^{N_g}_{i = 1}{s_i}}\text{, 其中 }p_i \neq 0, p_j \neq 0`

    强度是图像中基本元素的度量。当基本元素容易定义和可见时，其值高，即图像中强度变化缓慢但在灰度级强度上有更大的粗糙差异。

    注意 :math:`\sum^{N_g}_{i=1}{s_i}` 可能评估为0（在完全均匀的图像中）。如果是这种情况，返回0。
    """
    p_i = self.coefficients['p_i']  # shape (Nv, Ngp)
    s_i = self.coefficients['s_i']  # shape (Nv, Ngp)
    i = self.coefficients['ivector']  # shape (Nv, Ngp)
    p_zero = self.coefficients['p_zero']  # shape (2, z)

    sum_s_i = numpy.sum(s_i, 1)

    strength = (p_i[:, :, None] + p_i[:, None, :]) * (i[:, :, None] - i[:, None, :]) ** 2

    # 从总和中移除 p_i = 0 或 p_j = 0 的项
    strength[p_zero[0], :, p_zero[1]] = 0
    strength[p_zero[0], p_zero[1], :] = 0

    strength = numpy.sum(strength, (1, 2))
    strength[sum_s_i != 0] /= sum_s_i[sum_s_i != 0]
    strength[sum_s_i == 0] = 0

    return strength
