
import numpy
from six.moves import range

from radiomics import base, cMatrices

# 导入所需的库和模块

class RadiomicsGLSZM(base.RadiomicsFeaturesBase):
    """
    一个Gray Level Size Zone (GLSZM)特征计算器，用于量化图像中的灰度级大小区域。
    灰度级大小区域被定义为共享相同灰度级强度的连接体素数量。
    如果根据无穷范数距离（在3D中为26连接区域，2D中为8连接区域）距离为1，则将体素视为连接的。

    在灰度级大小区域矩阵P(i, j)中，(i, j)元素表示在图像中出现的特定灰度级和大小的区域数量。
    与GLCM和GLRLM不同，GLSZM特征是旋转无关的，只需为ROI中的所有方向计算一个矩阵。

    举例来说，如果有一个5x5的图像，包含5个不同的离散灰度级，代码可以计算出每个灰度级和大小区域的数量。

    随后，代码还计算了一些GLSZM特征，如Small Area Emphasis、Large Area Emphasis等，这些特征用于描述图像的纹理和特性。

    参考文献：
    - Guillaume Thibault; Bernard Fertil; Claire Navarro; Sandrine Pereira; Pierre Cau; Nicolas Levy; Jean Sequeira;
      Jean-Luc Mari (2009). "Texture Indexes and Gray Level Size Zone Matrix. Application to Cell Nuclei Classification".
      Pattern Recognition and Information Processing (PRIP): 140-145.
    - <https://en.wikipedia.org/wiki/Gray_level_size_zone_matrix>
    """

    def __init__(self, inputImage, inputMask, **kwargs):
        super(RadiomicsGLSZM, self).__init__(inputImage, inputMask, **kwargs)

        self.P_glszm = None
        self.imageArray = self._applyBinning(self.imageArray)

    # 初始化计算
    def _initCalculation(self, voxelCoordinates=None):
        self.P_glszm = self._calculateMatrix(voxelCoordinates)

        self._calculateCoefficients()

        self.logger.debug('GLSZM特征类已初始化，计算后的GLSZM矩阵形状为%s', self.P_glszm.shape)

    # 计算矩阵
    def _calculateMatrix(self, voxelCoordinates=None):
        """
        计算图像中出现灰度级和体素计数的次数。
        P_glszm(level, voxel_count) = # occurrences
        对于3D图像，这涉及到26个连接区域，对于2D图像，涉及到8个连接区域。
        """
        self.logger.debug('在C中计算GLSZM矩阵')
        Ng = self.coefficients['Ng']
        Ns = numpy.sum(self.maskArray)

        matrix_args = [
            self.imageArray,
            self.maskArray,
            Ng,
            Ns,
            self.settings.get('force2D', False),
            self.settings.get('force2Ddimension', 0)
        ]
        if self.voxelBased:
            matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

        P_glszm = cMatrices.calculate_glszm(*matrix_args)  # 形状为(Nvox, Ng, Ns)

        # 删除指定ROI中不存在的灰度级的行
        NgVector = range(1, Ng + 1)  # 所有可能的灰度值
        GrayLevels = self.coefficients['grayLevels']  # ROI中存在的灰度值
        emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)), dtype=int)  # ROI中不存在的灰度值

        P_glszm = numpy.delete(P_glszm, emptyGrayLevels - 1, 1)

        return P_glszm

    # 计算系数
    def _calculateCoefficients(self):
        self.logger.debug('计算GLSZM系数')

        ps = numpy.sum(self.P_glszm, 1)  # 形状为(Nvox, Ns)
        pg = numpy.sum(self.P_glszm, 2)  # 形状为(Nvox, Ng)

        ivector = self.coefficients['grayLevels'].astype(float)  # 形状为(Ng,)
        jvector = numpy.arange(1, self.P_glszm.shape[2] + 1, dtype=numpy.float64)  # 形状为(Ns,)

        # 获取此GLSZM中的区域数
        Nz = numpy.sum(self.P_glszm, (1, 2))  # 形状为(Nvox,)
        Nz[Nz == 0] = 1  # 如果总和为0，将其设置为numpy.spacing(1)

        # 获取由此GLSZM表示的体素数：将区域乘以其大小并求和
        Np = numpy.sum(ps * jvector[None, :], 1)  # 形状为(Nvox, )
        Np[Np == 0] = 1

        # 删除指定ROI中不存在的区域大小的列
        emptyZoneSizes = numpy.where(numpy.sum(ps, 0) == 0)
        self.P_glszm = numpy.delete(self.P_glszm, emptyZoneSizes, 2)
        jvector = numpy.delete(jvector, emptyZoneSizes)
        ps = numpy.delete(ps, emptyZoneSizes, 1)

        self.coefficients['Np'] = Np
        self.coefficients['Nz'] = Nz
        self.coefficients['ps'] = ps
        self.coefficients['pg'] = pg
        self.coefficients['ivector'] = ivector
        self.coefficients['jvector'] = jvector

    # 获取Small Area Emphasis特征值
    def getSmallAreaEmphasisFeatureValue(self):
        """
        1. Small Area Emphasis (SAE)

        SAE度量了小尺寸区域的分布，具有更大的值表示更多的小尺寸区域和更细的纹理。
        """
        ps = self.coefficients['ps']
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']

        sae = numpy.sum(ps / (jvector[None, :] ** 2), 1) / Nz
        return sae

    # 获取Large Area Emphasis特征值
    def getLargeAreaEmphasisFeatureValue(self):
        """
        2. Large Area Emphasis (LAE)

        LAE度量了大尺寸区域的分布，具有更大的值表示更多的大尺寸区域和更粗的纹理。
        """
        ps = self.coefficients['ps']
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']

        lae = numpy.sum(ps * (jvector[None, :] ** 2), 1) / Nz
        return lae

    # 获取Gray Level Non-Uniformity特征值
    def getGrayLevelNonUniformityFeatureValue(self):
        """
        3. Gray Level Non-Uniformity (GLN)

        GLN度量了图像中灰度级强度的变异性，较低的值表示灰度级强度更均匀。
        """
        pg = self.coefficients['pg']
        Nz = self.coefficients['Nz']

        iv = numpy.sum(pg ** 2, 1) / Nz
        return iv

    # 获取Gray Level Non-Uniformity Normalized特征值
    def getGrayLevelNonUniformityNormalizedFeatureValue(self):
        """
        4. Gray Level Non-Uniformity Normalized (GLNN)

        GLNN度量了图像中灰度级强度的变异性，较低的值表示灰度级强度更相似。
        这是GLN公式的归一化版本。
        """
        pg = self.coefficients['pg']
        Nz = self.coefficients['Nz']

        ivn = numpy.sum(pg ** 2, 1) / Nz ** 2
        return ivn

    # 获取Size-Zone Non-Uniformity特征值
    def getSizeZoneNonUniformityFeatureValue(self):
        """
        5. Size-Zone Non-Uniformity (SZN)

        SZN度量了图像中尺寸区域体积的变异性，较低的值表示尺寸区域体积更均匀。
        """
        ps = self.coefficients['ps']
        Nz = self.coefficients['Nz']

        szv = numpy.sum(ps ** 2, 1) / Nz
        return szv

    # 获取Size-Zone Non-Uniformity Normalized特征值
    def getSizeZoneNonUniformityNormalizedFeatureValue(self):
        """
        6. Size-Zone Non-Uniformity Normalized (SZNN)

        SZNN度量了图像中尺寸区域体积的变异性，较低的值表示尺寸区域体积更相似。
        这是SZN公式的归一化版本。
        """
        ps = self.coefficients['ps']
        Nz = self.coefficients['Nz']

        szvn = numpy.sum(ps ** 2, 1) / Nz ** 2
        return szvn

    # 获取Zone Percentage特征值
    def getZonePercentageFeatureValue(self):
        """
        7. Zone Percentage (ZP)

        ZP通过计算区域数量与ROI中的体素数量的比率来度量纹理的粗糙度。
        值在范围1/Np <= ZP <= 1之间，较高的值表示ROI中包含更多的小区域（表示更细的纹理）。
        """
        Nz = self.coefficients['Nz']
        Np = self.coefficients['Np']

        zp = Nz / Np
        return zp

    # 获取Gray Level Variance特征值
    def getGrayLevelVarianceFeatureValue(self):
        """
        8. Gray Level Variance (GLV)

        GLV度量了区域中灰度级强度的方差。
        """
        ivector = self.coefficients['ivector']
        Nz = self.coefficients['Nz']
        pg = self.coefficients['pg'] / Nz[:, None]  # 除以Nz以获得归一化矩阵

        u_i = numpy.sum(pg * ivector[None, :], 1, keepdims=True)
        glv = numpy.sum(pg * (ivector[None, :] - u_i) ** 2, 1)
        return glv

    # 获取Zone Variance特征值
    def getZoneVarianceFeatureValue(self):
        """
        9. Zone Variance (ZV)

        ZV度量了区域中尺寸区域体积的方差。
        """
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']
        ps = self.coefficients['ps'] / Nz[:, None]  # 除以Nz以获得归一化矩阵

        u_j = numpy.sum(ps * jvector[None, :], 1, keepdims=True)
        zv = numpy.sum(ps * (jvector[None, :] - u_j) ** 2, 1)
        return zv

    # 获取Zone Entropy特征值
    def getZoneEntropyFeatureValue(self):
        """
        10. Zone Entropy (ZE)

        ZE度量了区域尺寸和灰度级强度分布的不确定性/随机性。
        较高的值表示纹理模式更不均匀。
        """
        eps = numpy.spacing(1)
        Nz = self.coefficients['Nz']
        p_glszm = self.P_glszm / Nz[:, None, None]  # 除以Nz以获得归一化矩阵

        ze = -numpy.sum(p_glszm * numpy.log2(p_glszm + eps), (1, 2))
        return ze

    # 获取Low Gray Level Zone Emphasis特征值
    def getLowGrayLevelZoneEmphasisFeatureValue(self):
        """
        11. Low Gray Level Zone Emphasis (LGLZE)

        LGLZE度量了低灰度级大小区域的分布，具有更高的值表示图像中有更多的低灰度级值和大小区域。
        """
        pg = self.coefficients['pg']
        ivector = self.coefficients['ivector']
        Nz = self.coefficients['Nz']

        lie = numpy.sum(pg / (ivector[None, :] ** 2), 1) / Nz
        return lie

    # 获取High Gray Level Zone Emphasis特征值
    def getHighGrayLevelZoneEmphasisFeatureValue(self):
        """
        12. High Gray Level Zone Emphasis (HGLZE)

        HGLZE度量了高灰度级值的分布，具有更高的值表示图像中有更多的高灰度级值和大小区域。
        """
        pg = self.coefficients['pg']
        ivector = self.coefficients['ivector']
        Nz = self.coefficients['Nz']

        hie = numpy.sum(pg * (ivector[None, :] ** 2), 1) / Nz
        return hie

    # 获取Small Area Low Gray Level Emphasis特征值
    def getSmallAreaLowGrayLevelEmphasisFeatureValue(self):
        """
        13. Small Area Low Gray Level Emphasis (SALGLE)

        SALGLE度量了图像中较小尺寸区域和较低灰度级值的联合分布。
        """
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']

        lisae = numpy.sum(self.P_glszm / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)), (1, 2)) / Nz
        return lisae

    # 获取Small Area High Gray Level Emphasis特征值
    def getSmallAreaHighGrayLevelEmphasisFeatureValue(self):
        """
        14. Small Area High Gray Level Emphasis (SAHGLE)

        SAHGLE度量了图像中较小尺寸区域和较高灰度级值的联合分布。
        """
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']

        hisae = numpy.sum(self.P_glszm * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2), (1, 2)) / Nz
        return hisae

    # 获取Large Area Low Gray Level Emphasis特征值
    def getLargeAreaLowGrayLevelEmphasisFeatureValue(self):
        """
        15. Large Area Low Gray Level Emphasis (LALGLE)

        LALGLE度量了图像中较大尺寸区域和较低灰度级值的联合分布。
        """
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']

        lilae = numpy.sum(self.P_glszm * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2), (1, 2)) / Nz
        return lilae

    # 获取Large Area High Gray Level Emphasis特征值
    def getLargeAreaHighGrayLevelEmphasisFeatureValue(self):
        """
        16. Large Area High Gray Level Emphasis (LAHGLE)

        LAHGLE度量了图像中较大尺寸区域和较高灰度级值的联合分布。
        """
        ivector = self.coefficients['ivector']
        jvector = self.coefficients['jvector']
        Nz = self.coefficients['Nz']

        hilae = numpy.sum(self.P_glszm * (ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2), (1, 2)) / Nz
        return hilae
