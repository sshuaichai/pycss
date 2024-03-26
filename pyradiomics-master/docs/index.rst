.. pyradiomics 文档主文件，由sphinx-quickstart 在 2016 年 6 月 28 日 13:27:28 创建。
   您可以完全根据自己的喜好调整此文件，但它至少应包含根 `toctree` 指令。

欢迎来到 pyradiomics 文档！
=============================

这是一个用于从医学图像中提取放射学特征的开源 Python 包。
通过这个包，我们旨在建立放射学分析的参考标准，并提供一个经过测试和维护的开源平台，用于轻松和可重现的放射学特征提取。
通过这样做，我们希望增加对放射学能力的认识并扩大社区。该平台支持 2D 和 3D 特征提取，并可用于计算感兴趣区域的每个特征的单个值（“基于区段”）或生成特征图（“基于体素”）。

**如果您发表任何使用此软件包的工作，请引用以下出版物：**
*van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H.,
Fillon-Robin, J. C., Pieper, S.,  Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic
Phenotype. Cancer Research, 77(21), e104–e107. `https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>`_*

.. 注意::

   该工作部分得到了美国国家癌症研究所授予的资助，
   5U24CA194354，QUANTITATIVE RADIOMICS SYSTEM DECODING THE TUMOR PHENOTYPE。

.. 警告::

   不适用于临床使用。

加入社区！
-------------------
加入 PyRadiomics 社区，点击 `这里 <https://groups.google.com/forum/#!forum/pyradiomics>`_ 加入 google groups。

目录
-----------------

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 2

   installation
   usage
   customization
   radiomics
   features
   removedfeatures
   contributing
   developers
   labs
   FAQs <faq>
   changes

特征类别
---------------

目前支持以下特征类别：

* :py:class:`一阶统计特征 <radiomics.firstorder.RadiomicsFirstOrder>`
* :py:class:`基于形状（3D） <radiomics.shape.RadiomicsShape>`
* :py:class:`基于形状（2D） <radiomics.shape2D.RadiomicsShape2D>`
* :py:class:`灰度共生矩阵 <radiomics.glcm.RadiomicsGLCM>` (GLCM)
* :py:class:`灰度级别运行长度矩阵 <radiomics.glrlm.RadiomicsGLRLM>` (GLRLM)
* :py:class:`灰度大小区域矩阵 <radiomics.glszm.RadiomicsGLSZM>` (GLSZM)
* :py:class:`邻近灰度差异矩阵 <radiomics.ngtdm.RadiomicsNGTDM>` (NGTDM)
* :py:class:`灰度依赖矩阵 <radiomics.gldm.RadiomicsGLDM>` (GLDM)

平均而言，Pyradiomics 从每个图像提取 :math:`\approx 1500` 个特征，其中包括 16 个形状描述符和从原始和派生图像提取的特征（具有 5 个 sigma 级别的 LoG，1 个 Wavelet 分解级别产生 8 个派生图像，以及使用 Square、Square Root、Logarithm 和 Exponential 过滤器派生的图像）。

有关特征类别和单个特征的详细说明，请参阅 :ref:`radiomics-features-label` 部分。

过滤器类别
--------------

除了特征类别之外，还有一些内置的可选过滤器：

* :py:func:`高斯拉普拉斯 <radiomics.imageoperations.getLoGImage>` (LoG, 基于 SimpleITK 功能)
* :py:func:`小波 <radiomics.imageoperations.getWaveletImage>` (使用 PyWavelets 包)
* :py:func:`平方 <radiomics.imageoperations.getSquareImage>`
* :py:func:`平方根 <radiomics.imageoperations.getSquareRootImage>`
* :py:func:`对数 <radiomics.imageoperations.getLogarithmImage>`
* :py:func:`指数 <radiomics.imageoperations.getExponentialImage>`
* :py:func:`梯度 <radiomics.imageoperations.getGradientImage>`
* :py:func:`局部二值模式 (2D) <radiomics.imageoperations.getLBP2DImage>`
* :py:func:`局部二值模式 (3D) <radiomics.imageoperations.getLBP3DImage>`

更多信息，请参阅 :ref:`radiomics-imageoperations-label`。

支持可重复提取
----------------------------------

除了计算特征之外，pyradiomics 包还在输出中包含了额外的信息。
这些信息包含了有关使用的图像和掩码、以及应用的设置和过滤器的信息，从而使特征提取完全可重现。更多信息，请参阅 :ref:`radiomics-generalinfo-label`。

Pyradiomics 使用的第三方包
----------------------------------

* SimpleITK（图像加载和预处理）
* numpy（特征计算）
* PyWavelets（小波过滤器）
* pykwalify（启用 yaml 参数文件检查）
* six（Python 3 兼容性）

也请查看 `requirements 文件 <https://github.com/Radiomics/pyradiomics/blob/master/requirements.txt>`_。

安装
------------

PyRadiomics 不受操作系统限制，兼容 Python >=3.5。预先构建的二进制文件可在 PyPi 和 Conda 上获得。
要安装 PyRadiomics，请确保已安装 Python 并运行：

*  ``python -m pip install pyradiomics``

有关更详细的安装说明和从源代码构建，请参阅 :ref:`radiomics-installation-label` 部分。

Pyradiomics 索引和表
------------------------------

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`

许可
-------

本软件包由开源 `3-clause BSD License <https://github.com/Radiomics/pyradiomics/blob/master/LICENSE.txt>`_ 覆盖。

开发人员
----------

 - `Joost van Griethuysen <https://github.com/JoostJM>`_:sup:`1,3,4`
 - `Andriy Fedorov <https://github.com/fedorov>`_:sup:`2`
 - `Nicole Aucoin <https://github.com/naucoin>`_:sup:`2`
 - `Jean-Christophe Fillion-Robin <https://github.com/jcfr>`_:sup:`5`
 - `Ahmed Hosny <https://github.com/ahmedhosny>`_:sup:`1`
 - `Steve Pieper <https://github.com/pieper>`_:sup:`6`
 - `Hugo Aerts (PI) <https://github.com/hugoaerts>`_:sup:`1,2`

:sup:`1`\ Department of Radiation Oncology, Dana-Farber Cancer Institute, Brigham and Women's Hospital, Harvard Medical School, Boston, MA,
:sup:`2`\ Department of Radiology, Brigham and Women's Hospital, Harvard Medical School, Boston, MA
:sup:`3`\ Department of Radiology, Netherlands Cancer Institute, Amsterdam, The Netherlands,
:sup:`4`\ GROW-School for Oncology and Developmental Biology, Maastricht University Medical Center, Maastricht, The Netherlands,
:sup:`5`\ Kitware,
:sup:`6`\ Isomics

联系
-------
我们很乐意帮助您解决任何问题。请在 `3D Slicer 论坛的放射学社区部分 <https://discourse.slicer.org/c/community/radiomics/23>`_ 上与我们联系。

我们欢迎您对 PyRadiomics 的贡献。请阅读 :ref:`contributing guidelines <radiomics-contributing-label>` 以了解如何为 PyRadiomics 做出贡献。
有关添加 / 自定义特征类别和过滤器的信息，请参阅 :ref:`radiomics-developers` 部分。
