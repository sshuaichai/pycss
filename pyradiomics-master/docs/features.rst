# -*- coding: utf-8 -*-

"""
=================
放射学特征
=================

该部分包含使用PyRadiomics可以提取的各种特征的定义。它们分为以下类别：

* :py:class:`一阶统计特征 <radiomics.firstorder.RadiomicsFirstOrder>`（19个特征）
* :py:class:`基于形状（3D） <radiomics.shape.RadiomicsShape>`（16个特征）
* :py:class:`基于形状（2D） <radiomics.shape2D.RadiomicsShape2D>`（10个特征）
* :py:class:`灰度共生矩阵 <radiomics.glcm.RadiomicsGLCM>`（24个特征）
* :py:class:`灰度共生矩阵 <radiomics.glrlm.RadiomicsGLRLM>`（16个特征）
* :py:class:`灰度大小区域矩阵 <radiomics.glszm.RadiomicsGLSZM>`（16个特征）
* :py:class:`邻近灰度差异矩阵 <radiomics.ngtdm.RadiomicsNGTDM>`（5个特征）
* :py:class:`灰度依赖矩阵 <radiomics.gldm.RadiomicsGLDM>`（14个特征）

除了形状之外，所有特征类别都可以在原始图像和/或应用了几种滤波器之一的派生图像上进行计算。
形状描述符与灰度值无关，并从标签掩模中提取。如果启用，它们将与启用的输入图像类型分开计算，并在结果中列出，
就像在原始图像上计算一样。

下面定义的大多数特征符合Imaging Biomarker Standardization Initiative（IBSI）描述的特征定义，该
特征定义可在Zwanenburg等人（2016年）的独立文档中找到[1]_。
如果特征有所不同，则已添加注释以指明差异。

.. _radiomics-firstorder-label:

一阶特征
--------------------

.. automodule:: radiomics.firstorder
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-shape-label:

形状特征（3D）
-------------------

.. automodule:: radiomics.shape
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-shape2D-label:

形状特征（2D）
-------------------

.. automodule:: radiomics.shape2D
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-glcm-label:

灰度共生矩阵（GLCM）特征
-----------------------------------------------

.. automodule:: radiomics.glcm
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-glszm-label:

灰度大小区域矩阵（GLSZM）特征
--------------------------------------------

.. automodule:: radiomics.glszm
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-glrlm-label:

灰度级别运行长度矩阵（GLRLM）特征
---------------------------------------------

.. automodule:: radiomics.glrlm
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-ngtdm-label:

邻近灰度差异矩阵（NGTDM）特征
---------------------------------------------------------

.. automodule:: radiomics.ngtdm
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. _radiomics-gldm-label:

灰度依赖矩阵（GLDM）特征
--------------------------------------------

.. automodule:: radiomics.gldm
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. [1] Zwanenburg, A., Leger, S., Vallières, M., and Löck, S. (2016). Image biomarker
    standardisation initiative - feature definitions. In eprint arXiv:1612.07003 [cs.CV]
"""
