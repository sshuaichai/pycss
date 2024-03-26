.. _radiomics-labs-label:

================
pyradiomics 实验室
================

pyradiomics 实验室是存储在代码库中的一组探索性/实验性功能，但不是核心功能的一部分。我们欢迎用户对这些功能提供反馈。这些脚本和功能可能会在将来发生变化。

pyradiomics-dcm
---------------

'''
该文档介绍了 pyradiomics 实验室中的一个脚本 pyradiomics-dcm 的功能和用法，该脚本用于支持将 DICOM（医学图像存储和通信标准）数据与
pyradiomics（用于医学图像分析的 Python 库）一起使用。
该脚本的主要目的是将 DICOM 图像和 DICOM 分割数据转换为适合 pyradiomics 的格式，并提供用于特征提取的功能。
文档提供了该脚本的用法示例、先决条件、样例调用以及如何提出问题的指南，并引用了相关的参考资料。
使用者需要注意，该脚本属于 "pyradiomics 实验室"，是一个实验性功能，可能会在未来发生变化
'''

关于
#####

这是一个实验性脚本，用于支持将 pyradiomics 与 DICOM 数据一起使用。

该脚本将接受一个包含单个 DICOM 图像研究的目录作为输入图像，
以及指向 DICOM 分割图像（DICOM SEG）对象的文件名。

该脚本将透明地将 DICOM 图像转换为适合 pyradiomics 使用的表示形式，
可以使用 plastimatch 或 dcm2niix。

为什么？
#######

* 医学图像数据通常以 DICOM 格式提供，而 pyradiomics 用户经常寻求在处理 DICOM 数据时的帮助
* 在 TCIA 上有一些公共数据集，其中包含以 DICOM SEG 格式存储的分割数据
* 使用 DICOM 表示形式进行放射学特征提取

  * 引入了应存储以配合特征的属性的标准化形式
  * 允许将计算结果与描述所分析区域解剖学的各种本体论以及特征本身（例如，脚本生成的 SR 文档将利用 IBSI 命名法描述那些在 IBSI 中有对应关系的 pyradiomics 实现的特征）
  * 允许（通过唯一标识符）引用用于特征计算的 DICOM 图像序列和 DICOM 分割
  * 实现了对图像、分割和特征的数据的统一表示（即，所有数据类型可以使用相同的数据管理系统）
  * 不会阻止将结果用于不了解 DICOM 的软件工具 - dcmqi 可以用于将 DICOM 分割和具有测量的 DICOM SR 转换为非 DICOM 表示形式（分割的 ITK 可读图像格式，测量的 JSON）; 另外还提供了一个工具用于生成存储在这些 SR 中的 DICOM 属性和测量的制表符表示形式：
    https://github.com/QIICR/dcm2tables

先决条件
##########

* `plastimatch <http://plastimatch.org/plastimatch.html>`_ 或 `dcm2niix <https://github.com/rordenlab/dcm2niix>`_ 用于图像体积重建
* dcmqi [1]_ [2]_（从 `fedb41 <https://github.com/QIICR/dcmqi/commit/3638930723bf1a239515409c1f9ec886a9fedb41>`_ 或更新版本构建）用于读取 DICOM SEG 并将其转换为适合 pyradiomics 的表示形式，并将结果特征存储为 DICOM 结构化报告，实例化 SR TID 1500
* 在使用此脚本之前，您可能希望对 DICOM 数据进行排序，以便将单个系列存储在单独的目录中。您可能会发现此工具对此目的很有用：https://github.com/pieper/dicomsort
* 如果分割不是以 DICOM SEG 格式存储的，可以使用 dcmqi 生成这些分割的标准表示形式：https://github.com/QIICR/dcmqi

用法
#####

从命令行示例用法::

    $ python pyradiomics-dcm.py -h
    usage: pyradiomics-dcm.py --input-image <dir> --input-seg <name> --output-sr <name>

    警告：这是一个“pyradiomics 实验室”脚本，意味着它是一个正在开发中的实验性功能！
    此辅助脚本的目的是直接从/到 DICOM 数据启用 pyradiomics 特征提取。
    定义感兴趣区域的分割必须定义为 DICOM 分割图像。
    将来可能会添加对用于定义感兴趣区域的 DICOM 放射治疗结构集的支持。

    可选参数：
      -h, --help            显示此帮助消息并退出
      --input-image-dir Input DICOM image directory
                        输入 DICOM 系列的目录。预期每个系列对应一个标量体积。
      --input-seg-file Input DICOM SEG file
                        输入分割定义为 DICOM 分割对象。
      --output-dir Directory to store the output file
                        保存结果 DICOM 文件的目录。
      --parameters pyradiomics extraction parameters
      --temp-dir Temporary directory
      --features-dict Dictionary mapping pyradiomics feature names to the IBSI defined features.
      --volume-reconstructor Choose the tool to be used for reconstructing image volume from the DICOM image series. Allowed options are plastimatch or dcm2niix (should be installed on the system). plastimatch will be used by default.

样例调用
###########

::

    $ python pyradiomics-dcm.py --input-image-dir CT --input-seg SEG/1.dcm \
       --output-dir OutputSR --temp-dir TempDir --parameters Pyradiomics_Params.yaml
    dcmqi repository URL: https://github.com/QIICR/dcmqi.git revision: 3638930 tag: latest-4-g3638930
    Row direction: 1 0 0
    Col direction: 0 1 0
    Z direction: 0 0 1
    Total frames: 177
    Total frames with unique IPP: 177
    Total overlapping frames: 0
    Origin: [-227.475, -194.775, -1223]
    dcmqi repository URL: https://github.com/QIICR/dcmqi.git revision: 3638930 tag: latest-4-g3638930
    Total measurement groups: 1
    Adding to compositeContext: 1.dcm
    Composite Context initialized
    SR saved!

    $ dsrdump OutputSR/1.2.276.0.7230010.3.1.4.0.60427.1539113881.935517.dcm
    Enhanced SR Document

    Patient             : interobs05 (#interobs05)
    ENH: include pyradiomics identification and version
    Study               : interobs05_20170910_CT
    Series              : GTV segmentation - Reader AB - pyradiomics features (#1)
    Manufacturer        : QIICR (https://github.com/QIICR/dcmqi.git, #0)
    Completion Flag     : PARTIAL
    Verification Flag   : UNVERIFIED
    Content Date/Time   : 2018-10-09 15:38:01

    <CONTAINER:(,,"Imaging Measurement Report")=SEPARATE>
      <has concept mod CODE:(,,"Language of Content Item and Descendants")=(eng,RFC5646,"English")>
      <has obs context CODE:(,,"Observer Type")=(121007,DCM,"Device")>
      <has obs context UIDREF:(,,"Device Observer UID")="1.3.6.1.4.1.43046.3.1.4.0.60427.1539113880.935515">
      <has obs context TEXT:(,,"Device Observer Name")="pyradiomics">
      <has obs context TEXT:(,,"Device Observer Model Name")="2.1.0.post10.dev0+g51bc87f">
      <has concept mod CODE:(,,"Procedure reported")=(P0-0099A,SRT,"Imaging procedure")>
      <contains CONTAINER:(,,"Image Library")=SEPARATE>
        <contains CONTAINER:(,,"Image Library Group")=SEPARATE>
          <has acq context CODE:(,,"Modality")=(CT,DCM,"Computed Tomography")>
          <has acq context DATE:(,,"Study Date")="20170910">
          <has acq context UIDREF:(,,"Frame of Reference UID")="1.3.6.1.4.1.40744.29.28518703451127075549995420991770873582">

    ...

      <contains CONTAINER:(,,"Imaging Measurements")=SEPARATE>
        <contains CONTAINER:(,,"Measurement Group")=SEPARATE>
          <has obs context TEXT:(,,"Tracking Identifier")="Gross Target Volume">
          <has obs context UIDREF:(,,"Tracking Unique Identifier")="1.3.6.1.4.1.43046.3.1.4.0.60427.1539113881.935516"
    >
          <contains CODE:(,,"Finding")=(C112913,NCIt,"Gross Target Volume")>
          <contains IMAGE:(,,"Referenced Segment")=(SG image,,1)>
          <contains UIDREF:(,,"Source series for segmentation")="1.3.6.1.4.1.40744.29.18397950185694012790332812250603
    612437">
          <has concept mod CODE:(,,"Finding Site")=(T-28000,SRT,"Lung")>
          <contains NUM:(,,"shape_MeshVolume")="7.255467E+04" (1,UCUM,"no units")>
          <contains NUM:(,,"Maximum 3D diameter")="7.491328E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_Maximum2DDiameterSlice")="6.767570E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"Elongation")="7.993260E-01" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_MinorAxisLength")="4.699969E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"Flatness")="6.517569E-01" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_Maximum2DDiameterColumn")="6.746851E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"Surface to volume ratio")="1.572168E-01" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_Maximum2DDiameterRow")="6.072891E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_VoxelVolume")="7.285600E+04" (1,UCUM,"no units")>
          <contains NUM:(,,"Sphericity")="7.375024E-01" (1,UCUM,"no units")>
          <contains NUM:(,,"Surface area")="1.140681E+04" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_MajorAxisLength")="5.879915E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"shape_LeastAxisLength")="3.832275E+01" (1,UCUM,"no units")>
          <contains NUM:(,,"Small zone emphasis")="7.384502E-01" (1,UCUM,"no units")>
          <contains NUM:(,,"glszm_SmallAreaLowGrayLevelEmphasis")="3.381883E-03" (1,UCUM,"no units")>
          <contains NUM:(,,"Normalised grey level non-uniformity")="3.136554E-02" (1,UCUM,"no units")>
          <contains NUM:(,,"glszm_SmallAreaHighGrayLevelEmphasis")="5.478214E+02" (1,UCUM,"no units")>
          <contains NUM:(,,"Large zone emphasis")="3.873234E+03" (1,UCUM,"no units")>

    ...

问题？
#######

请在 `pyradiomics 邮件列表 <https://groups.google.com/forum/#!forum/pyradiomics>`_ 上发布您的反馈和问题。

参考资料
###########

.. [1] Herz C, Fillion-Robin J-C, Onken M, Riesmeier J, Lasso A, Pinter C, Fichtinger G, Pieper S, Clunie D, Kikinis R,
  Fedorov A. dcmqi: An Open Source Library for Standardized Communication of Quantitative Image Analysis Results Using
  DICOM. Cancer Research. 2017;77(21):e87–e90 http://cancerres.aacrjournals.org/content/77/21/e87
.. [2] Fedorov A, Clunie D, Ulrich E, Bauer C, Wahle A, Brown B, Onken M, Riesmeier J, Pieper S, Kikinis R, Buatti J,
  Beichel RR. (2016) DICOM for quantitative imaging biomarker development: a standards based approach to sharing
  clinical data and structured PET/CT analysis results in head and neck cancer research.
  PeerJ 4:e2057 https://doi.org/10.7717/peerj.2057
