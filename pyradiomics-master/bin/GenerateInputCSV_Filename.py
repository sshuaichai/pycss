# 从未来版本导入print函数特性，确保在Python 2.x和Python 3.x中print具有一致的功能
from __future__ import print_function

# 导入csv模块处理CSV文件，os模块处理文件路径和目录遍历
import csv
import os

# 导入six库以确保代码在Python 2和3中的兼容性
import six

''''
用途说明
这段代码主要用于处理医学影像数据，特别是MRI序列。
它通过扫描指定目录下的文件，自动识别和分类病人的MRI影像文件和对应的标记文件（ROI），然后将这些信息汇总到一个CSV文件中。
这段代码对于医学影像研究人员和数据科学家来说非常有用，特别是在处理大规模MRI数据集、进行数据预处理和准备机器学习训练数据集时
'''

# 初始化序列字典，用于将MRI序列的不同命名映射到统一的标识符
SqDic = {}
SqDic['T2-TSE-TRA'] = 't2tra'
SqDic['T2-TRA'] = 't2tra'
SqDic['T2-TSE-SAG'] = 't2sag'
SqDic['T2-SAG'] = 't2sag'
SqDic['T2-TSE-COR'] = 't2cor'
SqDic['T2-COR'] = 't2cor'
SqDic['B1000'] = 'dwi'
SqDic['B1100'] = 'dwi'
SqDic['ADC'] = 'adc'

# 初始化标签字典，用于将阅读者信息映射到特定的标签
LabelDic = {}
LabelDic['Reader-1'] = '2 Semi-Auto_Inexp-3'
LabelDic['Reader-2'] = '2 Semi-Auto_Inexp-4'
LabelDic['Reader-3'] = '3 Semi-Auto_Exp-1'
LabelDic['Reader-4'] = '3 Semi-Auto_Exp-2'
LabelDic['Reader-5'] = '4 Manual-1'
LabelDic['Reader-6'] = '4 Manual-2'
LabelDic['Reader-7'] = '1 Auto-1'
LabelDic['Reader-8'] = '3 Semi-Auto_Exp-1'
LabelDic['Reader-9'] = '2 Semi-Auto_Inexp-1'
LabelDic['Reader-10'] = '2 Semi-Auto_Inexp-2'
LabelDic['Reader-11'] = '3 Semi-Auto_Exp-2'

# 主函数
def main():
  # 数据根目录路径
  DATA_ROOT_PATH = r"T:/Research/07. Current Projects/2. Robust Radiomics/1. Slicer Dataset COMPLEET/"
  inputDirectory = DATA_ROOT_PATH + r"/Included"  # 输入目录
  outputFile = DATA_ROOT_PATH + r"/Included/FileList.csv"  # 输出文件
  filetype = ".nrrd"  # 文件类型

  print("Scanning files...")  # 打印扫描文件的信息

  # 调用scanpatients函数扫描病人信息，返回数据集层次结构字典
  datasetHierarchyDict = scanpatients(inputDirectory, filetype)

  print("Found %s patients, writing csv" % (len(datasetHierarchyDict.keys())))  # 打印找到的病人数量和写入CSV的信息

  try:
    with open(outputFile, 'wb') as outFile:
      cw = csv.writer(outFile, lineterminator='\n')
      cw.writerow(['Patient', 'Sequence', 'Reader', 'Image', 'Mask'])  # 写入CSV头部

      # 遍历数据集层次结构字典，写入病人信息到CSV文件
      for patient, Studies in sorted(six.iteritems(datasetHierarchyDict), key=lambda t: t[0]):
        for Study, im_fileList in sorted(six.iteritems(Studies['reconstructions']), key=lambda t: t[0]):
          for i_idx, im_file in enumerate(im_fileList):

            if Studies['segmentations'].has_key(Study):
              for Reader, seg_fileList in sorted(six.iteritems(Studies['segmentations'][Study]), key=lambda t: t[0]):
                for s_idx, seg_file in enumerate(sorted(seg_fileList)):

                  i_name = Study
                  if i_idx > 0: i_name += " (%s)" % (str(i_idx + 1))

                  s_name = Reader
                  if s_idx > 0: s_name += " (%s)" % (str(s_idx + 1))

                  cw.writerow([patient, i_name, s_name, im_file, seg_file])
  except Exception as exc:
    print(exc)  # 打印异常信息

# 扫描病人函数，用于遍历指定目录下的文件，根据文件名提取病人信息和影像信息
def scanpatients(f, filetype):
  outputDict = {}

  for dirpath, dirnames, filenames in os.walk(f):
    # 遍历所有文件名，检查病人编号，检查是否为ROI，检查序列
    for fname in filenames:
      if (fname[0:3] == "Pt-") & (fname.endswith(filetype)):
        PtNo = fname[3:7]

        if not outputDict.has_key(PtNo):
          outputDict[PtNo] = {'reconstructions': {}}
          outputDict[PtNo]['segmentations'] = {}

        for SqKey, SqVal in six.iteritems(SqDic):
          if ("ROI_" + SqVal) in fname:
            for ReaderKey, ReaderVal in six.iteritems(LabelDic):
              if (ReaderKey + '_') in fname:
                if not outputDict[PtNo]['segmentations'].has_key(SqVal):
                  outputDict[PtNo]['segmentations'][SqVal] = {}
                if not outputDict[PtNo]['segmentations'][SqVal].has_key(ReaderVal):
                  outputDict[PtNo]['segmentations'][SqVal][ReaderVal] = set()
                outputDict[PtNo]['segmentations'][SqVal][ReaderVal].add(os.path.join(dirpath, fname))
                break
          elif SqKey in fname:
            if not outputDict[PtNo]['reconstructions'].has_key(SqVal):
              outputDict[PtNo]['reconstructions'][SqVal] = set()
            outputDict[PtNo]['reconstructions'][SqVal].add(os.path.join(dirpath, fname))
            break
  return outputDict  # 返回输出字典

# 当脚本作为主程序运行时，调用main函数
if __name__ == '__main__':
  main()


