from __future__ import print_function  # 导入未来的打印函数特性

import csv
import os

from DatasetHierarchyReader import DatasetHierarchyReader  # 导入DatasetHierarchyReader模块

'''
这段代码是一个用于生成输入CSV文件的脚本，其功能包括：
设置数据根目录路径、输入目录路径、输出文件路径以及文件类型。
创建关键字设置字典，其中包含图像关键字、图像排除关键字、掩模关键字和掩模排除关键字。
该脚本的目的是从给定的数据集目录中提取图像和掩模的文件路径，并将其写入CSV文件中，以便后续的数据处理和分析。
'''

def main():
  DATA_ROOT_PATH = r'R:\TEMP'  # 数据根目录路径
  inputDirectory = os.path.join(DATA_ROOT_PATH, 'TEST')  # 输入目录路径，使用os.path.join确保路径正确
  outputFile = os.path.join(DATA_ROOT_PATH, 'FileListNEW.csv')  # 输出文件路径
  filetype = '.nrrd'  # 文件类型

  keywordSettings = {  # 关键字设置字典
    'image': '',  # 图像关键字
    'imageExclusion': 'label',  # 图像排除关键字
    'mask': 'label',  # 掩模关键字
    'maskExclusion': ''  # 掩模排除关键字
  }

  print("Scanning files...")

  datasetReader = DatasetHierarchyReader(inputDirectory, filetype=filetype, keywordSettings=keywordSettings)
  datasetHierarchyDict = datasetReader.ReadDatasetHierarchy()

  print("Found %s patients, writing csv" % len(datasetHierarchyDict.keys()))

  try:
    with open(outputFile, 'w', newline='') as outFile:  # 在Python 3中使用'w'模式和newline=''参数
      cw = csv.writer(outFile)
      cw.writerow(['Patient', 'StudyDate', 'Image', 'Mask'])

      for patientIndex, patientDirectory in enumerate(datasetHierarchyDict):
        patientID = os.path.basename(patientDirectory)

        for studyDirectory in datasetHierarchyDict[patientDirectory]:
          studyDate = os.path.basename(studyDirectory)
          imageFilepaths = datasetHierarchyDict[patientDirectory][studyDirectory]["reconstructions"]
          maskFilepaths = datasetHierarchyDict[patientDirectory][studyDirectory]["segmentations"]

          # 假设findImageAndLabelPair是DatasetHierarchyReader类的一个方法，且其实现是正确的
          imageFilepath, maskFilepath = datasetReader.findImageAndLabelPair(imageFilepaths, maskFilepaths)

          if imageFilepath and maskFilepath:
            cw.writerow([patientID, studyDate, imageFilepath, maskFilepath])
  except Exception as exc:
    print(exc)


if __name__ == '__main__':
  main()
