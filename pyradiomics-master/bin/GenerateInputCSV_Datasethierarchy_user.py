from __future__ import print_function  # 导入未来的打印函数特性

import csv
import os

# 假设 DatasetHierarchyReader 类已经正确定义在 DatasetHierarchyReader.py 文件中
from DatasetHierarchyReader import DatasetHierarchyReader
'文件夹用GenerateInputCSV_Filename_user.py，不用这个'
def main():
    DATA_ROOT_PATH = r""  # 数据根目录路径更新为您的数据集路径
    inputDirectory = DATA_ROOT_PATH  # 输入目录路径，直接使用数据根目录
    outputFile = r'D:\zhuomian\pyradiomics\pyradiomics-master\bin\output\FileListNEW.csv'  # 输出文件路径更新为您希望保存结果的路径
    filetype = '.nrrd'  # '.nii.gz' # 文件类型

    keywordSettings = {}  # 关键字设置字典
    keywordSettings['image'] = '_iamge'  # 图像关键字
    keywordSettings['imageExclusion'] = ''  # 图像排除关键字
    keywordSettings['mask'] = '_label'  # 掩模关键字
    keywordSettings['maskExclusion'] = ''  # 掩模排除关键字

    print("Scanning files...")

    datasetReader = DatasetHierarchyReader(inputDirectory, filetype=filetype)
    datasetHierarchyDict = datasetReader.ReadDatasetHierarchy()

    print("Found %s patients, writing csv" % (str(len(datasetHierarchyDict.keys()))))

    try:
        with open(outputFile, 'w', newline='') as outFile:  # 对于Python 3，使用'w'模式并添加newline=''参数
            cw = csv.writer(outFile, lineterminator='\n')
            cw.writerow(['Patient', 'StudyDate', 'Image', 'Mask'])

            for patientIndex, patientDirectory in enumerate(datasetHierarchyDict):
                patientID = os.path.basename(patientDirectory)

                for studyDirectory in datasetHierarchyDict[patientDirectory]:
                    studyDate = os.path.basename(studyDirectory)

                    imageFilepaths = datasetHierarchyDict[patientDirectory][studyDirectory]["reconstructions"]
                    maskFilepaths = datasetHierarchyDict[patientDirectory][studyDirectory]["segmentations"]

                    imageFilepath, maskFilepath = datasetReader.findImageAndLabelPair(imageFilepaths, maskFilepaths, keywordSettings)

                    if (imageFilepath is not None) and (maskFilepath is not None):
                        cw.writerow([patientID, studyDate, imageFilepath, maskFilepath])

    except Exception as exc:
        print(exc)

if __name__ == '__main__':
    main()
