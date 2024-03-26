from __future__ import print_function
import csv
import os

# 初始化序列字典和标签字典
SqDic = {
  'T2-TSE-TRA': 't2tra',
  'T2-TRA': 't2tra',
  'T2-TSE-SAG': 't2sag',
  'T2-SAG': 't2sag',
  'T2-TSE-COR': 't2cor',
  'T2-COR': 't2cor',
  'B1000': 'dwi',
  'B1100': 'dwi',
  'ADC': 'adc'
}

LabelDic = {
  'Reader-1': '2 Semi-Auto_Inexp-3',
  'Reader-2': '2 Semi-Auto_Inexp-4',
  'Reader-3': '3 Semi-Auto_Exp-1',
  'Reader-4': '3 Semi-Auto_Exp-2',
  'Reader-5': '4 Manual-1',
  'Reader-6': '4 Manual-2',
  'Reader-7': '1 Auto-1',
  'Reader-8': '3 Semi-Auto_Exp-1',
  'Reader-9': '2 Semi-Auto_Inexp-1',
  'Reader-10': '2 Semi-Auto_Inexp-2',
  'Reader-11': '3 Semi-Auto_Exp-2'
}


def main():
  DATA_ROOT_PATH = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Task02_Heart"
  OUTPUT_PATH = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output"
  images_dir = os.path.join(DATA_ROOT_PATH, "imagesTr")
  labels_dir = os.path.join(DATA_ROOT_PATH, "labelsTr")
  outputFile = os.path.join(OUTPUT_PATH, "Task02_Heart_MRI.csv")

  if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

  with open(outputFile, 'w', newline='') as outFile:
    cw = csv.writer(outFile, lineterminator='\n')
    cw.writerow(['Image', 'Label', 'Sequence', 'Reader'])  # 添加序列和阅读者信息列

    for image_filename in os.listdir(images_dir):
      if image_filename.endswith(".nii.gz"):
        image_path = os.path.join(images_dir, image_filename)
        label_path = os.path.join(labels_dir, image_filename)

        # 从文件名中提取序列和阅读者信息
        sequence = SqDic.get(image_filename.split('_')[0], 'Unknown')
        reader = LabelDic.get(image_filename.split('_')[1], 'Unknown')

        if os.path.exists(label_path):
          cw.writerow([image_path, label_path, sequence, reader])
        else:
          print("Label file not found for", image_filename)

  print("CSV file has been generated at:", outputFile)


if __name__ == '__main__':
  main()


'''
这段代码整合了两个脚本的功能：它遍历指定的图像和标签目录，为每个图像文件生成对应的标签文件路径，
并且尝试从文件名中提取序列和阅读者信息，然后将这些信息一起写入到CSV文件中。
请注意，这里假设图像文件名包含了足够的信息来识别序列和阅读者，
这可能需要您根据实际情况调整SqDic和LabelDic的映射逻辑。
'''
