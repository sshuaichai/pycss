from __future__ import print_function
import csv
import os

def main():
    # 设置数据根目录路径
    DATA_ROOT_PATH = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_lung"
    # 图像和标签目录
    images_dir = os.path.join(DATA_ROOT_PATH, "imagesTr")
    labels_dir = os.path.join(DATA_ROOT_PATH, "labelsTr")
    # 设置输出目录路径
    OUTPUT_PATH = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output"
    # 输出CSV文件路径
    outputFile = os.path.join(OUTPUT_PATH, "Dataset021_Lung.csv")  # 更改名称

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # 打开输出CSV文件
    with open(outputFile, 'w', newline='') as outFile:
        cw = csv.writer(outFile, lineterminator='\n')
        cw.writerow(['Image', 'Label'])  # 写入CSV头部

        # 遍历图像目录，匹配图像和标签文件
        for image_filename in os.listdir(images_dir):
            if image_filename.endswith(".nii.gz"):
                # 构建图像和标签的完整路径
                image_path = os.path.join(images_dir, image_filename)
                label_path = os.path.join(labels_dir, image_filename)  # 假设图像和标签文件名相同

                # 检查标签文件是否存在
                if os.path.exists(label_path):
                    cw.writerow([image_path, label_path])  # 写入图像和标签路径到CSV
                else:
                    print("Label file not found for", image_filename)

    print("CSV file has been generated at:", outputFile)

if __name__ == '__main__':
    main()
