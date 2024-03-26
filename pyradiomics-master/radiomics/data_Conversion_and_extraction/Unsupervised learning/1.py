# 文件和输出路径
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_set.csv"  # 确保路径正确且为CSV文件
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Unsupervised learning\PART_1"
encoders_folder = os.path.join(output_folder, "encoders")  # 创建encoders子文件夹路径

# 确保输出目录存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(encoders_folder):  # 确保encoders目录也被创建
    os.makedirs(encoders_folder)
运用无监督学习分析提取的特征
