import pandas as pd
import pingouin as pg
import yaml
from radiomics import featureextractor
import os

# 配置PyRadiomics特征提取器
params = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\CT_lung_filter.yaml"
# 以UTF-8编码读取YAML文件
with open(params, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)
# 使用读取的参数初始化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# 特征提取函数
def extract_features(image_path, label_path, extractor):
    return extractor.execute(image_path, label_path)

# 初始化特征字典
features = []

# 文件夹路径
image_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset022_lung\imagesTr"
label_folder_read1 = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset022_lung\labelsTr"
label_folder_read2 = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset023_lung\labelsTr"

# 遍历图像进行特征提取
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    label_path_read1 = os.path.join(label_folder_read1, image_name.replace('image', 'label'))
    label_path_read2 = os.path.join(label_folder_read2, image_name.replace('image', 'label'))

    features_read1 = extract_features(image_path, label_path_read1, extractor)
    features_read2 = extract_features(image_path, label_path_read2, extractor)

    for feature_name, value in features_read1.items():
        features.append({'ImageName': image_name, 'Feature': feature_name, 'Value': value, 'Rater': 'read1'})
    for feature_name, value in features_read2.items():
        features.append({'ImageName': image_name, 'Feature': feature_name, 'Value': value, 'Rater': 'read2'})

# 转换特征数据为DataFrame
df_features = pd.DataFrame(features)

# 计算ICC
icc_results = pg.intraclass_corr(data=df_features, targets='ImageName', raters='Rater', ratings='Value', nan_policy='omit')

# 筛选ICC大于0.8的特征
selected_features = icc_results[icc_results['ICC'] > 0.8]['Feature'].unique()

# 输出保留的特征
print("Selected features with ICC > 0.8:", selected_features)

# 保存选定的特征到指定目录
output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\selected_features_with_high_ICC.csv"
selected_features_df = pd.DataFrame(selected_features, columns=['Selected Features'])
selected_features_df.to_csv(output_path, index=False)
print(f"Selected features saved to {output_path}")
