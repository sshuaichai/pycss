import pandas as pd

# 定义CSV文件路径
global_features_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\radiomics_022_features_filter.csv"
voxel_features_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\Voxel_feature_R3B12.csv"
output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_features.csv"

# 读取CSV文件
global_features = pd.read_csv(global_features_path)
voxel_features = pd.read_csv(voxel_features_path)

# 为voxel_features中的列（除了'Image'列）添加后缀_Voxel
voxel_features_renamed = voxel_features.rename(columns=lambda x: x if x == 'Image' else x + '_R3B12')

# 合并两个DataFrame，基于'Image'列进行合并
merged_features = pd.merge(global_features, voxel_features_renamed, on='Image', how='inner')

# 保存合并后的DataFrame到新的CSV文件
merged_features.to_csv(output_path, index=False)

print(f"Merged features with unique columns saved to {output_path}")
