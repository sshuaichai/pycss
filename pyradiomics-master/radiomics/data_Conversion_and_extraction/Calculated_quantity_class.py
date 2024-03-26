import pandas as pd

# 提取的影像组学特征文件路径
features_file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_features.csv"
# 输出CSV文件路径
output_csv_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\calculated\final_R3B12_ClassCounts.csv"#过滤器/原始特征

# 读取CSV文件
df = pd.read_csv(features_file_path)

# 获取除了前几列（通常是Image, Mask等信息列）之外的所有特征列名
# 假设前3列为非特征列（如Image, Mask, Label等信息），调整这个数值以适应您的数据
feature_columns = df.columns[3:]

# 计算每个特征类别下的特征数
feature_counts = {}
for feature_name in feature_columns:
    # 特征名称通常格式为 "类别_特征名"，例如 "firstorder_Entropy"
    category = feature_name.split('_')[0]
    feature_counts[category] = feature_counts.get(category, 0) + 1

# 打印每个特征类别下的特征数并准备数据以保存到CSV
feature_counts_for_csv = []
print("Feature Category,Count")
for category, count in feature_counts.items():
    print(f"{category},{count}")
    feature_counts_for_csv.append([category, count])

# 打印总特征数
total_features = len(feature_columns)
print(f"Total number of features: {total_features}")
feature_counts_for_csv.append(["Total", total_features])

# 保存到CSV
df_output = pd.DataFrame(feature_counts_for_csv, columns=['Feature Category', 'Count'])
df_output.to_csv(output_csv_path, index=False)

print(f"Feature counts have been saved to {output_csv_path}")
