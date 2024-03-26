import pandas as pd

# 数据集路径
dataset_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\Voxel_feature_R1B12.csv"
# 输出文件路径
output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\calculated\Voxel_R1B12_SubfeatureCounts.csv"

# 读取CSV文件
df = pd.read_csv(dataset_path)

# 特征类别列表
feature_categories = ['firstorder','shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']


# 初始化计数字典
feature_counts = {category: 0 for category in feature_categories}

# 计算每个特征类别下的特征数量
for column in df.columns:
    for category in feature_categories:
        # 根据提供的命名规则调整，确保匹配正确
        if f"_{category}_" in column:
            feature_counts[category] += 1

# 打印计数结果
print("Feature Category, Count")
feature_counts_summary = []
for category, count in feature_counts.items():
    print(f"{category}: {count}")
    feature_counts_summary.append([category, count])

# 总特征数量
total_features = sum(feature_counts.values())
print(f"Total: {total_features}")
feature_counts_summary.append(["Total", total_features])

# 保存结果到CSV文件
df_output = pd.DataFrame(feature_counts_summary, columns=['Feature Category', 'Count'])
df_output.to_csv(output_path, index=False)

print(f"Feature counts have been saved to {output_path}")
