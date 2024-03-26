# visualization_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def split_feature_names(feature_names):
    """
    将特征名称在最接近中点的下划线处分割成两行。
    """
    split_names = []
    for name in feature_names:
        mid_point = len(name) // 2
        left_index = name.rfind('_', 0, mid_point)
        right_index = name.find('_', mid_point)
        if left_index == -1 and right_index == -1:
            split_point = len(name)
        elif left_index == -1:
            split_point = right_index
        elif right_index == -1:
            split_point = left_index
        else:
            split_point = left_index if (mid_point - left_index) < (right_index - mid_point) else right_index
        split_name = name if split_point == len(name) else name[:split_point] + '\n' + name[split_point + 1:]
        split_names.append(split_name)
    return split_names

def plot_feature_correlation_heatmap(selected_features_df, output_dir, file_name="selected_features_correlation_heatmap.tiff"):
    """
    绘制特征相关性热图并保存到指定目录。
    """
    correlation_matrix = selected_features_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    feature_names = selected_features_df.columns
    split_names = split_feature_names(feature_names)
    plt.xticks(np.arange(len(split_names)) + .5, split_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(len(split_names)) + .5, split_names, rotation=0, fontsize=10)
    plt.title('Selected Features Correlation Heatmap', fontsize=16)
    heatmap_path = os.path.join(output_dir, file_name)
    plt.savefig(heatmap_path, dpi=300, format='tiff', bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")
