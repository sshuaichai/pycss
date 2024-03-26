import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tifffile as tiff

# 1. 移除高度相关的特征
def remove_highly_correlated_features(df, threshold=0.7):  # threshold：相关系数的阈值，用于判断特征是否高度相关。
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop


# 2. 使用ANOVA选择特征： k：要选择的特征数量。意义：通过选择与目标变量最相关的特征，可以提高模型的预测准确性，并减少训练时间。
def feature_selection_anova(X, y, k=10):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_features), selected_features

# 3. 保存特征相关性的热图  意义：热图可视化有助于直观地理解特征之间的相关性，为特征选择和模型优化提供参考。
def save_heatmap(df, output_folder, filename="feature_heatmap_k10.tiff"):
  plt.figure(figsize=(10, 8))
  # 将 annot 参数设置为 True 以显示数值
  sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', xticklabels=True, yticklabels=True)
  plt.xticks(rotation=45, ha='right', fontsize=8)  # 调整x轴标签的字体大小和旋转角度
  plt.yticks(fontsize=8)  # 调整y轴标签的字体大小
  plt.title('Feature Correlation Heatmap')
  plt.tight_layout()
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  output_path = os.path.join(output_folder, filename)
  plt.savefig(output_path, format='tiff')
  plt.close()
  print(f"Heatmap saved to {output_path}")

# 4. 预处理并选择特征
# 意义：这个流程通过减少特征数量和去除不必要的特征，有助于提高模型的性能和解释性。同时，通过可视化特征之间的相关性，可以更好地理解数据集的特性。
def preprocess_and_select_features(data_path, output_folder):
    """
    加载数据集，分离目标变量event和不参与特征选择的time列。
    对数值型特征进行标准化处理。
    移除高度相关的特征。
    使用ANOVA选择与目标变量最相关的k=10个特征。
    将筛选后的特征、time列和目标变量event保存到CSV文件中。
    绘制并保存特征相关性的热图。
    """
    df = pd.read_csv(data_path)
    y = df.pop('event')  # Remove the target variable
    time = df.pop('time')  # 删除时间列，但保留它以供以后使用
    ID = df.pop('ID')
    X = df.select_dtypes(include=[np.number])  # 仅选择数字列
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_filtered_corr, dropped = remove_highly_correlated_features(X_scaled)
    print(f"Dropped features due to high correlation: {dropped}")

    X_selected, selected_features = feature_selection_anova(X_filtered_corr, y, k=10) #使用ANOVA选择与目标变量最相关的10个特征。
    print(f"Selected features: {selected_features}")

    # 将'time'列添加回最终CSV的选定特性中
    X_selected['time'] = time.values
    X_selected['event'] = y.values  # 重新添加目标变量
    X_selected['ID'] = ID.values
    selected_features_file = os.path.join(output_folder, "selection_anova_k10.csv")
    X_selected.to_csv(selected_features_file, index=False)
    print(f"Selected features k10 with event and time saved to {selected_features_file}")

    # Draw and save the heatmap
    save_heatmap(X_selected.drop(['event', 'time','ID'], axis=1), output_folder)  # 从热图中排除“事件”和“时间”

# Example call
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_ID_updated.csv"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data"
preprocess_and_select_features(data_path, output_folder)
