import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tifffile as tiff
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 在数据预处理和特征选择之前，绘制原始数据的热图
def save_initial_heatmap(df, output_folder, filename="initial_feature_heatmap_allHeat.tiff"):
  # 选择数值型列
  df_numeric = df.select_dtypes(include=[np.number])

  plt.figure(figsize=(12, 10))
  sns.heatmap(df_numeric.corr(), annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
  plt.title('Initial Feature Correlation Heatmap')
  plt.tight_layout()
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  output_path = os.path.join(output_folder, filename)
  plt.savefig(output_path, format='tiff', dpi=300)
  plt.close()
  print(f"Initial heatmap saved to {output_path}")


# 1. 移除高度相关的特征
def remove_highly_correlated_features(df, threshold=0.99):  # ↓下面实际调用。 阈值越低，筛掉的越多（默认阈值）
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

# 移除高度相关的特征后，绘制热图
def save_heatmap_after_removal(df, output_folder, filename="feature_heatmap_after_removal_allHeat.tiff"):
  plt.figure(figsize=(12, 10))  # 可以根据需要调整大小
  sns.heatmap(df.corr(), annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
  plt.title('Feature Correlation Heatmap After Removal')
  plt.tight_layout()
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  output_path = os.path.join(output_folder, filename)
  plt.savefig(output_path, format='tiff', dpi=300)  # 增加dpi参数提高图像质量
  plt.close()
  print(f"Heatmap saved to {output_path}")


# 2. 保存特征相关性的热图  意义：热图可视化有助于直观地理解特征之间的相关性，为特征选择和模型优化提供参考。
def save_heatmap(df, output_folder, filename="feature_heatmap_bestk_allHeat.tiff"):
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

np.random.seed(42)

# 3. 使用ANOVA选择特征：通过交叉验证找到最佳的k值。选择与目标变量最相关的特征，可以提高（逻辑回归）模型的预测准确性，并减少训练时间。
def find_best_k(X, y, min_k=1, max_k=None, cv=10):  #  CV 交叉验证的折数！！！！！
  """
  通过交叉验证找到最佳的k值。

  参数:
  - X: 特征数据集。
  - y: 目标变量。
  - min_k: 考虑的最小特征数量。
  - max_k: 考虑的最大特征数量。如果为None，则设置为特征数量。
  - cv: 交叉验证的折数。

  返回:
  - best_k: 最佳的特征数量。
  - best_score: 对应的最佳分数。
  """
  if max_k is None:
    max_k = X.shape[1]

  scores = []
  k_values = range(min_k, max_k + 1)

  cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)


  for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline([('selector', selector), ('model', model)])
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy').mean()
    scores.append(score)

  best_score = max(scores)
  best_k = k_values[scores.index(best_score)]

  print(f"Best k: {best_k} with score: {best_score}")
  return best_k, best_score

#  预处理和选择最佳k个特征
def preprocess_and_select_features_with_best_k(data_path, output_folder):
  df = pd.read_csv(data_path)
  y = df.pop('event')
  time = df.pop('time')
  ID =  df.pop('ID')
  X = df.select_dtypes(include=[np.number])

  # 在数据预处理和特征选择之前，绘制并保存原始数据的热图
  save_initial_heatmap(df, output_folder)

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

  # 移除高度相关的特征后，保存热图
  X_filtered, dropped_features = remove_highly_correlated_features(X_scaled, threshold=0.9)  #（实际调用，覆盖默认）阈值越低，筛掉的越多
  save_heatmap_after_removal(X_filtered, output_folder, "heatmap_after_removing_correlation_allHeat.tiff")
  # 0.95 Best k: 5 with score: 0.7447619047619047
  # 0.9 Best k: 8 with score: 0.7514285714285714
  # 0.89 Best k: 8 with score: 0.7514285714285714
  # 0.88 Best k: 5 with score: 0.7314285714285715

  # 找到最佳的k值
  best_k, _ = find_best_k(X_filtered, y)

  # 使用最佳的k值进行特征选择
  selector = SelectKBest(f_classif, k=best_k)
  X_selected = selector.fit_transform(X_filtered, y)
  selected_features = X_filtered.columns[selector.get_support()]

  # 保存筛选后的特征
  X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

  X_selected_df['time'] = time.values
  X_selected_df['event'] = y.values
  X_selected_df['ID'] = ID.values
  selected_features_file = os.path.join(output_folder, "selected_features_with_bestk_allHeat.csv")
  X_selected_df.to_csv(selected_features_file, index=False)
  print(f"Selected features with best k saved to {selected_features_file}")

  # 绘制并保存热图
  save_heatmap(X_selected_df.drop(['event', 'time','ID'], axis=1), output_folder)


# Example call
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_ID_updated.csv"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data"
preprocess_and_select_features_with_best_k(data_path, output_folder)

