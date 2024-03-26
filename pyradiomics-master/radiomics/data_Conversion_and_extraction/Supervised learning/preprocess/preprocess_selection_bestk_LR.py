import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# 1. 移除高度相关的特征
def remove_highly_correlated_features(df, threshold=0.9):  #  ↓下面还有！！！ 阈值越低，筛掉的越多（默认阈值）
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop



# 2. 保存特征相关性的热图  意义：热图可视化有助于直观地理解特征之间的相关性，为特征选择和模型优化提供参考。
def save_heatmap(df, output_folder, filename="feature_heatmap_bestk.tiff"):
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

# 3. 基于模型使用ANOVA选择特征：通过交叉验证找到最佳的k值。选择与目标变量最相关的特征，可以提高模型的预测准确性，并减少训练时间。
def find_best_k(X, y, min_k=1, max_k=None, cv=10):   #  CV 交叉验证的折数！！！！！
  """
  通过交叉验证找到最佳的k值。
  """
  if max_k is None:
    max_k = X.shape[1]

  scores = []
  k_values = range(min_k, max_k + 1)

  # 使用固定的随机状态进行分层k折交叉验证
  cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

  for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline([('selector', selector),
                         ('model', model)])
    score = cross_val_score(pipeline, X, y,
                            cv=cv,
                            scoring='roc_auc').mean()
    scores.append(score)

  best_score = max(scores)
  best_k = k_values[scores.index(best_score)]

  print(f"Best k: {best_k} with score: {best_score}")
  return best_k, best_score


def preprocess_and_select_features_with_best_k(data_path, output_folder):
  df = pd.read_csv(data_path)
  y = df.pop('event')
  time = df.pop('time')
  ID =  df.pop('ID')
  X = df.select_dtypes(include=[np.number])

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

  X_filtered, dropped_features = remove_highly_correlated_features(X_scaled, threshold=0.98) # （实际调用，覆盖默认）阈值越低，筛掉的越多

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

  selected_features_file = os.path.join(output_folder, "selected_features_with_bestk.csv") # 所有体素特徵
  X_selected_df.to_csv(selected_features_file, index=False)
  print(f"Selected features with best k saved to {selected_features_file}")

  # 绘制并保存热图
  save_heatmap(X_selected_df.drop(['event', 'time','ID'], axis=1), output_folder)


# Example call
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_ID_updated.csv"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data"
preprocess_and_select_features_with_best_k(data_path, output_folder)
