import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(data_path):
    """加载处理后的数据"""
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    y = df['event']
    X = df.drop(['event', 'time', 'ID'], axis=1)
    return X, y


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


def save_heatmap(df, output_folder, filename):
  """
  绘制特征相关性热图并保存到指定目录。
  """
  # 确保使用的是函数参数df来计算相关性矩阵
  correlation_matrix = df.corr()
  plt.figure(figsize=(12, 10))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
  feature_names = df.columns
  split_names = split_feature_names(feature_names)
  plt.xticks(np.arange(len(split_names)) + .5, split_names, rotation=45, ha="right", fontsize=10)
  plt.yticks(np.arange(len(split_names)) + .5, split_names, rotation=0, fontsize=10)
  plt.title('Selected Features Correlation Heatmap', fontsize=16)
  # 使用正确的变量名
  heatmap_path = os.path.join(output_folder, filename)
  plt.savefig(heatmap_path, dpi=300, format='tiff', bbox_inches='tight')
  plt.close()
  print(f"Correlation heatmap saved to {heatmap_path}")


def find_best_k(X, y, cv=5):
    """基于模型使用ANOVA选择特征"""
    logging.info("Finding best k for feature selection")
    max_k = X.shape[1]
    scores = []
    k_values = range(1, max_k + 1)
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for k in k_values:
      selector = SelectKBest(f_classif, k=k)
      model = LogisticRegression(max_iter=1000,
                                 random_state=42)
      pipeline = Pipeline([('selector', selector), ('model', model)])
      score = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc').mean()
      scores.append(score)
    best_score = max(scores)
    best_k = k_values[scores.index(best_score)]
    logging.info(f"Best k: {best_k} with score: {best_score}")
    return best_k


def apply_feature_selection_and_save(X_train, y_train, dataset_paths, dataset_names, output_folder):
  """应用特征选择并保存结果"""
  best_k = find_best_k(X_train, y_train)
  selector = SelectKBest(f_classif, k=best_k).fit(X_train, y_train)

  for data_path, dataset_name in zip(dataset_paths, dataset_names):
    X, y = load_processed_data(data_path)
    X_selected = selector.transform(X)
    selected_features = X_train.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    X_selected_df['event'] = y
    filename = f"selected_features_{dataset_name}_with_bestk.csv"
    output_path = os.path.join(output_folder, filename)
    X_selected_df.to_csv(output_path, index=False)
    logging.info(f"Selected features with best k for {dataset_name} saved to {filename}")


def main(config):
  """主函数"""
  X_train, y_train = load_processed_data(config['train_data_path'])
  dataset_paths = [config['train_data_path'], config['val_data_path'], config['test_data_path']]
  dataset_names = ["train", "val", "test"]

  apply_feature_selection_and_save(X_train, y_train, dataset_paths, dataset_names, config['output_folder'])


if __name__ == "__main__":
  config = {
    'train_data_path': r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_train.csv",
    'val_data_path': r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_val.csv",
    'test_data_path': r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_test.csv",
    'output_folder': r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data"
  }
  main(config)

