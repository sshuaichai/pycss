# Pearson_and_standardize.py

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(data_path):
    """加载数据"""
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df, drop_columns=['ID', 'time', 'event'], target_column='event'):
    """预处理数据，提取特征和标签，同时保留ID列"""
    X = df.drop(drop_columns, axis=1)
    y = df[target_column]
    return X, y, df[['ID', 'time']]

def select_numeric_columns(X, output_path_initial_heatmap):
  """仅选择数值型列，并绘制初始相关性热图"""
  X_numeric = X.select_dtypes(include=[np.number])
  corr_matrix_initial = X_numeric.corr()
  plt.figure(figsize=(10, 8))
  sns.heatmap(corr_matrix_initial, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
  plt.title('Initial Feature Correlation Matrix Heatmap')
  plt.savefig(output_path_initial_heatmap, format='tiff', dpi=300)
  plt.close()
  return X_numeric


def analyze_correlation(X_numeric, output_path_heatmap):
  """分析特征间的相关性，并绘制热图"""
  corr_matrix = X_numeric.corr()
  plt.figure(figsize=(10, 8))
  sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
  plt.title('Feature Correlation Matrix Heatmap After Removing Highly Correlated Features')
  plt.savefig(output_path_heatmap, format='tiff', dpi=300)
  plt.close()
  return corr_matrix

def remove_highly_correlated_features(X_numeric, corr_matrix, threshold=0.64):
  """移除高度相关的特征"""
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
  X_filtered = X_numeric.drop(columns=to_drop)
  return X_filtered, to_drop


def standardize_features(X_filtered):
  """标准化特征"""
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_filtered)
  X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered.columns)
  return X_scaled_df, scaler


def apply_preprocessing(X, scaler):
  """应用预处理到数据集"""
  # 直接使用scaler进行transform，不再需要to_drop参数
  X_scaled = scaler.transform(X)
  X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
  return X_scaled_df


def save_processed_data(X_scaled_df, y, time, ID, output_path):
  """保存处理后的数据，包括ID列"""
  processed_df = pd.concat([ID, X_scaled_df, y, time], axis=1)
  processed_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # 更新路径以匹配您的文件结构
    train_data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_set_RADIOMICS.csv"
    val_data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_set_RADIOMIC.csv"
    test_data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\test_set_RADIOMIC.csv"  # 测试集路径，如果没有测试集则忽略此行

    output_path_initial_heatmap = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\initial_correlation_matrix_heatmap.tiff"
    output_path_heatmap = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\Removing_Highly_Correlated_Features_correlation_matrix_heatmap.tiff"

    output_path_processed_data_train = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_train.csv"
    output_path_processed_data_val = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_val.csv"
    output_path_processed_data_test = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_test.csv"  # 测试集处理后的数据保存路径

    # 加载和预处理训练数据
    df_train = load_data(train_data_path)
    X_train, y_train, ID_time_train = preprocess_data(df_train)
    time_train = ID_time_train['time']
    ID_train = ID_time_train['ID']

    X_train_numeric = select_numeric_columns(X_train, output_path_initial_heatmap)
    corr_matrix = analyze_correlation(X_train_numeric, output_path_heatmap)
    X_train_filtered, to_drop = remove_highly_correlated_features(X_train_numeric, corr_matrix)
    X_train_scaled, scaler = standardize_features(X_train_filtered)
    save_processed_data(X_train_scaled, y_train, time_train, ID_train, output_path_processed_data_train)

    # 加载和预处理验证数据
    df_val = load_data(val_data_path)
    X_val, y_val, ID_time_val = preprocess_data(df_val)
    time_val = ID_time_val['time']
    ID_val = ID_time_val['ID']
    # 确保特征名称与训练集一致，并且顺序也相同
    X_val_filtered = X_val.reindex(columns=X_train_filtered.columns, fill_value=0)
    X_val_scaled = apply_preprocessing(X_val_filtered, scaler)
    save_processed_data(X_val_scaled, y_val, time_val, ID_val, output_path_processed_data_val)

    # 如果存在测试集，加载和预处理测试数据
    try:
      df_test = load_data(test_data_path)
      X_test, y_test, ID_time_test = preprocess_data(df_test)
      time_test = ID_time_test['time']
      ID_test = ID_time_test['ID']
      # 确保特征名称与训练集一致，并且顺序也相同
      X_test_filtered = X_test.reindex(columns=X_train_filtered.columns, fill_value=0)
      X_test_scaled = apply_preprocessing(X_test_filtered, scaler)
      save_processed_data(X_test_scaled, y_test, time_test, ID_test, output_path_processed_data_test)
    except FileNotFoundError:
      print("Test dataset not found. Skipping test dataset processing.")
