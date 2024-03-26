# Lasso2.py

from sklearn.linear_model import LassoCV, Lasso
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.linear_model import lasso_path
import seaborn as sns
from utils.visualization_utils import plot_feature_correlation_heatmap


import pandas as pd
import numpy as np

def load_data(data_path):
    """加载数据"""
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df, drop_columns=['ID', 'time'], target_column='event'):
  """预处理数据，提取特征和标签，同时保留ID列和时间列"""
  X = df.drop(drop_columns + [target_column], axis=1)
  y = df[target_column]
  ID_time = df[['ID', 'time']]
  event = df[target_column]  # 这里假设event列就是您想要的target_column
  return X, y, ID_time, event


def plot_mse_vs_log_alpha(lasso_cv, best_alpha, output_dir):
  mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
  mse_std = np.std(lasso_cv.mse_path_, axis=1)
  log_alphas = np.log10(lasso_cv.alphas_)

  # 选择性地减少点的数量以简化图形
  # indices = np.linspace(0, len(log_alphas) - 1, 200, dtype=int)  # 仅选择20个点进行绘图
  # selected_log_alphas = log_alphas[indices]
  # selected_mse_mean = mse_mean[indices]
  # selected_mse_std = mse_std[indices]

  # 显示所有的点： log_alphas, mse_mean, yerr=mse_std,
  # 显示选择个数的点 ：selected_log_alphas, selected_mse_mean, yerr=selected_mse_std,

  plt.figure(figsize=(10, 6))
  plt.errorbar(log_alphas, mse_mean, yerr=mse_std, fmt='o', color='red', ecolor='lightgray',
               elinewidth=3, capsize=0, label='Average MSE with CI')
  plt.axvline(np.log10(best_alpha), linestyle='--', color='black', label='Best alpha')
  plt.xlabel('Log(λ)')
  plt.ylabel('Mean Square Error (MSE)')
  plt.title('MSE vs. Log(λ) in LASSO')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "lasso_mse_vs_alpha_simplified.tiff"), dpi=300)
  plt.close()
  print(f"Simplified MSE vs. Log(λ) plot saved to {os.path.join(output_dir, 'lasso_mse_vs_alpha_simplified.tiff')}")


def run_lasso_preprocess(X, y, df, output_dir):
  """使用LASSO进行特征选择并保存选定的特征"""
  lasso_cv = LassoCV(cv=5,
                     random_state=42,
                     max_iter=1000000,
                     alphas=np.logspace(-6, 2, 100))
  lasso_cv.fit(X, y)
  best_alpha = lasso_cv.alpha_
  print(f"最佳alpha: {best_alpha}")

  lasso = Lasso(alpha=best_alpha, max_iter=1000000)
  lasso.fit(X, y)

  # 绘制LASSO系数剖面图
  alphas_lasso, coefs_lasso, _ = lasso_path(X, y,
                                            alphas=[best_alpha] + list(np.logspace(-6, 2, 100)))  # 长度与num有关
  plt.figure(figsize=(10, 6))
  log_alphas_lasso = np.log10(alphas_lasso)
  for coef_l in coefs_lasso:
    plt.plot(log_alphas_lasso, coef_l, linestyle='--')
  plt.axvline(np.log10(best_alpha), linestyle='-', color='k', label='Best alpha')
  plt.xlabel('Log(alpha)')
  plt.ylabel('Coefficients')
  plt.title('LASSO Coefficients Profile')
  plt.legend()
  plt.savefig(os.path.join(output_dir, "lasso_coefficients_profile.tiff"), dpi=300)
  plt.close()

  coef_mask = lasso.coef_ != 0
  selected_features = X.columns[coef_mask]
  print(f"Selected features with non-zero coefficients: {selected_features}")

  # 绘制选定特征的重要性并保存
  plt.figure(figsize=(10, 6))
  feature_importance = np.abs(lasso.coef_[coef_mask])
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(X.columns)[coef_mask][sorted_idx])
  plt.xlabel('Feature Importance')
  plt.title('Feature Importance (LASSO)')
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "lasso_feature_importance.tiff"), dpi=300)
  plt.close()

  selected_features_df = df.loc[:, selected_features]
  plot_feature_correlation_heatmap(selected_features_df, output_dir)
  plot_mse_vs_log_alpha(lasso_cv, best_alpha, output_dir)

  return selected_features


def apply_feature_selection(X, selected_features):
  """应用特征选择结果"""
  return X[selected_features]


def save_processed_data(X, ID_time, event, output_path):
  """保存处理后的数据，包括ID、时间和event列"""
  processed_df = pd.concat([ID_time.reset_index(drop=True), event.reset_index(drop=True), X.reset_index(drop=True)],
                           axis=1)
  processed_df.to_csv(output_path, index=False)
  print(f"Processed data saved to {output_path}")


def main(data_path_train, data_path_val, output_dir, data_path_test=None):
  # 加载训练数据
  df_train = load_data(data_path_train)
  # 使用preprocess_data函数预处理训练数据
  X_train, y_train, ID_time_train, event_train = preprocess_data(df_train)  # 确保preprocess_data返回event列

  # 运行LASSO进行特征选择
  selected_features = run_lasso_preprocess(X_train, y_train, df_train, output_dir)

  df_train_selected = apply_feature_selection(X_train, selected_features)

  save_processed_data(df_train_selected, ID_time_train, event_train,
                      os.path.join(output_dir, "processed_train_lasso.csv"))  # 传入event列
  if data_path_val:
    df_val = load_data(data_path_val)
    X_val_selected = apply_feature_selection(df_val.drop(['ID', 'time', 'event'], axis=1), selected_features)
    save_processed_data(X_val_selected, df_val[['ID', 'time']], df_val['event'],
                        os.path.join(output_dir, "processed_val_lasso.csv"))
  if data_path_test:
    df_test = load_data(data_path_test)
    X_test_selected = apply_feature_selection(df_test.drop(['ID', 'time', 'event'], axis=1), selected_features)
    save_processed_data(X_test_selected, df_test[['ID', 'time']], df_test['event'],
                        os.path.join(output_dir, "processed_test_lasso.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LASSO preprocessing and feature selection.")
    parser.add_argument('--data-path-train', type=str, default=r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_train.csv", help='Path to the training data file.')
    parser.add_argument('--data-path-val', type=str, default=r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_val.csv", help='Path to the validation data file.')
    parser.add_argument('--data-path-test', type=str,default=r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data_test.csv",  help='Path to the test data file, if exists.')
    parser.add_argument('--output-dir', type=str,default=r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data",help='Directory to save the output files.')

    args = parser.parse_args()

    main(args.data_path_train, args.data_path_val, args.output_dir, args.data_path_test)
