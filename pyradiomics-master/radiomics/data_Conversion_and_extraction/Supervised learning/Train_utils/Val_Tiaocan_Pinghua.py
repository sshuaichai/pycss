import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
import numpy as np

def load_data_and_model(model_path, val_set_path):
  # 加载模型
  model = joblib.load(model_path)
  # 加载验证集
  val_df = pd.read_csv(val_set_path)
  X_val = val_df.drop(['time', 'event','ID'], axis=1)  # 假设'event'是目标变量，'time'是不需要的列
  y_val = val_df['event']
  return model, X_val, y_val


def evaluate_model(model, X_val, y_val):
  # 使用模型对验证集进行预测
  y_pred = model.predict(X_val)
  y_pred_proba = model.predict_proba(X_val)[:, 1]
  # 计算性能指标
  metrics = {
    "ROC AUC": roc_auc_score(y_val, y_pred_proba),
    "Accuracy": accuracy_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1 Score": f1_score(y_val, y_pred)
  }
  return metrics, y_pred_proba


def plot_roc_curve(y_val, y_pred_proba, output_folder):
  fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
  roc_auc = roc_auc_score(y_val, y_pred_proba)

  '''
  # 使用UnivariateSpline进行平滑
  '''
  spline = UnivariateSpline(fpr, tpr)
  new_fpr = np.linspace(0, 1, 300)
  smooth_tpr = spline(new_fpr)

  sns.set_style("whitegrid")
  plt.figure(figsize=(10, 8))
  plt.plot(new_fpr, smooth_tpr, color='red', label=f'ROC curve (area = {roc_auc:.2f})',lw=3, alpha=.8) ## 红色来表示平均ROC曲线,加粗用 lw 线宽
  plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance', alpha=.8) #alpha = 0 表示完全透明，0 < alpha < 1 表示部分透明，具体程度取决于 alpha 的值。
  plt.xlabel('1-Specificity%')
  plt.ylabel('Sensitivity%')
  plt.title('Train_utils ROC Curve')
  plt.legend(loc="lower right")
  plt.tight_layout()

  output_path = os.path.join(output_folder, "Val_roc_curve_validation_Tiancan.tiff")
  plt.savefig(output_path, format='tiff', dpi=300)
  plt.close()

  print(f"ROC curve saved to {output_path}")


if __name__ == "__main__":
  # 模型和数据集的路径
  model_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_TiaoCan_models\LGBMClassifier_model.joblib"
  val_set_path = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Data_set_partition\\val022_set.csv"
  output_folder = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Val_Output"

  model, X_val, y_val = load_data_and_model(model_path, val_set_path)
  metrics, y_pred_proba = evaluate_model(model, X_val, y_val)

  for metric, value in metrics.items():
    print(f"{metric}: {value}")

  plot_roc_curve(y_val, y_pred_proba, output_folder)
