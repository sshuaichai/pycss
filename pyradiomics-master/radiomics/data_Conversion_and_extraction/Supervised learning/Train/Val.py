import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_roc_curve(y_val, y_pred_proba, model_name, output_folder):
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='red', label=f'ROC curve (area = {roc_auc:.2f})', lw=3, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance', alpha=.8)
    plt.xlabel('1-Specificity%')
    plt.ylabel('Sensitivity%')
    plt.title(f'Train_utils ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_path = os.path.join(output_folder, f"ROC_Curve_{model_name}.tiff")
    plt.savefig(output_path, format='tiff', dpi=300)
    plt.close()

    print(f"ROC curve for {model_name} saved to {output_path}")


def plot_combined_roc_curves(results_df, output_folder, y_val):
  plt.figure(figsize=(10, 8))
  sns.set_style("whitegrid")

  # 准备一个列表来存储每个模型的AUC和颜色信息
  model_aucs = []

  # 使用cividis调色板
  palette = plt.get_cmap('tab20')

  # 遍历results_df来计算每个模型的AUC，并添加到model_aucs列表中
  for index, row in results_df.iterrows():
    y_pred_proba = row['y_pred_proba']
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    model_aucs.append((row['Model'], roc_auc))

  # 根据AUC排序
  model_aucs.sort(key=lambda x: x[1], reverse=True)

  # 绘制每个模型的ROC曲线
  for i, (model_name, roc_auc) in enumerate(model_aucs):
    y_pred_proba = results_df.loc[results_df['Model'] == model_name, 'y_pred_proba'].values[0]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, color=palette(i / len(model_aucs)), label=f'{model_name} (AUC = {roc_auc:.2f})', lw=2)

  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)
  plt.xlabel('False Positive Rate', fontsize=16, weight='bold')
  plt.ylabel('True Positive Rate', fontsize=16, weight='bold')
  plt.title('Combined ROC Curves for All Models', fontsize=20, weight='bold')
  plt.legend(loc="lower right", fontsize=8)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)

  output_path = os.path.join(output_folder, "Combined_ROC_Curves.tiff")
  plt.savefig(output_path, format='tiff', dpi=300)
  plt.close()

  print(f"Combined ROC curves saved to {output_path}")


def validate_models(model_folder, val_set_path, output_folder):
  # 加载验证集
  val_df = pd.read_csv(val_set_path)
  X_val = val_df.drop(['time', 'event', 'ID'], axis=1)
  y_val = val_df['event']

  # 初始化一个空的DataFrame来存储所有模型的结果
  results_df = pd.DataFrame(columns=['Model', 'ROC AUC', 'Accuracy', 'Recall', 'F1 Score', 'y_pred_proba'])

  # 遍历模型文件夹中的所有模型
  for model_file in os.listdir(model_folder):
    if model_file.endswith(".joblib"):
      model_path = os.path.join(model_folder, model_file)
      model = joblib.load(model_path)
      model_name = model_file.replace("_model.joblib", "")

      # 使用模型对验证集进行预测
      y_pred = model.predict(X_val)
      y_pred_proba = model.predict_proba(X_val)[:, 1]

      # 计算性能指标
      roc_auc = roc_auc_score(y_val, y_pred_proba)
      accuracy = accuracy_score(y_val, y_pred)
      recall = recall_score(y_val, y_pred)
      f1 = f1_score(y_val, y_pred)

      # 打印性能指标
      print(f"Performance metrics for {model_name}:")
      print(f"ROC AUC: {roc_auc:.4f}")
      print(f"Accuracy: {accuracy:.4f}")
      print(f"Recall: {recall:.4f}")
      print(f"F1 Score: {f1:.4f}\n")

      # 将结果添加到DataFrame
      results_df = pd.concat([results_df, pd.DataFrame(
        {'Model': [model_name], 'ROC AUC': [roc_auc], 'Accuracy': [accuracy], 'Recall': [recall], 'F1 Score': [f1],
         'y_pred_proba': [y_pred_proba.tolist()]})], ignore_index=True)

      # 为每个模型绘制ROC曲线
      plot_roc_curve(y_val, y_pred_proba, model_name, output_folder)

  # 保存结果到CSV文件
  results_df.to_csv(os.path.join(output_folder, "validation_results.csv"), index=False)
  print("Train_utils results saved to validation_results.csv")

  # 绘制汇总的ROC曲线
  plot_combined_roc_curves(results_df, output_folder, y_val)


if __name__ == "__main__":
    model_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_lasso_models"
    val_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_val_lasso.csv"
    output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Val_Output"

    validate_models(model_folder, val_set_path, output_folder)
