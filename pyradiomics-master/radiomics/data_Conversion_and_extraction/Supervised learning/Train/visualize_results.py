import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_predict
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
# 3.可视化结果

# 修改可视化结果函数以接收额外的参数
# 修改函数定义，添加 results 参数
def visualize_results(X, y, models, cv, output_folder, results):
    plot_roc_curves(X, y, models, cv, output_folder)
    plot_precision_recall_curves(X, y, models, cv, output_folder)
    # 现在 results 参数已经被定义，可以被下面的函数使用
    plot_performance_heatmap(results, output_folder)


# 1. 繪製jama格式的每个AUC图
def plot_model_roc_curve(X, y, model, model_name, cv, output_folder):
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)

  sns.set_style("whitegrid")
  fig, ax = plt.subplots(figsize=(8, 6))
  for i, (train, test) in enumerate(cv.split(X, y)):
    model.fit(X.iloc[train], y.iloc[train])
    y_pred_proba = model.predict_proba(X.iloc[test])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.iloc[test], y_pred_proba)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

  mean_tpr = np.mean(tprs, axis=0)
  mean_auc = np.mean(aucs)
  std_auc = np.std(aucs)
  ax.plot(mean_fpr, mean_tpr, color='red', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
          alpha=.8)  # 红色来表示平均ROC曲线
  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)  # 黑色表示随机机会线（Chance line）

  # JAMA格式美化
  ax.set_xlabel('1-Specificity%', fontsize=14)
  ax.set_ylabel('Sensitivity%', fontsize=14)
  ax.set_title(f'Training ROC Curve for {model_name}', fontsize=16)
  ax.legend(loc="lower right", fontsize=12)
  ax.grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(output_folder, f'ROC_Curve_{model_name}.tiff'), format='tiff', dpi=300)
  plt.close()


# 2.模型ROC AUC分数·箱式图·
def plot_model_roc_auc_boxplot(results, output_folder):
  roc_auc_scores = {model: scores['test_roc_auc'] for model, scores in results.items()}
  plt.figure(figsize=(10, 8))
  # 使用Accent或Pastel1调色板，这些调色板提供了鲜艳而清晰的颜色，有助于区分不同的模型。
  # 修改为纵向排列，模型名称在x轴，ROC AUC分数在y轴
  sns.boxplot(data=pd.DataFrame(roc_auc_scores).melt(var_name='Model', value_name='ROC AUC Score'), y='ROC AUC Score',
              x='Model', palette="Set3", orient="v")
  plt.title('ROC AUC Scores for Different Models', fontsize=20, weight='bold')
  plt.ylabel('ROC AUC Score', fontsize=16, weight='bold')  # 更新为ylabel因为现在分数在y轴
  plt.xticks(fontsize=14, rotation=45)  # 旋转x轴标签以更好地展示模型名称
  plt.yticks(fontsize=14)
  plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
  plt.tight_layout()
  # 保存为高清TIFF格式
  plt.savefig(os.path.join(output_folder, 'model_roc_auc_boxplot.tiff'), format='tiff', dpi=300)
  plt.close()


# 3.平均ROC AUC分数·柱状图·
def plot_mean_roc_auc_scores_barplot(results, output_folder):
  mean_roc_auc_scores = {model: np.mean(scores['test_roc_auc']) for model, scores in results.items()}
  plt.figure(figsize=(12, 8))
  # 建议：使用tab20调色板，它提供了丰富的颜色选择，适合区分多个模型。
  # 'tab10', 'tab20'：为分类数据提供了一组鲜艳且易于区分的颜色。
  # 'Pastel1', 'Pastel2'：提供了较为柔和的颜色，适合温和的视觉效果。
  sns.barplot(x=list(mean_roc_auc_scores.keys()), y=list(mean_roc_auc_scores.values()), palette="tab20")
  plt.xticks(rotation=45, fontsize=14, weight='bold')
  plt.yticks(fontsize=14, weight='bold')
  plt.title('Mean ROC AUC Scores for Different Models', fontsize=20, weight='bold')
  plt.xlabel('Model', fontsize=16, weight='bold')
  plt.ylabel('Mean ROC AUC Score', fontsize=16, weight='bold')
  plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
  plt.tight_layout()
  # 修改保存图形的代码
  plt.savefig(os.path.join(output_folder, 'mean_roc_auc_scores_barplot.tiff'), format='tiff', dpi=300)
  plt.close()


# 4.排序的平均ROC AUC分数横状图
def plot_sorted_mean_roc_auc_scores_barplot(results, output_folder):
  mean_roc_auc_scores = {model: np.mean(scores['test_roc_auc']) for model, scores in results.items()}
  sorted_models = sorted(mean_roc_auc_scores.items(), key=lambda x: x[1], reverse=True)  # 使用items()获取键值对并排序

  models = [model for model, _ in sorted_models]
  scores = [score for _, score in sorted_models]

  plt.figure(figsize=(10, 8))
  # 建议：使用Spectral调色板，它通过颜色的变化强调了性能的高低，非常适合展示性能排序。
  sns.barplot(x=scores, y=models, palette="coolwarm", orient='h')  # 使用orient='h'绘制水平条形图
  plt.title('Sorted Mean ROC AUC Scores for Different Models', fontsize=20, weight='bold')
  plt.xlabel('Mean ROC AUC Score', fontsize=16, weight='bold')
  plt.ylabel('Model', fontsize=16, weight='bold')

  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
  plt.tight_layout()
  # 修改保存图形的代码
  plt.savefig(os.path.join(output_folder, 'sorted_mean_roc_auc_scores_barplot.tiff'), format='tiff', dpi=300)
  plt.close()


# 5. ROC曲线比较图 ： ROC曲线比较了多个模型的性能，使用不同的颜色来区分每个模型是有帮助的。
def plot_roc_curves(X, y, models, cv, output_folder):
  plt.figure(figsize=(10, 8))  # 优化图形大小
  sns.set_style("whitegrid")  # 设置背景为白色网格，增加可读性

  model_aucs = []

  for name, model in models.items():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_idx, test_idx in cv.split(X, y):
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]
      model.fit(X_train, y_train)
      y_score = model.predict_proba(X_test)[:, 1]
      fpr, tpr, _ = roc_curve(y_test, y_score)
      tprs.append(np.interp(mean_fpr, fpr, tpr))
      roc_auc = auc(fpr, tpr)
      aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    model_aucs.append((name, mean_auc, std_auc, mean_fpr, mean_tpr))

  # Sort models based on mean AUC
  model_aucs.sort(key=lambda x: x[1], reverse=True)

  # cividis是一个为视觉障碍者设计的颜色方案，同时也是一个在科学出版物中广受欢迎的选择，因为它在黑白打印和彩色打印中都保持一致性。
  palette = plt.get_cmap('tab20')
  # tab10和tab20调色板提供了一组设计精良的颜色，适合区分多个类别或模型。这些颜色在科学可视化中使用广泛，因为它们在视觉上具有很好的区分度。
  # palette = plt.get_cmap('tab10')

  for i, (name, mean_auc, std_auc, mean_fpr, mean_tpr) in enumerate(model_aucs):
    plt.plot(mean_fpr, mean_tpr, color=palette(i / len(models)),
             label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)
  plt.xlabel('1-Specificity%', fontsize=16, weight='bold')
  plt.ylabel('Sensitivity%', fontsize=16, weight='bold')
  plt.title('Training ROC Curves Comparison', fontsize=20, weight='bold')
  plt.legend(loc="lower right", fontsize=8)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)

  plt.savefig(os.path.join(output_folder, 'roc_curves_comparison.tiff'), format='tiff', dpi=300)
  plt.close()


# 6.精确率-召回率曲线
def plot_precision_recall_curves(X, y, models, cv, output_folder):
  plt.figure(figsize=(8, 6))  # 调整图形大小以适应Radiology格式
  sns.set(style="whitegrid")  # 使用白色网格背景提高清晰度

  model_scores = []

  for name, model in models.items():
    y_real = []
    y_proba = []

    for train_idx, test_idx in cv.split(X, y):
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]
      model.fit(X_train, y_train)
      y_score = model.predict_proba(X_test)[:, 1]
      y_real.append(y_test)
      y_proba.append(y_score)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ap_score = average_precision_score(y_real, y_proba)
    model_scores.append((name, ap_score, precision, recall))

  # 根据AP分数对模型进行排序
  model_scores.sort(key=lambda x: x[1], reverse=True)

  # 建议：使用matplotlib的tab10或Set2调色板，这些调色板提供了良好的颜色区分度。
  palette = plt.get_cmap('tab20')  # 获取调色板

  for i, (name, ap_score, precision, recall) in enumerate(model_scores):
    plt.plot(recall, precision, color=palette(i), label=f'{name} (AP = {ap_score:.2f})', lw=2)

  plt.xlabel('Recall', fontsize=14, weight='bold')
  plt.ylabel('Precision', fontsize=14, weight='bold')
  plt.title('Precision-Recall Curves Comparison', fontsize=16, weight='bold')

  plt.legend(loc="upper right", fontsize=8, frameon=True, shadow=True)  # 左下角插圖太大
  # 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'

  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)

  plt.savefig(os.path.join(output_folder, 'precision_recall_curves_comparison.tiff'), format='tiff', dpi=300)
  plt.close()


# plt.legend(loc="lower left", fontsize=12): 添加图例，并将图例放置在左下角位置，指定图例字体大小为 12。

# 7. 性能对比热图
# 性能热图 ： 性能热图展示了不同模型在多个评价指标上的性能，使用渐变色可以很好地表示性能的高低。
def plot_performance_heatmap(results, output_folder):
  # 创建性能DataFrame
  performance_df = pd.DataFrame.from_dict(
    {model: {metric: np.mean(scores[f'test_{metric}']) for metric in ['roc_auc', 'accuracy', 'recall', 'f1']} for
     model, scores in results.items()}, orient='index')

  plt.figure(figsize=(10, 7))  # 调整图形大小以适应格式
  sns.set(style="white")  # 使用白色背景增强清晰度

  # 建议：使用coolwarm调色板，它提供了从蓝色（表示低值）到红色（表示高值）的渐变，非常适合表示性能的好坏。
  # sns.heatmap(performance_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black', cbar_kws={'shrink': .82})
  # 建议：RdYlBu调色板提供了从红色（表示高值）到蓝色（表示低值）的颜色变化，非常适合用于表示性能的对比。
  sns.heatmap(performance_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'shrink': .5})

  # 美化图形
  plt.title('Model Performance Heatmap', fontsize=16, weight='bold', pad=20)
  plt.ylabel('Model', fontsize=14, weight='bold')
  plt.xlabel('Performance Metric', fontsize=14, weight='bold')
  plt.xticks(fontsize=12, rotation=45, ha="right", weight='bold')
  plt.yticks(fontsize=12, rotation=0, weight='bold')

  # 保存图形
  plt.tight_layout()
  plt.savefig(os.path.join(output_folder, 'model_performance_heatmap.tiff'), format='tiff', dpi=300)
  plt.close()
