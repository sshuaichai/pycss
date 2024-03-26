import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from joblib import dump
import os
import logging
logging.basicConfig(filename='../Train/training_errors.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# 定义文件和输出路径  train_RF_mean_ / train_RF_median_   /Lasso_preprocessing_select_train_RF_参数_
'先定义参数！！！！预处理后的训练集'
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train022_set.csv"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output"
encoders_folder = os.path.join(output_folder, "train_encoders") #Lasso_preprocessing_
model_train_folder = os.path.join(output_folder, "train_models")
visualization_folder = os.path.join(output_folder, "train_visualizations")

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(encoders_folder, exist_ok=True)
os.makedirs(model_train_folder, exist_ok=True)
os.makedirs(visualization_folder, exist_ok=True)

# 数据预处理和编码 ： 对分类特征进行编码
def preprocess_data(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        # 保存LabelEncoder实例
        encoder_file = os.path.join(encoders_folder, f"{column}_encoder.joblib")
        dump(le, encoder_file)
    return df, label_encoders

# 选择并训练模型，保存模型和特征选择结果：定义了train_and_save_models函数来训练多个模型并使用5折交叉验证评估它们的性能。
def train_and_save_models(X, y, models, cv, output_folder):
  results = {}
  evaluation_results = []
  for name, model in models.items():
        # 简化的管道，只包括分类器
        pipeline = Pipeline([
          ('classifier', model)  # `model` 可以是任何 sklearn 兼容的分类器
        ])
        try:
          # 交叉验证训练模型
          cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=['roc_auc', 'accuracy', 'recall', 'f1'],
                                      return_train_score=False, n_jobs=-1)
          results[name] = cv_results

          evaluation_results.append({
            'Model': name,
            'ROC AUC Mean': np.mean(cv_results['test_roc_auc']),
            'ROC AUC Std': np.std(cv_results['test_roc_auc']),
            'Accuracy Mean': np.mean(cv_results['test_accuracy']),
            'Recall Mean': np.mean(cv_results['test_recall']),
            'F1 Mean': np.mean(cv_results['test_f1'])
          })

          try:
            pipeline.fit(X, y)
            # 获取特征选择器并使用它来筛选特征
            support_mask = pipeline.named_steps['feature_selection'].get_support()
            selected_features = X.columns[support_mask]
            selected_data = X[selected_features]
            # 保存被选中的特征及其数据
            selected_data.to_csv(os.path.join(output_folder, f'{name}_selected_features_data.csv'), index=False)
            print(f"Selected features data for {name} saved successfully.")
          except Exception as e:
            logging.error(f"Error processing model {name}: {e}", exc_info=True)
            print(f"Error processing model {name}: {e}")

          model_path = os.path.join(model_train_folder, f'{name}_model.joblib')
          dump(pipeline, model_path)
          print(f"{name}: Model saved to {model_path}. Mean ROC AUC = {np.mean(cv_results['test_roc_auc']):.4f}")
        except Exception as e:
          print(f"处理模型{name}时发生错误: {e}")

  evaluation_df = pd.DataFrame(evaluation_results)
  evaluation_df.to_csv(os.path.join(output_folder, 'model_evaluation_results.csv'), index=False)

  return results

# 3.可视化结果

# 修改可视化结果函数以接收额外的参数
def visualize_results(X, y, models, cv, output_folder):
    # 直接在此函数中调用 plot_roc_curves 和 plot_precision_recall_curves 函数时传入必要的参数
    plot_roc_curves(X, y, models, cv, output_folder)
    plot_precision_recall_curves(X, y, models, cv, output_folder)
    plot_performance_heatmap(results, output_folder) # 绘制热图比较各个模型的性能


# 1. 繪製jama格式的每个AUC图
def plot_model_roc_curve(X, y, model, model_name, cv, output_folder):
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)

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
          alpha=.8) #蓝色来表示平均ROC曲线
  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Chance', alpha=.8) #红色表示随机机会线（Chance line）

  # JAMA格式美化
  ax.set_xlabel('False Positive Rate', fontsize=14)
  ax.set_ylabel('True Positive Rate', fontsize=14)
  ax.set_title(f'ROC Curve for {model_name}', fontsize=16)
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
              x='Model', palette="Accent", orient="v")
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
  sns.barplot(x=list(mean_roc_auc_scores.keys()), y=list(mean_roc_auc_scores.values()), palette="tab10")
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

# 4.排序的平均ROC AUC分数柱状图
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


# 5. ROC曲线 ： ROC曲线比较了多个模型的性能，使用不同的颜色来区分每个模型是有帮助的。
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
    palette = plt.get_cmap('cividis')
    # tab10和tab20调色板提供了一组设计精良的颜色，适合区分多个类别或模型。这些颜色在科学可视化中使用广泛，因为它们在视觉上具有很好的区分度。
    # palette = plt.get_cmap('tab10')

    for i, (name, mean_auc, std_auc, mean_fpr, mean_tpr) in enumerate(model_aucs):
      plt.plot(mean_fpr, mean_tpr, color=palette(i / len(models)),
               label=f'{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate', fontsize=16, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, weight='bold')
    plt.title('ROC Curves Comparison', fontsize=20, weight='bold')
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
    palette = plt.get_cmap('Set2')  # 获取调色板

    for i, (name, ap_score, precision, recall) in enumerate(model_scores):
      plt.plot(recall, precision, color=palette(i), label=f'{name} (AP = {ap_score:.2f})', lw=2)

    plt.xlabel('Recall', fontsize=14, weight='bold')
    plt.ylabel('Precision', fontsize=14, weight='bold')
    plt.title('Precision-Recall Curves Comparison', fontsize=16, weight='bold')

    plt.legend(loc="upper right", fontsize=10, frameon=True, shadow=True) # 左下角插圖太大
    # 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(output_folder, 'precision_recall_curves_comparison.tiff'), format='tiff', dpi=300)
    plt.close()
#plt.legend(loc="lower left", fontsize=12): 添加图例，并将图例放置在左下角位置，指定图例字体大小为 12。

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
  sns.heatmap(performance_df, annot=True, cmap='RdYlBu', fmt=".2f", linewidths=.5, cbar_kws={'shrink': .5})

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

# 主程序
if __name__ == "__main__":
    df = pd.read_csv(data_path)# 加载数据集
    df, label_encoders = preprocess_data(df) # 数据预处理
    # 分离特征和目标变量，同时保留event列以供后续分析
    X = df.drop(['time', 'event', 'ID'], axis=1)  # 从特征集中排除'event、time'列
    y = df['event'] #设置目标变量
    event_data = df['time']  # 保留time列以供后续分析使用

    # StratifiedKFold ：将数据集划分为 k 个折叠（folds），并确保每个折叠中的类别分布与整个数据集中的类别分布相似。
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 在机器学习中，交叉验证是一种评估模型性能的技术，它通过将数据集分为训练集和测试集，并多次重复此过程来评估模型的性能。StratifiedKFold 是一种特殊的交叉验证方法，它在划分数据集时会尽量保持每个类别的样本比例相似，以确保模型在不同类别上的性能能够得到充分评估。
    # 使用 StratifiedKFold 可以有效地减少因为数据不均衡而引起的问题，特别是在分类问题中。通过保持每个类别的样本比例相似，可以更好地确保模型在每个类别上的预测性能。
    # 在使用 StratifiedKFold 进行交叉验证时，通常会将数据集划分为 k 个折叠，并在每个折叠上进行训练和测试。然后将每次训练的性能评估指标（如准确率、精确度、召回率等）进行平均，得到最终的评估结果。

    # 定义模型
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(solver='saga', max_iter=10000, tol=1e-2),   #数据集很大saga'求解器 、增加max_iter的值 、tol参数来指定收敛的容忍度。
        "Gradient Boosting": GradientBoostingClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "LDA": LinearDiscriminantAnalysis(),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(max_iter=2000),  # 增加最大迭代次数
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(verbose=0, iterations=100),
        "LGBMClassifier": LGBMClassifier(min_child_samples=20) # 控制叶子节点的最小数据量。 min_gain_to_split=0.01  ,No further splits with positive gain, best gain: -inf表示LightGBM在某个节点上没有找到任何可以提高模型性能的分割点。
       #共12个模型
    }

    results = train_and_save_models(X, y, models, skf, model_train_folder)
    visualize_results(X, y, models, skf, visualization_folder)

    for name, model in models.items():
      plot_model_roc_curve(X, y, model, name, skf, visualization_folder)

    # 调用新的排序和可视化函数
    plot_sorted_mean_roc_auc_scores_barplot(results, visualization_folder)
    plot_model_roc_auc_boxplot(results, visualization_folder)
    plot_mean_roc_auc_scores_barplot(results, visualization_folder)

