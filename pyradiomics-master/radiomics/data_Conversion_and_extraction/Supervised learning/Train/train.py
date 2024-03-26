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

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from joblib import dump
import os
import logging
from visualize_results import visualize_results, plot_model_roc_curve, plot_model_roc_auc_boxplot, plot_mean_roc_auc_scores_barplot, plot_sorted_mean_roc_auc_scores_barplot, plot_roc_curves, plot_precision_recall_curves, plot_performance_heatmap

logging.basicConfig(filename='training_errors.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# 定义文件和输出路径  train_RF_mean_ / train_RF_median_   /Lasso_preprocessing_select_train_RF_参数_
'先定义参数！！！！预处理后的训练集'
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_train_lasso.csv"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output"
encoders_folder = os.path.join(output_folder, "train_lasso_encoders")  # Lasso_preprocessing_
model_train_folder = os.path.join(output_folder, "train_lasso_models")
visualization_folder = os.path.join(output_folder, "train_lasso_visualizations")

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


# 主程序
if __name__ == "__main__":
  df = pd.read_csv(data_path)  # 加载数据集
  df, label_encoders = preprocess_data(df)  # 数据预处理
  # 分离特征和目标变量，同时保留event列以供后续分析
  X = df.drop(['time', 'event','ID'], axis=1)  # 从特征集中排除'event、time'列
  y = df['event']  # 设置目标变量

  # StratifiedKFold ：将数据集划分为 k 个折叠（folds），并确保每个折叠中的类别分布与整个数据集中的类别分布相似。
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  # 在机器学习中，交叉验证是一种评估模型性能的技术，它通过将数据集分为训练集和测试集，并多次重复此过程来评估模型的性能。StratifiedKFold 是一种特殊的交叉验证方法，它在划分数据集时会尽量保持每个类别的样本比例相似，以确保模型在不同类别上的性能能够得到充分评估。
  # 使用 StratifiedKFold 可以有效地减少因为数据不均衡而引起的问题，特别是在分类问题中。通过保持每个类别的样本比例相似，可以更好地确保模型在每个类别上的预测性能。
  # 在使用 StratifiedKFold 进行交叉验证时，通常会将数据集划分为 k 个折叠，并在每个折叠上进行训练和测试。然后将每次训练的性能评估指标（如准确率、精确度、召回率等）进行平均，得到最终的评估结果。

  # 定义模型,训练模型尽量不调参
  # 设置随机种子，在初始化模型时，对于支持random_state参数的模型，确保设置一个固定的随机种子。
  models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=1000),  # 增加最大迭代次数，并设置随机种子
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42),
    "LGBMClassifier": LGBMClassifier(random_state=42)  # 共12个模型
  }

  results = train_and_save_models(X, y, models, skf, model_train_folder)
  visualize_results(X, y, models, skf, visualization_folder, results)

  for name, model in models.items():
    plot_model_roc_curve(X, y, model, name, skf, visualization_folder)

  # 调用新的排序和可视化函数
  plot_sorted_mean_roc_auc_scores_barplot(results, visualization_folder)
  plot_model_roc_auc_boxplot(results, visualization_folder)
  plot_mean_roc_auc_scores_barplot(results, visualization_folder)

