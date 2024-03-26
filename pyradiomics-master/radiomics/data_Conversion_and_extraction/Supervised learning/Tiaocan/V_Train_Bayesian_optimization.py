import optuna  # 导入Optuna库，用于自动化超参数优化
import pandas as pd  # 导入pandas库，用于数据处理和读取
from lightgbm import LGBMClassifier  # 从LightGBM库导入LGBMClassifier，一个轻量级梯度提升框架
from sklearn.metrics import roc_auc_score  # 从sklearn.metrics导入roc_auc_score，用于评估模型性能
from sklearn.model_selection import train_test_split  # 导入train_test_split，用于拆分训练集和测试集

# 加载数据集
df_train = pd.read_csv("D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Data_set_partition\\train022_set.csv")
X = df_train.drop(['event', 'time','ID'], axis=1)
y = df_train['event']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 基于你的历史评估结果和当前最佳参数，使用贝叶斯优化进一步评估和优化模型参数的方法如下：
# learning_rate = 0.024993847715299486,
# min_child_samples = 16,
# n_estimators = 221,
# num_leaves = 61

# 1. 定义搜索空间
# 首先，围绕你已经找到的最佳参数，定义一个合适的搜索空间。由于贝叶斯优化能够更智能地探索参数空间，你可以在当前最佳参数的基础上设置一个更细致的搜索范围。
# 例如，如果当前最佳的learning_rate是0.024993847715299486，你可以在其附近定义一个范围进行搜索。
def objective(trial):
  # 定义一个搜索空间，为模型的几个关键超参数设置范围
  param = {
    'objective': 'binary',  # 设置任务类型为二分类
    'metric': 'auc',  # 评价指标为AUC
    'verbosity': -1,  # 设置运行时的信息显示等级，-1表示不输出信息
    'boosting_type': 'gbdt',  # 设置提升类型为传统的梯度提升决策树
    # 使用trial对象提供的方法为以下参数设定搜索范围
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),  # 学习率，从0.01到0.03之间选择
    'num_leaves': trial.suggest_int('num_leaves', 50, 70),  # 叶子节点数，从50到70之间选择
    'min_child_samples': trial.suggest_int('min_child_samples', 10, 20),  # 每个叶子的最小样本数，从10到20之间选择
    'n_estimators': trial.suggest_int('n_estimators', 200, 240),  # 建立树的数量，从200到240之间选择
  }

  model = LGBMClassifier(**param)  # 使用上面定义的参数初始化LGBM分类器
  model.fit(X_train, y_train)  # 训练模型
  preds = model.predict_proba(X_test)[:, 1]  # 对测试集进行概率预测
  roc_auc = roc_auc_score(y_test, preds)  # 计算并返回AUC分数
  return roc_auc

# 2.使用 optuna 进行贝叶斯优化，寻找最佳的 LGBMClassifier 参数配置

# 创建一个Optuna研究对象，目标是最大化目标函数
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # 执行50次试验来寻找最佳参数

print("Best trial:")
trial = study.best_trial  # 获取最佳试验结果

print("  Value: ", trial.value)  # 打印最佳试验的AUC分数
print("  Params: ")  # 打印最佳试验的参数
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))  # 循环打印参数名和对应的最佳值


# Best trial:
#   Value:  0.8400000000000001
#   Params:
#     learning_rate: 0.019224804534297986
#     num_leaves: 65
#     min_child_samples: 18
#     n_estimators: 237
# [I 2024-03-01 14:17:27,482] Trial 99 finished with value: 0.7866666666666666 and parameters: {'learning_rate': 0.0216827019815265, 'num_leaves': 68, 'min_child_samples': 13, 'n_estimators': 239}. Best is trial 26 with value: 0.8400000000000001.
