import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# 加载数据集
df_train = pd.read_csv("D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Data_set_partition\\train022_set.csv")
X = df_train.drop(['event', 'time','ID'], axis=1)
y = df_train['event']

# 定义目标函数
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),
        'num_leaves': trial.suggest_int('num_leaves', 20, 40),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 30),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
    }

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, preds)
    return roc_auc

# 创建研究对象并执行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 打印最佳试验结果
best_trial = study.best_trial

# 使用最佳参数重新创建模型并执行5折交叉验证
best_params = best_trial.params
model = LGBMClassifier(**best_params)

scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

# 打印交叉验证的结果
print("\nCross-validation results:")
print("ROC AUC scores for each fold are: ", scores)
print("Mean ROC AUC score: ", np.mean(scores))
print("Standard deviation of ROC AUC scores: ", np.std(scores))
