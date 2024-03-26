import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# 加载数据集
train_set_path = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Data_set_partition\\train022_set.csv"
df_train = pd.read_csv(train_set_path)

# 分离特征和目标变量
X = df_train.drop(['event', 'time','ID'], axis=1)
y = df_train['event']

# 将已经划分的训练数据集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机搜索得到的参数初始化模型
model = LGBMClassifier(
    num_leaves=41,
    min_child_samples=16,
    n_estimators=392,
    learning_rate=0.017994816064535703,
    reg_alpha=0.5,
    reg_lambda=0.5,
    subsample=0.8,
    colsample_bytree=0.8
)

# 训练模型，使用早停
from lightgbm import early_stopping

# 训练模型，使用早停的callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
)

# 检查是否使用了早停，如果使用了早停，打印最佳迭代次数
if model.best_iteration_:
    print(f"Best iteration: {model.best_iteration_}")

# 打印最佳模型的参数
print("Best model parameters: ", model.get_params())

# 使用最佳迭代次数（如果适用）在验证集上进行预测
if model.best_iteration_:
    y_pred = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
else:
    y_pred = model.predict_proba(X_val)[:, 1]

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score

# 计算并打印性能指标
roc_auc = roc_auc_score(y_val, y_pred)
accuracy = accuracy_score(y_val, (y_pred > 0.5).astype(int))
recall = recall_score(y_val, (y_pred > 0.5).astype(int))
f1 = f1_score(y_val, (y_pred > 0.5).astype(int))

print(f"Validation ROC AUC Score: {roc_auc}")
print(f"Validation Accuracy: {accuracy}")
print(f"Validation Recall: {recall}")
print(f"Validation F1 Score: {f1}")

# Early stopping, best iteration is:
# [93]	valid_0's auc: 0.786667	valid_0's binary_logloss: 0.507739
# Best iteration: 93
# Best model parameters:  {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.8, 'importance_type': 'split', 'learning_rate': 0.017994816064535703, 'max_depth': -1, 'min_child_samples': 16, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 392, 'n_jobs': None, 'num_leaves': 41, 'objective': None, 'random_state': None, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'subsample': 0.8, 'subsample_for_bin': 200000, 'subsample_freq': 0}
# Train_utils ROC AUC Score: 0.7866666666666667
# Train_utils Accuracy: 0.9
# Train_utils Recall: 0.6
# Train_utils F1 Score: 0.75

