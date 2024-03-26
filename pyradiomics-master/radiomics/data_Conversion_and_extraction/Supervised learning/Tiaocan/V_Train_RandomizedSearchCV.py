import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# 加载数据集
train_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_LASSO_set.csv"
val_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_LASSO_set.csv"

train_set = pd.read_csv(train_set_path)
val_set = pd.read_csv(val_set_path)

# 分离特征和标签
X_train = train_set.drop(columns=['event', 'time', 'ID'])
y_train = train_set['event']
X_val = val_set.drop(columns=['event', 'time', 'ID'])
y_val = val_set['event']

# 定义pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LGBMClassifier(random_state=42))
])

# 定义参数分布

param_dist = {
    'classifier__num_leaves': sp_randint(10, 30),  # 减少叶子节点数
    'classifier__max_depth': sp_randint(2, 8),  # 减少最大深度
    'classifier__learning_rate': sp_uniform(0.01, 0.05),  # 减小学习率
    'classifier__n_estimators': sp_randint(50, 200),  # 减少树的数量
    'classifier__min_child_weight': sp_uniform(0.001, 0.05),
    'classifier__reg_alpha': sp_uniform(0.0, 1.0),  # L1正则化
    'classifier__reg_lambda': sp_uniform(0.0, 1.0),  # L2正则化
}

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(pipeline,
                                   param_distributions=param_dist,
                                   n_iter=100,
                                   cv=5,
                                   scoring='accuracy',
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

# 执行随机搜索
random_search.fit(X_train, y_train)

# 最佳参数和模型
print("Best parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

# 使用最佳模型在验证集上进行预测
# 使用最佳模型在训练集和验证集上进行预测
y_train_pred = best_model.predict(X_train)
y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
y_val_pred = best_model.predict(X_val)
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 计算并打印性能指标
print("Training set ROC AUC:", roc_auc_score(y_train, y_train_pred_proba))
print("Train_utils set ROC AUC:", roc_auc_score(y_val, y_val_pred_proba))
print("Training set accuracy:", accuracy_score(y_train, y_train_pred))
print("Train_utils set accuracy:", accuracy_score(y_val, y_val_pred))
print("Training set recall:", recall_score(y_train, y_train_pred))
print("Train_utils set recall:", recall_score(y_val, y_val_pred))

# Best parameters: {'classifier__learning_rate': 0.04129299578571182, 'classifier__max_depth': 3, 'classifier__min_child_weight': 0.026051994195762963, 'classifier__n_estimators': 118, 'classifier__num_leaves': 22, 'classifier__reg_alpha': 0.1629344270814297, 'classifier__reg_lambda': 0.07056874740042984}
# Training set ROC AUC: 0.99375
# Train_utils set ROC AUC: 0.9
# Training set accuracy: 0.9642857142857143
# Train_utils set accuracy: 0.8
# Training set recall: 0.925
# Train_utils set recall: 0.8
