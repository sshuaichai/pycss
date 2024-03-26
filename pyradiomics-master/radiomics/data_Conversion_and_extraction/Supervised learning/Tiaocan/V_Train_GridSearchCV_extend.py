import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# 数据集路径
train_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_0.88_set.csv"
val_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_0.88_set.csv"

# 加载数据集
train_set = pd.read_csv(train_set_path)
val_set = pd.read_csv(val_set_path)

# 分割特征和标签
X_train = train_set.drop(columns=['event', 'time', 'ID'])
y_train = train_set['event']
X_val = val_set.drop(columns=['event', 'time', 'ID'])
y_val = val_set['event']

# 定义管道，包括预处理（标准化）和分类器
pipeline = Pipeline([
    ('classifier', LogisticRegression(max_iter=1000))  # 确保收敛，适用于较大的数据集
])

# 调整参数网格
# 扩展参数网格，包括class_weight和更多C的值
param_grid = {
    'classifier__C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],  # 正则化强度的逆
    'classifier__penalty': ['l2'],  # 使用L2正则化
    'classifier__solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # 优化算法选择
    'classifier__class_weight': [None, 'balanced']  # 类别权重
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)

# 在训练集上执行网格搜索
grid_search.fit(X_train, y_train)

# 查看最佳参数
print("Best parameters:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 评估模型性能
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
print("Training set ROC AUC:", roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]))
print("Train_utils set ROC AUC:", roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]))
print("Training set accuracy:", accuracy_score(y_train, y_train_pred))
print("Train_utils set accuracy:", accuracy_score(y_val, y_val_pred))
print("Training set recall:", recall_score(y_train, y_train_pred))
print("Train_utils set recall:", recall_score(y_val, y_val_pred))

# 可以选择保存最佳模型
dump(best_model, r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\output\best_logistic_regression_model.joblib")
