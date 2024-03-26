import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 数据集路径
train_set_path =r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_0.88_set.csv"
val_set_path =r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_0.88_set.csv"

# 加载数据集
train_set = pd.read_csv(train_set_path)
val_set = pd.read_csv(val_set_path)

# 分割特征和标签
# 假设 'event' 是标签列的名称，同时排除 'time' 和 'ID' 列
X_train = train_set.drop(columns=['event', 'time', 'ID'])
y_train = train_set['event']
X_val = val_set.drop(columns=['event', 'time', 'ID'])
y_val = val_set['event']

# 定义管道，包括预处理（标准化）和分类器
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 特征标准化
    ('classifier', LogisticRegression(max_iter=5000)),
     # 确保收敛，适用于较大的数据集
])

# 更新的参数网格（减少过拟合，即验证集接近训练集ROC）
# 细化正则化强度C的搜索范围
param_grid = {
    'classifier__C': [0.02, 0.025, 0.03, 0.035, 0.04],  # 细化C值的搜索范围
    'classifier__penalty': ['l2'],  # 保持最佳惩罚项
}


# 创建 GridSearchCV 对象，将管道传递给它
grid_search = GridSearchCV(pipeline,
                           param_grid,
                           cv=5,
                           scoring='roc_auc',  # 使用 ROC AUC 作为评分标准
                           verbose=1)

# 在训练集上执行网格搜索
grid_search.fit(X_train, y_train)

# 查看最佳参数
print("Best parameters:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在训练集和验证集上评估模型性能
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)

# 输出 ROC AUC 和准确率指标
print("Training set ROC AUC:", roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]))
print("Train_utils set ROC AUC:", roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]))
print("Training set accuracy:", accuracy_score(y_train, y_train_pred))
print("Train_utils set accuracy:", accuracy_score(y_val, y_val_pred))
