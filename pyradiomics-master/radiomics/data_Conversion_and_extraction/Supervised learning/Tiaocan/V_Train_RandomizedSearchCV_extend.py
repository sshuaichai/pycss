from scipy.stats import randint as sp_randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import pandas as pd

# 加载数据集
train_set_path = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Data_set_partition\\train022_set.csv"
df_train = pd.read_csv(train_set_path)

# 分离特征和目标变量
X_train = df_train.drop(['event', 'time','ID'], axis=1)
y_train = df_train['event']

# 微调参数分布
param_dist = {
    'num_leaves': sp_randint(60, 75),  # 在找到的最佳参数周围微调
    'min_child_samples': sp_randint(15, 25),  # 基于原参数微调
    'n_estimators': sp_randint(200, 250),  # 微调树的数量
    'learning_rate': uniform(0.01, 0.03)  # 微调学习率
}

# 初始化模型
model = LGBMClassifier()

# 设置RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model,
                                    param_distributions=param_dist,
                                    n_iter=1000,  # 根据需要调整迭代次数
                                    scoring='roc_auc',
                                    cv=5,
                                    n_jobs=-1,
                                    verbose=1)

# 执行随机搜索
random_search.fit(X_train, y_train)

# 打印最佳参数和最佳分数
print("Best parameters found: ", random_search.best_params_)
print("Best ROC AUC score found: ", random_search.best_score_)

# Best parameters found:  {'learning_rate': 0.024993847715299486, 'min_child_samples': 16, 'n_estimators': 221, 'num_leaves': 61}
# Best ROC AUC score found:  0.7946886446886448
