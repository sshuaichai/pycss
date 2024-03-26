from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

# 加载数据集
train_set_path = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Data_set_partition\\train022_set.csv"
df_train = pd.read_csv(train_set_path)

# 分离特征和目标变量
X = df_train.drop(['event', 'time','ID'], axis=1)
y = df_train['event']

model = LGBMClassifier(
  lambda_l1 = 0.0012569052447613503,
  lambda_l2 = 0.006229396018060933,
  learning_rate= 0.02002382030929063,
  min_child_samples=18,
  n_estimators=212,
  num_leaves=25
)

# 执行交叉验证
# 这里使用ROC AUC作为评分指标，进行5折交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

# 打印每折的得分以及平均得分
print("ROC AUC scores for each fold are: ", scores)
print("Mean ROC AUC score: ", np.mean(scores))
print("Standard deviation of ROC AUC scores: ", np.std(scores))


