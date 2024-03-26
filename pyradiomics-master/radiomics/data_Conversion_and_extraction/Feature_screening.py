''
'''
特征选择（Feature Selection）、最佳特征：

特征选择是在特征已经提取或生成的基础上，从原始特征集中选择最重要或最相关的特征子集。
目的是降低模型的复杂性、减少特征的维度、提高模型的解释性，并有助于减少模型的过拟合。
特征选择方法可以基于统计测试、特征重要性评估、嵌入式方法（与模型一起训练）、包装式方法（通过尝试不同特征子集来评估性能）等。
例子包括方差阈值选择、互信息选择、基于树模型的特征选择、L1正则化等

vs.

特征处理（Feature Engineering）、降维：

特征处理是数据预处理的一部分，通常在训练模型之前执行。
它涉及到对原始数据进行转换、创建新的特征、缩放、归一化等操作，以使数据更适合用于机器学习模型。
目的是改善模型的性能，提高模型对数据的拟合能力，减少过拟合或欠拟合的可能性。
例子包括标准化（将特征缩放到相同的尺度）、独热编码（将分类变量转换为二进制向量）、多项式特征生成、PCA等。
特征处理通常不涉及特征的选择或减少，而是着重于特征的转换和增强。
'''
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE, SequentialFeatureSelector, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import os

# 定义文件和输出路径
file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\Data_set_partition\train_set.xlsx"  # 用訓練集筛选特徵
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\Feature_screening"  # 输出目录路径

# 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取Excel文件中的数据
df = pd.read_excel(file_path)
# 将所有特征列转换为数值型，无法转换的转为NaN
for col in df.columns[:-1]:  # 遍历除了最后一列（目标变量）之外的所有列
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(0, inplace=True)  # 将NaN值填充为0

# 确保目标变量列 'status' 存在并转换为数值型
if 'status' not in df.columns:
    raise ValueError("DataFrame 中没有找到 'status' 列，请检查列名。")
try:
    df['status'] = pd.to_numeric(df['status'], errors='raise')
except ValueError:
    raise ValueError("无法将 'status' 列转换为数值型，请检查数据。")

# 分离特征和目标变量
X = df.drop(columns=['status'])  # 特征数据
y = df['status']  # 目标变量

# 特征选择方法

## 1. 方差阈值法
def variance_threshold_selection(X):
    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))  # 设置方差阈值
    return X.columns[selector.fit(X).get_support()]  # 返回选中的特征
'''方差阈值法 (VarianceThreshold):
方差阈值（Variance Threshold）：这种方法适用于去除数据中方差较小（即信息量较少）的特征。
它不考虑目标变量，因此主要用于去除那些几乎没有变化的特征。'''

## 2. 基于模型的特征选择（提供几种可选模型）
def model_based_selection(X, y, model_type='RandomForest'):   # 使用随机森林模型
  if model_type == 'RandomForest':
    clf = RandomForestClassifier(n_estimators=100)  # 随机森林模型
  elif model_type == 'GradientBoosting':
    clf = GradientBoostingClassifier(n_estimators=100)  # 梯度提升树模型
  elif model_type == 'LinearSVC':
    clf = LinearSVC(max_iter=10000, dual=False)# 线性SVC模型      # 显式设置dual并增加 max_iter
  elif model_type == 'LassoCV':
    clf = LassoCV(cv=5)   # Lasso 回归。
  elif model_type == 'ExtraTrees':
    clf = ExtraTreesClassifier(n_estimators=100)# 基于树的集成模型。  与随机森林类似，但在构建每棵树时使用了更多的随机性，从而增加了树之间的差异性。
  else:
    raise ValueError(f"Unsupported model_type: {model_type}")

  selector = SelectFromModel(clf, threshold="median")  # 基于模型重要性选择特征
  selector.fit(X, y)
  return X.columns[selector.get_support()]

'''基于模型的特征选择 (SelectFromModel):
基于模型的特征选择（Model-Based Selection）：这种方法利用特定的机器学习算法的特性来选择特征，例如随机森林或Lasso回归。
这种方法适用于当你有预期的模型时，可以直接根据模型的重要性评分来选择特征。'''

## 3. 递归特征消除
def recursive_feature_elimination(X, y, model_type='LogisticRegression'):   # 使用逻辑回归模型
  if model_type == 'LogisticRegression':
    estimator = LogisticRegression(solver='liblinear')  # 逻辑回归模型
  elif model_type == 'RandomForest':
    estimator = RandomForestClassifier(n_estimators=100)
  elif model_type == 'GradientBoosting':
    estimator = GradientBoostingClassifier(n_estimators=100)  # 梯度提升树模型
  elif model_type == 'LinearSVC':
    estimator = LinearSVC(max_iter=10000, dual=False)  # 线性SVC模型
  elif model_type == 'ExtraTrees':
    estimator = ExtraTreesClassifier(n_estimators=100)  # 基于树的集成模型。
  else:
    raise ValueError(f"Unsupported model_type: {model_type}")

  rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)  # 选择的特征数量10 , 1步
  rfe.fit(X, y)
  return X.columns[rfe.support_]
'''递归特征消除 (RFE, Recursive Feature Elimination):
通过逐步剔除对模型影响最小的特征来选择特征。使用的模型是'逻辑回归模型'，通过重复训练和删除特征，找到最优的特征子集。
可以处理特征之间的相关性。
这种方法通过递归减少特征集的大小来选择特征。它适用于当你需要选择特征数量固定的情况。'''

## 4. 序列特征选择
def sequential_feature_selection(X, y):
    sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), n_features_to_select=10, direction='forward')  #10个特征  向后
    sfs.fit(X, y)
    return X.columns[sfs.get_support()]  # 返回选中的特征
'''序列特征选择 (SequentialFeatureSelector):
使用指定的模型（这里是K近邻分类器）进行序列特征选择。通过向前或向后的方法逐步添加或删除特征，选择最优的特征子集。
可以处理特征之间的相关性.
这种方法通过贪心算法逐步添加或删除特征，直到达到所需数量的特征。它适用于需要精细控制特征数量的情况。'''

## 5. Lasso回归
def lasso_selection(X, y):
    lasso = LassoCV(cv=5)  # 使用Lasso回归进行特征选择，’cv=5‘表示使用了 5 折交叉验证
    lasso.fit(X, y)
    return X.columns[lasso.coef_ != 0]  # 返回选中的特征
'''Lasso回归 (LassoCV):
使用Lasso回归进行特征选择，Lasso回归的特点是可以使得一部分特征的系数变为零，从而实现特征选择的目的。选取系数不为零的特征作为选中的特征。
可以处理特征之间的相关性。

在 5 折交叉验证中，数据集被分成 5 个相等的部分，其中一部分被保留作为验证集，剩余的部分被用来训练模型。这个过程被重复执行 5 次，
每次使用不同的验证集。最后，模型的性能指标通常是这 5 次交叉验证的平均值。
在 LassoCV 中，交叉验证被用来选择最优的正则化参数（也称为 alpha）。LassoCV 将尝试不同的 alpha 值，
并选择使交叉验证性能最佳的 alpha 值，以优化模型的性能。'''

## 6. 互信息
def mutual_information_selection(X, y):
    mi = mutual_info_regression(X, y)  # 计算特征和目标之间的互信息
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series.nlargest(10).index  # 返回互信息最高的10个特征
'互信息和相关性筛选：这种方法基于特征与目标变量之间的统计关系来选择特征，适用于发现与目标变量有强相关性的特征'

## 7.特征间相关性筛选
def correlation_feature_selection(X, threshold=0.95):#阈值＞0.8-0.95之间
    corr_matrix = X.corr().abs()  # 计算特征之间的相关性矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # 取上三角矩阵，避免重复计算
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]  # 找出高度相关的特征
    return X.drop(columns=to_drop, axis=1).columns  # 返回去除高度相关特征后的特征列表
'通过这种方式，你可以保留目标变量列，同时去除高度相关的特征，从而减少数据集中的冗余信息并可能提升模型的性能.'

# 应用特征选择方法并保存结果
methods = {
    '1. VarianceThreshold': variance_threshold_selection(X),
    '2. ModelBasedSelection': model_based_selection(X, y, 'RandomForest'),
    '3. RecursiveFeatureElimination': recursive_feature_elimination(X, y, 'LogisticRegression'),
    '4. SequentialFeatureSelection': sequential_feature_selection(X, y),
    '5. LassoSelection': lasso_selection(X, y),
    '6. MutualInformation': mutual_information_selection(X, y),
    '7. CorrelationFeatureSelection': correlation_feature_selection(X)
}

# 遍历方法并保存结果
for method_name, selected_features in methods.items():
    selected_data = X[selected_features] if method_name != '7. CorrelationFeatureSelection' else X.loc[:, selected_features]
    save_path = os.path.join(output_folder, f"{method_name}_Selected_Features.xlsx")
    selected_data.to_excel(save_path, index=False)
    print(f"{method_name}: Selected features saved to {save_path}")


'请提供完整的代码：根据上面代码选择逻辑回归、随机森林、支持向量机等作为评估对象，使用交叉验证（如k折交叉验证）来评估特征子集对模型性能的影响，评估使用不同特征子集的模型性能，包括准确率、召回率、F1分数等指标，并可视化这些评估指标。'

'''
如果你的目标是简化模型并减少计算成本，同时保持模型性能，可以考虑使用 → 2.基于模型的特征选择或3.RFE。
如果你需要探索数据，发现哪些特征与目标变量最相关，可以使用 → 6.互信息和7.相关性筛选。
对于高维数据（如影像组学数据），通常建议先使用1.方差阈值去除低方差特征，然后应用2.基于模型的方法或3.RFE进一步筛选特征。
'''


'''注意事项:
训练集：数据预处理（标准化）+数据特征提取（管道、交叉验证）+多模型训练和性能比较
当使用基于模型的特征选择方法时，特别是涉及到线性模型（如LinearSVC和LassoCV）时，确保数据已经进行了适当的预处理，比如标准化，
因为这些模型对输入数据的尺度敏感。
因此不需要对特征进行复杂的预处理。对于树形模型（如RandomForest、GradientBoosting和ExtraTrees），模型本身就能处理特征的非线性关系和交互作用，

在选择特征选择方法和模型时，考虑到模型的计算成本和特征的类型（例如，连续变量、分类变量）是很重要的。
不同的模型对不同类型的数据有不同的处理方式和效果'''
