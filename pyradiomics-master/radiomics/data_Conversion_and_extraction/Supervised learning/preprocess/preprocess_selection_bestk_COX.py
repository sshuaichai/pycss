import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_set_RADIOMICS.csv"
df = pd.read_csv(data_path)

# 提取特征和标签
X = df.drop(['ID', 'time', 'event'], axis=1)  # 从数据框中删除ID、时间和事件列，剩下的作为特征
y = df['event']  # 事件列作为标签
time = df['time']  # 时间列

# 仅选择数值型列
X_numeric = X.select_dtypes(include=[np.number])  # 选择数据框中的数值型列
plt.figure(figsize=(10, 8))


# 计算相关性矩阵并去除高度相关的特征
corr_matrix = X_numeric.corr().abs()  # 计算特征之间的相关性矩阵
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # 取上三角矩阵来避免重复元素
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]  # 选择相关性大于0.9的特征进行删除(去除高度共綫性的特徵)
X_filtered = X_numeric.drop(columns=to_drop)  # 删除这些高度相关的特征

sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title('Feature Correlation Matrix Heatmap - Initial')
plt.savefig(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\initial_correlation_matrix_heatmap.tiff", format='tiff', dpi=300)
plt.close()

# 计算相关性矩阵
corr_matrix = X_numeric.corr()

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False) #不显示xy标签（特征名称）
plt.title('Feature Correlation Matrix Heatmap')
plt.savefig(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\correlation_matrix_heatmap.tiff", format='tiff', dpi=300)
plt.close()  # 关闭图形，避免重复显示

# 标准化处理
scaler = StandardScaler()  # 初始化标准化器
X_scaled = scaler.fit_transform(X_filtered)  # 对筛选后的特征进行标准化处理
X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered.columns)  # 将标准化后的数据转换为DataFrame

# 分割数据用于交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10折交叉验证

# 初始化最佳分数和k值
best_score = 0
best_k = 0

# 在循环开始前初始化列表
k_values = []
average_c_indexes = []

# 进行特征选择和交叉验证
for k in range(1, X_scaled_df.shape[1] + 1):  # 遍历所有可能的k值
    c_index_scores = []

    # 特征选择
    selector = SelectKBest(f_classif, k=k)  # 初始化特征选择器
    X_selected = selector.fit_transform(X_scaled_df, y)  # 选择k个最好的特征
    selected_features = X_scaled_df.columns[selector.get_support(indices=True)]  # 获取所选特征的名称

    # 在交叉验证循环中重新拟合模型
    for train_index, test_index in kf.split(X_scaled_df):
        # 使用选择的特征和对应的索引分割数据
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        time_train, time_test = time.iloc[train_index], time.iloc[test_index]
        """
        警告信息：ConvergenceWarning: Newton-Raphson failed to converge sufficiently. 这意味着在拟合Cox比例风险模型时，
        使用的Newton-Raphson迭代算法没有充分收敛。这可能是由于模型中存在问题，
        如数据不满足比例风险假设、存在高度共线性的特征、样本量太小或者模型过于复杂等。
        """
        # 拟合Cox模型，这次添加penalizer参数
        cph = CoxPHFitter(penalizer=0.5)  # 初始化Cox比例风险模型并添加惩罚项，  # 可以尝试增加这个值，比如0.5或更高，直到警告消失
        df_train = pd.DataFrame(X_train, columns=selected_features)
        df_train['time'] = time_train.values
        df_train['event'] = y_train.values
        cph.fit(df_train, duration_col='time', event_col='event')  # 拟合模型

        # 在测试集上计算C指数
        df_test = pd.DataFrame(X_test, columns=selected_features)
        df_test['time'] = time_test.values
        df_test['event'] = y_test.values
        c_index = cph.score(df_test, scoring_method="concordance_index")  # 计算C指数
        c_index_scores.append(c_index)

    # 计算平均C指数
    average_c_index = np.mean(c_index_scores)
    k_values.append(k)  # 将当前的k值添加到k_values列表中
    average_c_indexes.append(average_c_index)  # 将计算出的平均C指数添加到average_c_indexes列表中

    if average_c_index > best_score:
        best_score = average_c_index
        best_k = k


print(f"Best k: {best_k}, Best C-index: {best_score}")


plt.figure(figsize=(12, 7))  # 增加图表尺寸
# 使用Seaborn样式美化图表
sns.set(style="whitegrid")
# 绘制线图，使用不同的颜色和样式
plt.plot(k_values, average_c_indexes, marker='o', linestyle='-', color='royalblue', markersize=5, linewidth=2)
# 设置标题和坐标轴标签
plt.title('Average C-index for Different Numbers of Selected Features', fontsize=16)
plt.xlabel('Number of Features (k)', fontsize=14)
plt.ylabel('Average C-index', fontsize=14)
# 设置x轴的刻度间隔，假设我们每5个显示一个刻度
plt.xticks(ticks=np.arange(min(k_values), max(k_values)+1, 5))
# 设置网格线为更适合阅读的样式
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\Average_C-index.tiff", format='tiff', dpi=300)
plt.close()

# 假设cph模型已经拟合
coefficients = cph.params_

plt.figure(figsize=(12, 10))  # 增加图表尺寸以提供更多空间
sorted_indices = coefficients.sort_values().index
colors = ['skyblue' if x > 0 else 'lightcoral' for x in coefficients.sort_values()]  # 正系数使用蓝色，负系数使用红色
# 绘制条形图，同时为正负系数使用不同的颜色
plt.barh(range(len(coefficients)), coefficients.sort_values(), color=colors)
# 设置y轴的刻度标签为排序后的特征名称，调整字体大小以提高可读性
plt.yticks(range(len(coefficients)), ['']*len(coefficients), fontsize=8)
# 添加x轴和y轴的标签以及标题
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Feature Importance in Cox Proportional Hazards Model', fontsize=16)
# 添加网格线以便于阅读
plt.grid(True, which='both', linestyle='--', linewidth=0.5, axis='x')
# 优化布局以确保所有标签和标题都能完整显示
plt.tight_layout()
# 保存图表
plt.savefig(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\Feature_Importance_in_cox.tiff", format='tiff', dpi=300)
plt.close()

# 使用最佳k值进行特征选择并保存
selector = SelectKBest(f_classif, k=best_k).fit(X_scaled_df, y)  # 使用最佳k值重新进行特征选择
X_best = selector.transform(X_scaled_df)
selected_features_final = X_scaled_df.columns[selector.get_support(indices=True)]

X_best_df = pd.DataFrame(X_best, columns=selected_features_final)
X_best_df['time'] = time
X_best_df['event'] = y
# 将ID列添加回DataFrame
X_best_df['ID'] = df['ID'].values

# 计算最佳k值筛选特征的相关性矩阵
# 假设 X_best_df 是你已经选出的最佳特征组成的DataFrame
corr_matrix_best_k = X_best_df.drop(['time', 'event', 'ID'], axis=1).corr()

plt.figure(figsize=(14, 12))  # 维持图像尺寸
# 直接绘制热图，显示所有数值，不使用mask条件
sns.heatmap(corr_matrix_best_k, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={"shrink": .5},
            annot_kws={"size": 6})  # 维持注释的字体大小

plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=8)  # 维持x轴标签的旋转和字体大小
plt.yticks(fontsize=8)  # 维持y轴标签的字体大小
plt.title('Feature Correlation Matrix Heatmap - Selected Features')
plt.tight_layout()  # 确保图像边缘和标签完整显示
plt.savefig(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\selected_features_correlation_matrix_heatmap_cox.tiff",
            format='tiff', dpi=300, pil_kwargs={"compression": "tiff_jpeg"})
plt.close()



# 保存筛选后的特征到指定目录
output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\selected_features_cox.csv"
X_best_df.to_csv(output_path, index=False)  # 保存DataFrame到csv文件
print(f"Selected features saved to {output_path}")

# 0.88 10 0.4
# Best k: 28, Best C-index: 0.8635669885669884
# 0.88 10 0.5
# Best k: 28, Best C-index: 0.8635669885669884
# 0.88 10 0.6
#Best k: 28, Best C-index: 0.8432289932289934

# 0.87 10 0.7
# Best k: 14, Best C-index: 0.8320386678745813
# 0.87 10 0.6
# Best k: 26, Best C-index: 0.867030192030192
# 0.87 10 0.5
# Best k: 26, Best C-index: 0.867030192030192
# 0.87 10 0.4
# Best k: 26, Best C-index: 0.8639638139638139

# 0.86 10 0.4 no
# 0.86 10 0.5
# Best k: 24, Best C-index: 0.8604367854367855
# 0.86 10 0.6
# Best k: 24, Best C-index: 0.8616272616272618
# 0.86 10 0.7
# Best k: 14, Best C-index: 0.8320386678745813

# 0.85 10 0.8
# Best k: 23, Best C-index: 0.8472416472416473
# 0.85 10 0.7
# Best k: 23, Best C-index: 0.8472416472416473
# 0.85 10 0.6
# Best k: 23, Best C-index: 0.8395493395493396
# 0.85 10 0.5 no


# from lifelines import CoxPHFitter
# import gc  # 导入垃圾回收模块
#
# # 假设cph是已经训练的Cox模型
# cph = CoxPHFitter()
# # 这里进行模型训练等操作
#
# # 清除模型
# del cph
# gc.collect()  # 调用垃圾回收器
