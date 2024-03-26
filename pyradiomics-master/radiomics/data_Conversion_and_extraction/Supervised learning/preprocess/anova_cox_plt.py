import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt

# 加载预处理和标准化后的数据
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\processed_data.csv"
df = pd.read_csv(data_path)

X_scaled_df = df.drop(['time', 'event', 'ID'], axis=1)
y = df['event']
time = df['time']

# 分割数据用于交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化最佳分数和k值
best_score = 0
best_k = 0

# 在循环开始前初始化列表
k_values = []
average_c_indexes = []

# 进行特征选择和交叉验证
for k in range(1, X_scaled_df.shape[1] + 1):
    c_index_scores = []

    # 特征选择
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled_df, y)
    selected_features = X_scaled_df.columns[selector.get_support(indices=True)]

    # 在交叉验证循环中重新拟合模型
    for train_index, test_index in kf.split(X_scaled_df):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        time_train, time_test = time.iloc[train_index], time.iloc[test_index]

        # 拟合Cox模型
        cph = CoxPHFitter(penalizer=0.5)
        df_train = pd.DataFrame(X_train, columns=selected_features)
        df_train['time'] = time_train
        df_train['event'] = y_train
        cph.fit(df_train, 'time', 'event')

        # 在测试集上计算C指数
        df_test = pd.DataFrame(X_test, columns=selected_features)
        df_test['time'] = time_test
        df_test['event'] = y_test
        c_index = cph.score(df_test, scoring_method="concordance_index")
        c_index_scores.append(c_index)

    # 计算平均C指数
    average_c_index = np.mean(c_index_scores)
    k_values.append(k)
    average_c_indexes.append(average_c_index)

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
