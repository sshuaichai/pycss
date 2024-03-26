import pandas as pd
from lifelines import CoxPHFitter
import pickle

# 加载数据
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\selected_features_cox.csv"
df = pd.read_csv(data_path)

# 初始化CoxPHFitter对象
cph = CoxPHFitter()

# 存储显著的特征
significant_features_univariate = []

# 准备保存单变量分析结果的字符串
univariate_results = "Feature,P-value\n"

# 对每个特征进行单变量Cox回归分析
for feature in df.columns.drop(['time', 'event', 'ID']):  # 假设数据中的时间、事件和ID列不参与模型拟合
    # 使用当前特征进行分析
    cph.fit(df[['time', 'event', feature]].dropna(), duration_col='time', event_col='event')
    # 获取p值
    p_value = cph.summary.loc[feature, 'p']
    # 保存结果
    univariate_results += f"{feature},{p_value}\n"
    # 检查p值，如果小于0.1，则认为该特征显著
    if p_value < 0.1:
        significant_features_univariate.append(feature)

# 保存单变量分析结果到文件
univariate_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\univariate_cox_analysis.csv"
with open(univariate_path, "w") as f:
    f.write(univariate_results)

# 使用单变量分析筛选出的特征进行多变量Cox回归分析
if significant_features_univariate:
    cph.fit(df[['time', 'event'] + significant_features_univariate].dropna(), duration_col='time', event_col='event')
    # 进一步筛选在多变量模型中显著的特征
    significant_features_multivariate = cph.summary[cph.summary['p'] < 0.1].index.tolist()
    # 保存多变量分析结果到文件
    multivariate_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\multivariate_cox_analysis.csv"
    cph.summary.to_csv(multivariate_path)
    print("Multivariate CoxPH Model Summary saved.")
else:
    print("No significant features found for multivariate analysis.")

# 保存多变量分析中显著特征的列表
significant_features_multivariate_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\significant_features_multivariate.csv"
with open(significant_features_multivariate_path, "w") as f:
    f.write("\n".join(significant_features_multivariate))
print("Significant features from multivariate analysis saved.")

# 计算每位患者的放射组学评分（Rad评分），仅使用多变量分析中显著的特征
if significant_features_multivariate:
    # 获取模型系数
    coefficients = cph.params_[significant_features_multivariate]
    # 计算Rad评分
    df['Rad_score'] = df[significant_features_multivariate].dot(coefficients)
    # 保存包含Rad评分的数据集，仅包含多变量分析中筛选出的特征
    df_subset = df[['time', 'event', 'ID', 'Rad_score'] + significant_features_multivariate]
    rad_score_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\patients_with_rad_scores_and_selected_features.csv"
    df_subset.to_csv(rad_score_path, index=False)
    print("Rad scores and selected features dataset saved.")
else:
    print("No significant features, Rad scores calculation skipped.")

# 使用pickle保存模型
model_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\multivariate_cox_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(cph, f)
print("Multivariate CoxPH Model saved.")
