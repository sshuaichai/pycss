import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pickle
import os

# 加载包含分组信息的数据集
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\patients_with_simplified_risk_groups.csv"
df = pd.read_csv(data_path)

# 检查数据分布和数据质量：
# 检查风险组别的样本数量
print(df['Risk_Group'].value_counts())
# 检查事件发生的比例
print(df.groupby('Risk_Group')['event'].mean())
# 检查是否有缺失值
print(df.isnull().sum())
# 检查time列的统计摘要，寻找可能的异常值
print(df['time'].describe())
"""
风险组别的样本不平衡：高危组有20个样本，而低危组有78个样本。这种不平衡可能不是导致模型拟合问题的直接原因，但它确实表明两组的风险可能有很大差异。

事件发生率差异显著：高危组的事件发生率为95%，而低危组的事件发生率仅为20.5%。这表明高危组和低危组在生存时间上可能有显著差异，这是好的，因为我们希望模型能够捕捉到这种差异。然而，这也意味着模型的预测变量（在这里是Risk_Group_Num）与生存时间高度相关，这通常是期望的情况。

没有缺失值：time、event和Risk_Group列中没有缺失值，这意味着数据质量问题不是导致模型拟合失败的原因。

time列的统计摘要：生存时间的范围从2到34.5个单位，平均生存时间约为11个单位，标准差约为6.84。这表明生存时间数据具有一定的变异性，但没有明显的异常值。
"""

# # 将风险组别转换为数值型变量（高危为1，低危为0）
# df['Risk_Group_Num'] = df['Risk_Group'].apply(lambda x: 1 if x == 'High' else 0)
#
# # 初始化CoxPHFitter对象并拟合模型
# cph = CoxPHFitter()
# cph.fit(df[['time', 'event', 'Risk_Group_Num']], 'time', event_col='event')
#
# # 指定输出目录
# output_dir = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Output"
#
# # 保存Cox模型摘要到文本文件
# summary_path = os.path.join(output_dir, "cox_model_summary.txt")
# with open(summary_path, 'w') as f:
#     f.write(cph.summary.to_string())
#
# # 使用KaplanMeierFitter绘制生存曲线并保存图像
# kmf = KaplanMeierFitter()
# fig, ax = plt.subplots(figsize=(10, 6))
#
# for name, grouped_df in df.groupby('Risk_Group'):
#     kmf.fit(grouped_df['time'], grouped_df['event'], label=name)
#     kmf.plot_survival_function(ax=ax)
#
# plt.title('Kaplan-Meier Survival Curves by RadScore Groups')
# plt.xlabel('Time')
# plt.ylabel('Survival Probability')
# plt.savefig(os.path.join(output_dir, "km_survival_curves.png"))
#
# # 使用logrank_test比较生存曲线，并保存结果
# results = logrank_test(df[df['Risk_Group'] == 'High']['time'], df[df['Risk_Group'] == 'Low']['time'],
#                        event_observed_A=df[df['Risk_Group'] == 'High']['event'],
#                        event_observed_B=df[df['Risk_Group'] == 'Low']['event'])
#
# results_summary = f"Log-rank Test p-value: {results.p_value}\n"
# hr = cph.hazard_ratios_['Risk_Group_Num']
# p_value = cph.summary.loc['Risk_Group_Num', 'p']
# results_summary += f"Hazard Ratio (HR) for High vs. Low Risk Group: {hr}\np-value for HR: {p_value}"
#
# results_path = os.path.join(output_dir, "cox_model_results.txt")
# with open(results_path, 'w') as f:
#     f.write(results_summary)
#
# # 使用pickle保存拟合好的Cox模型
# model_path = os.path.join(output_dir, "fitted_cox_model.pkl")
# with open(model_path, 'wb') as f:
#     pickle.dump(cph, f)
#
# print("Model summary, KM survival curves, results, and fitted model have been saved to the specified directory.")
