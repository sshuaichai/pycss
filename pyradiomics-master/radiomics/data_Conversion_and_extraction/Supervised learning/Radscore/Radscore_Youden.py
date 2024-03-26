# import pandas as pd
# import joblib
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 为模拟"radiology"风格，我们选择了白底黑字的简洁风格，并增大字体大小以提高可读性
# # 设置图形风格plt.style.available查看所有可用的风格。
# # 常用的内置风格包括seaborn-whitegrid, seaborn-darkgrid, seaborn-ticks, classic, ggplot, 和 bmh 等
# plt.style.use('seaborn-whitegrid')  # 设置图形样式为 Seaborn 白色风格
# plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体族为 DejaVu Sans
# plt.rcParams['font.size'] = 12  # 设置字体大小为 14
# plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签大小为 16
# plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题大小为 18
# plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度标签大小为 14
# plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度标签大小为 14
# plt.rcParams['legend.fontsize'] = 12  # 设置图例（小框）字体大小为 14
# plt.rcParams['figure.figsize'] = [10, 8]  # 设置图形尺寸为 10x8 英寸
# plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边缘颜色为黑色
# plt.rcParams['axes.linewidth'] = 2  # 设置坐标轴线宽为 2
# plt.rcParams['legend.frameon'] = True  # 设置图例边框可见
# plt.rcParams['legend.framealpha'] = 0.5  # 设置图例边框透明度为 1（不透明）
#
# # 加载预先训练好的模型
# model_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_0.88_models\Random Forest_model.joblib"
# model = joblib.load(model_path)
#
# # 定义计算RadSCORE的函数
# def calculate_rad_score(data_path):
#     df = pd.read_csv(data_path)
#     X = df.drop(['time', 'event','ID'], axis=1)
#     # 使用模型预测每个样本的RadSCORE
#     rad_scores = model.predict_proba(X)[:, 1]
#     # 将RadSCORE添加到原始DataFrame中
#     df['RadSCORE'] = rad_scores
#     return df
#
# # 计算训练集、验证集和测试集的RadSCORE
# train_df = calculate_rad_score(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_0.88_set.csv")
# val_df = calculate_rad_score(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_0.88_set.csv")
# test_df = calculate_rad_score(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\test_0.88_set.csv")
#
# # 使用训练数据集的RadSCORE计算最佳阈值（最大化Youden指数）
# fpr, tpr, thresholds = roc_curve(train_df['event'], train_df['RadSCORE'])
# youden_index = tpr - fpr
# optimal_threshold = thresholds[youden_index.argmax()]
# print(f"最佳阈值为: {optimal_threshold}")
#
# # 根据最佳阈值分配风险组
# def assign_risk_group(df):
#     df['Risk_Group'] = ['High-risk' if score >= optimal_threshold else 'Low-risk' for score in df['RadSCORE']]
#     return df
#
# train_df = assign_risk_group(train_df)
# val_df = assign_risk_group(val_df)
# test_df = assign_risk_group(test_df)
#
# # 保存处理后的数据集
# train_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\train_RadSCORE_RiskGroup.csv", index=False)
# val_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\validation_RadSCORE_RiskGroup.csv", index=False)
# test_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\test_RadSCORE_RiskGroup.csv", index=False)
#
# # 合并数据集以便于可视化
# combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
# combined_df['DataSet'] = ['Train']*len(train_df) + ['Train_utils']*len(val_df) + ['Test']*len(test_df)
#
# # 可视化RadSCORE分布和风险组
# sns.scatterplot(data=combined_df, x='DataSet', y='RadSCORE', hue='Risk_Group', style='DataSet',
#                 palette={'High-risk': 'red', 'Low-risk': 'blue'}, markers={'Train': 'o', 'Train_utils': 'X', 'Test': 'D'}, s=150, alpha=0.8, edgecolor='k', linewidth=1.5)
#
# # 在图像中以横线标注最佳阈值
# plt.axhline(y=optimal_threshold, color='gray', linestyle='--', linewidth=2, label=f'Optimal Threshold: {optimal_threshold:.2f}')
#
# plt.title('RadSCORE Distribution by Risk Group')
# plt.xlabel('Dataset')
# plt.ylabel('RadSCORE')
#
# # 获取当前图形的轴线对象，并设置框线的颜色为黑色
# for spine in plt.gca().spines.values():
#     spine.set_edgecolor('gray')
#
# # 将图例移动到图形的右侧中间，并设置透明度
# plt.legend(title='Risk Group', loc='upper right', frameon=True, edgecolor='gray', framealpha=0.5)
#
# plt.tight_layout()
# # 保存可视化图形
# output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\Youden_RadSCORE_Distribution.tiff"
# plt.savefig(output_path, format='tiff', dpi=300, bbox_inches='tight')
# plt.close()
#
# print(f"RadSCORE分布可视化图已保存至 {output_path}")
