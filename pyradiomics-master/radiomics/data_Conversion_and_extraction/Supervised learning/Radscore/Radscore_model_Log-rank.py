
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns


# 为模拟"radiology"风格，我们选择了白底黑字的简洁风格，并增大字体大小以提高可读性
# 设置图形风格plt.style.available查看所有可用的风格。
# 常用的内置风格包括seaborn-whitegrid, seaborn-darkgrid, seaborn-ticks, classic, ggplot, 和 bmh 等
plt.style.use('seaborn-whitegrid')  # 设置图形样式为 Seaborn 白色风格
plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体族为 DejaVu Sans
plt.rcParams['font.size'] = 12  # 设置字体大小为 14
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签大小为 16
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题大小为 18
plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度标签大小为 14
plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度标签大小为 14
plt.rcParams['legend.fontsize'] = 12  # 设置图例（小框）字体大小为 14
plt.rcParams['figure.figsize'] = [10, 8]  # 设置图形尺寸为 10x8 英寸
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边缘颜色为黑色
plt.rcParams['axes.linewidth'] = 2  # 设置坐标轴线宽为 2
plt.rcParams['legend.frameon'] = True  # 设置图例边框可见
plt.rcParams['legend.framealpha'] = 0.5  # 设置图例边框透明度为 1（不透明）

# 加载预先训练好的模型
model_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_LOGIS_models\Logistic Regression_model.joblib"
model = joblib.load(model_path)

# 定义计算RadSCORE的函数
def calculate_rad_score(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(['time', 'event','ID'], axis=1)
    # 使用模型预测每个样本的RadSCORE
    rad_scores = model.predict_proba(X)[:, 1]
    # 将RadSCORE添加到原始DataFrame中
    df['RadSCORE'] = rad_scores
    return df

# 计算训练集、验证集和测试集的RadSCORE
train_df = calculate_rad_score(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_LOGIS_set.csv")
val_df = calculate_rad_score(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_LOGIS_set.csv")
test_df = calculate_rad_score(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\test_LOGIS_set.csv")

# 使用训练数据集的RadSCORE计算基于交叉验证和对数秩检验（Log-rank Test）寻找最佳阈值
def find_best_threshold_with_cv(data, features, n_splits=5):# 通过增加交叉验证的折数（n_splits）来提高模型的泛化能力，这通常比单纯增加搜索的轮数来得更有效。
  """
  使用交叉验证找到最佳阈值。
  返回:
  - final_best_threshold: 所有交叉验证折中平均最佳阈值。
  对数秩检验是一种非参数统计测试，用于比较两个样本的生存时间分布是否存在显著差异。在此方法中，选择使得对数秩检验的P值最小的阈值作为最佳阈值。
  P值衡量的是在零假设（两组生存时间分布相同）下观察到的数据（或更极端的数据）出现的概率。较小的P值意味着两组之间存在显著差异。
  结合了交叉验证的稳健性和对数秩检验的统计意义，旨在找到一个能够最有效地区分高风险和低风险患者群体的RadSCORE阈值。通过这种方法得到的阈值可以最大化模型对于生存分析的预测准确性。
  """
  skf = StratifiedKFold(n_splits=n_splits)  # 初始化分层k折交叉验证
  best_thresholds = []  # 存储每一折中找到的最佳阈值
  best_ps = []  # 存储每一折中最佳阈值对应的p值


  # 根据交叉验证索引分割数据
  for train_index, test_index in skf.split(data[features], data['event']):
    train_data = data.iloc[train_index].copy()
    test_data = data.iloc[test_index].copy()

    # 确定用于预测的特征列，确保不包括后来添加的`RadSCORE`列
    predict_features = [col for col in train_data.columns if col not in ['time', 'event', 'ID', 'RadSCORE']]

    # 使用确定的特征列进行预测
    # 计算训练集和测试集的RadSCORE
    train_scores = model.predict_proba(train_data[predict_features])[:, 1]
    test_scores = model.predict_proba(test_data[predict_features])[:, 1]
    train_data['RadSCORE'] = train_scores
    test_data['RadSCORE'] = test_scores

    best_p = 1  # 初始化最佳p值
    best_threshold = 0  # 初始化最佳阈值
    for threshold in np.linspace(train_data['RadSCORE'].min(), train_data['RadSCORE'].max(), 500):# 通过初始较少的轮数进行测试，观察阈值是否稳定，然后逐渐增加轮数直到阈值稳定。
      # 对每个阈值，根据RadSCORE将训练数据分为高低风险组
      train_data['RiskGroup'] = np.where(train_data['RadSCORE'] > threshold, 'High', 'Low')

      # 使用对数秩检验评估高低风险组的生存差异
      high_risk = train_data[train_data['RiskGroup'] == 'High']
      low_risk = train_data[train_data['RiskGroup'] == 'Low']
      results = logrank_test(high_risk['time'], low_risk['time'], event_observed_A=high_risk['event'],
                             event_observed_B=low_risk['event'])
      p = results.p_value

      # 更新最佳阈值和对应的p值
      if p < best_p:
        best_p = p
        best_threshold = threshold

    best_thresholds.append(best_threshold)
    best_ps.append(best_p)

    # 计算所有交叉验证折中最佳阈值的平均值作为最终最佳阈值
  final_best_threshold = np.mean(best_thresholds)
  print(f"Final Best Threshold: {final_best_threshold}, Average P-Value: {np.mean(best_ps):.16f}")
  return final_best_threshold

features = train_df.drop(columns=['ID', 'event', 'time']).columns.tolist()
# 执行寻找最佳阈值的函数
best_threshold = find_best_threshold_with_cv(train_df, features=features, n_splits=8)



# 根据最佳阈值分配风险组
def assign_risk_group(df, best_threshold):
  df['Risk_Group'] = ['High-risk' if score >= best_threshold else 'Low-risk' for score in df['RadSCORE']]
  return df

train_df = assign_risk_group(train_df, best_threshold)
val_df = assign_risk_group(val_df, best_threshold)
test_df = assign_risk_group(test_df, best_threshold)

# 保存处理后的数据集
train_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\train_RadSCORE_RiskGroup.csv", index=False)
val_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\validation_RadSCORE_RiskGroup.csv", index=False)
test_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\test_RadSCORE_RiskGroup.csv", index=False)

# 合并数据集以便于可视化
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
combined_df['DataSet'] = ['Train']*len(train_df) + ['Train_utils']*len(val_df) + ['Test']*len(test_df)

# 保存合并后的数据集到指定位置
combined_csv_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\combined_RadSCORE_RiskGroup.csv"
combined_df.to_csv(combined_csv_path, index=False)

print(f"合并后的RadSCORE分布数据集已保存至 {combined_csv_path}")


# 可视化RadSCORE分布和风险组
sns.scatterplot(data=combined_df, x='DataSet', y='RadSCORE', hue='Risk_Group', style='DataSet',
                palette={'High-risk': 'red', 'Low-risk': 'blue'},
                markers={'Train': 'o', 'Train_utils': 'X', 'Test': 'D'},
                s=150, alpha=0.8, edgecolor='k', linewidth=1.5)

# 在图像中以横线标注最佳阈值
plt.axhline(y=best_threshold, color='gray', linestyle='--', linewidth=2, label=f'Optimal Threshold: {best_threshold:.2f}')

plt.title('RadSCORE Distribution by Risk Group')
plt.xlabel('Dataset')
plt.ylabel('RadSCORE')

# 获取当前图形的轴线对象，并设置框线的颜色为黑色
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')

# 将图例移动到图形的右侧中间，并设置透明度
plt.legend(title='Risk Group', loc='upper right', frameon=True, edgecolor='gray', framealpha=0.5)

plt.tight_layout()
# 保存可视化图形
output_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Output\Log-rank_RadSCORE_Distribution.tiff"
plt.savefig(output_path, format='tiff', dpi=300, bbox_inches='tight')
plt.close()

print(f"RadSCORE分布可视化图已保存至 {output_path}")

'根据最佳阈值logrank_test的p最小的radscore阈值划分高低危分组，应该在建立cox模型之后，带入最佳的cox模型来分组？'

"""
建立Cox模型：首先，基于您的数据集建立一个Cox比例风险模型，该模型可能包括RadSCORE以及其他临床或生物标志物作为协变量。
模型验证：通过内部验证（如交叉验证）和/或外部验证确保模型的稳健性和预测能力。
使用模型确定最佳阈值：利用建立的Cox模型，通过改变RadSCORE的阈值，并使用对数秩检验（log-rank test）来比较不同阈值下的高低危组生存曲线差异，从而找到使得生存差异最大（即log-rank test的p值最小）的RadSCORE阈值。
高低危分组：根据第3步确定的最佳RadSCORE阈值，将患者分为高危和低危两组。
进一步分析：可以进一步分析高低危组在临床结果上的差异，或者使用这个分组作为一个协变量在其他模型中进行分析。
这个过程中，确定最佳阈值的步骤是在建立了Cox模型之后进行的，因为您需要模型来评估不同RadSCORE阈值下的生存差异。
通过这种方式，您可以确保高低危分组是基于模型预测生存概率的差异，这样的分组对于预测患者的预后更有意义。
"""
