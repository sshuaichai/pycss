import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

# 加载Cox模型
model_path = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\radiomics\\data_Conversion_and_extraction\\Supervised learning\\FeaturesOutput\\Train_Output\\multivariate_cox_model.pkl"
with open(model_path, 'rb') as f:
    cox_model = pickle.load(f)

# 加载包含RadScore的数据集
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\patients_with_rad_scores_and_selected_features.csv"
df = pd.read_csv(data_path)

# 假设df中有一列名为'Rad_score'包含RadScore，另一列名为'event'表示患者是否发生了感兴趣的事件（例如复发或死亡）
# 1表示发生事件，0表示未发生事件

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(df['event'], df['Rad_score'])

# 计算每个阈值的Youden指数
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

# 计算AUC
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f"Optimal RadScore Threshold: {optimal_threshold}")

# 使用最佳阈值将患者分为高低危组
df['Risk_Group'] = np.where(df['Rad_score'] >= optimal_threshold, 'High', 'Low')

# 保存带有风险分组的完整数据集
output_full_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\patients_with_full_risk_groups.csv"
df.to_csv(output_full_path, index=False)

# 创建并保存只包含time、event和风险分组的简化数据集
df_simplified = df[['time', 'event', 'Risk_Group']]
output_simplified_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\patients_with_simplified_risk_groups.csv"
df_simplified.to_csv(output_simplified_path, index=False)

print("Patients classified into risk groups and saved. Full and simplified datasets are saved.")
# Optimal RadScore Threshold: 4.39316942579165
