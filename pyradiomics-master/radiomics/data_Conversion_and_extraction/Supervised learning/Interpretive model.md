''
'解释模型shape值/...'
''


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import os
import bisect  # Add this import statement at the top of your script

# 定义文件路径
model_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\Part_2_.1\optimized_Random_Forest_model.joblib"
test_set_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\Data_set_partition\test_set.xlsx"
encoders_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\Part_1\encoders"
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\Part_3"

# 确保输出目录存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载测试数据
test_data = pd.read_excel(test_set_path)
X_test = test_data.drop('status', axis=1)
y_test = test_data['status']

# 对测试数据应用LabelEncoder
# 假设我们已经加载了测试集到X_test
import numpy as np
import bisect  # Ensure this import is at the top of your script
# 将未知标签替换为一个通用标签
# 将所有未知标签替换为一个已知的“通用”标签，比如'unknown'。这要求在训练集的LabelEncoder学习阶段就加入这个通用标签，以确保模型能够处理未知类别的情况。
# Assuming you're in the loop where you're handling categorical features with LabelEncoder
for column in X_test.columns:
    if X_test[column].dtype == 'object':
        encoder_path = os.path.join(encoders_folder, f"{column}_encoder.pkl")
        if os.path.exists(encoder_path):
            le = joblib.load(encoder_path)
            # 将测试集中的标签映射到已知标签
            X_test[column] = X_test[column].apply(lambda x: x if x in le.classes_ else 'unknown')
            # 使用 LabelEncoder 转换标签
            # 首先将未知标签 'unknown' 添加到 classes_ 中（如果还没有的话）
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            X_test[column] = le.transform(X_test[column])


# 应用StandardScaler
scaler_path = os.path.join(encoders_folder, 'scaler.joblib')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
else:
    print("Scaler file not found. Make sure the scaler has been saved correctly.")
    X_test_scaled = X_test  # 如果找不到scaler，使用未缩放的数据

# 加载模型
model = joblib.load(model_path)

# 进行预测
predictions_proba = model.predict_proba(X_test_scaled)[:, 1]

# 计算ROC AUC
roc_auc = roc_auc_score(y_test, predictions_proba)
print(f"ROC AUC: {roc_auc}")

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, predictions_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 保存ROC曲线图像
roc_curve_path = os.path.join(output_folder, 'roc_curve.png')
plt.savefig(roc_curve_path)
plt.close()

print(f"ROC curve saved to: {roc_curve_path}")

import shap

# 确保SHAP已经安装
# pip install shap
# 初始化SHAP解释器
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# 生成并保存SHAP图形
# 总结图（Summary Plot）
# 确保在调用shap.summary_plot时使用show=False参数，这样可以防止图形在保存前被显示（这可能导致图形内容丢失）。
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
summary_path = os.path.join(output_folder, 'summary_plot.png')
plt.savefig(summary_path)
plt.close()
print(f"Summary plot saved to: {summary_path}")

# 力量图（Force Plot）- 需要使用SHAP的保存方法
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test_scaled[0,:], show=False, matplotlib=True)
force_path = os.path.join(output_folder, 'force_plot.png')
plt.savefig(force_path)
plt.close()
print(f"Force plot saved to: {force_path}")

'由于多输出模型，选择一个类别的SHAP值进行展示'
# 这里我们选择正类（通常是shap_values的第二个元素，取决于模型和数据）
shap_values_pos_class = shap_values[1]
# 蜂窝图（Beeswarm Plot）
# 假设shap_values是一个列表，每个元素对应一个类别的SHAP值
# 选择一个类别的SHAP值来生成蜂窝图;
selected_shap_values = shap_values[1]  #  例如，选择第一个类别的SHAP值

plt.figure()
shap.summary_plot(selected_shap_values, X_test_scaled, plot_type="dot",show=False)
beeswarm_path = os.path.join(output_folder, 'beeswarm_plot_class_1.png')
plt.savefig(beeswarm_path)
plt.close()
print(f"Beeswarm plot for class 1 saved to: {beeswarm_path}")
'在调用shap.summary_plot函数时，添加show=False参数。这个参数的目的是防止图形立即显示，从而允许您在图形完全生成后保存它。'

# 依赖图（Dependence Plot）
feature_index = 0  # 选择特征索引，这里以第一个特征为例
plt.figure()
shap.dependence_plot(feature_index, shap_values_pos_class, X_test_scaled, show=False)
dependence_path = os.path.join(output_folder, f'dependence_plot_feature_{feature_index}.png')
plt.savefig(dependence_path)
plt.close()
print(f"Dependence plot for feature {feature_index} saved to: {dependence_path}")
'在调用shap.summary_plot函数时，添加show=False参数。这个参数的目的是防止图形立即显示，从而允许您在图形完全生成后保存它。'

# 决策图（Decision Plot）
# 如果X_test是Pandas DataFrame，获取特征名称
feature_names = X_test.columns.tolist()

# 决策图（Decision Plot）
plt.figure()
# 确保传递特征名称给决策图
shap.decision_plot(explainer.expected_value[1], shap_values_pos_class[0,:], feature_names=feature_names, show=False)
decision_path = os.path.join(output_folder, 'decision_plot.png')
plt.savefig(decision_path)
plt.close()
print(f"Decision plot saved to: {decision_path}")

# 注意：对于其他图形如蜂窝图、依赖图和决策图，如果您的模型是多输出模型，您可能需要根据实际情况调整代码以适应SHAP的限制。


'您遇到的错误是因为在尝试使用SHAP生成蜂窝图（Beeswarm Plot）时，对于多输出模型（例如随机森林在多分类问题中），' \
'SHAP目前只支持条形图（Bar Plot）类型的总结图（Summary Plot）。' \
'错误信息“AssertionError: Only plot_type = bar is supported for multi-output explanations!”明确指出了这一点。'

'''
在 shap.force_plot 函数中，explainer.expected_value[1] 和 shap_values[1][0,:] 所表示的数字具体含义如下：
explainer.expected_value[1]：这个值代表模型的基线预测值或者基础预测值。
SHAP（SHapley Additive exPlanations）方法将每个特征对模型输出的影响解释为相对于这个基线预测值的增益或减少。
通常，基线预测值是模型对整个训练集的平均预测值。
'''

'shap_values[1][0,:]：这个值是由 SHAP 方法计算得到的针对某个样本的 SHAP 值。' \
'SHAP 值表示了每个特征对于模型输出的影响程度，' \
'正值表示特征对于增加模型输出有正向影响，负值表示特征对于减少模型输出有负向影响。'

'而 X_test_scaled[0,:] 则是对应于特征值的样本数据，它表示了要解释的样本的特征值。'
