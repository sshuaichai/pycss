import pandas as pd
import os

def evaluate_models(file_path, output_folder, weights, roc_auc_std_factor):
    # 读取模型性能指标
    df = pd.read_csv(file_path)

    # 调整ROC AUC Mean以反映模型性能的稳定性
    df['Adjusted ROC AUC Mean'] = df['ROC AUC Mean'] - (df['ROC AUC Std'] * roc_auc_std_factor)

    # 计算加权评分
    df['Weighted Score'] = (
        df['Adjusted ROC AUC Mean'] * weights['ROC AUC Mean'] +
        df['Accuracy Mean'] * weights['Accuracy Mean'] +
        df['Recall Mean'] * weights['Recall Mean'] +
        df['F1 Mean'] * weights['F1 Mean']
    )

    # 找到加权评分最高的模型
    best_model_idx = df['Weighted Score'].idxmax()
    best_model = df.loc[best_model_idx]

    print("最佳模型及其性能指标：")
    print(best_model)

    # 打印所有模型的评分
    print("\n所有模型的评分：")
    print(df[['Model', 'Weighted Score']].sort_values(by='Weighted Score', ascending=False))

    # 保存评分结果到CSV文件
    output_file_path = os.path.join(output_folder, "Weighted_model_stability_evaluation_scores.csv")
    df[['Model', 'Weighted Score']].sort_values(by='Weighted Score', ascending=False).to_csv(output_file_path, index=False)

    print(f"\n评分结果已保存到：{output_file_path}")

# 模型性能指标文件路径
file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_models\model_evaluation_results.csv"

# 输出文件夹路径
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_models"

# 设置各性能指标的权重
weights = {
    'ROC AUC Mean': 0.4,
    'Accuracy Mean': 0.2,
    'Recall Mean': 0.2,
    'F1 Mean': 0.2,
}

# 设置ROC AUC Std在评分中的影响因子
roc_auc_std_factor = 1  # 这个值可以根据您对稳定性的重视程度进行调整

# 调用函数
evaluate_models(file_path, output_folder, weights, roc_auc_std_factor)
