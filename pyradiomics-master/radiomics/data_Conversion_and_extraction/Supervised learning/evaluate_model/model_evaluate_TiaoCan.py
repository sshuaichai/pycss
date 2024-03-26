# 为了综合评估和选择最佳模型，您可以考虑使用加权评分方法，
# 将不同的性能指标（如ROC AUC Mean、Accuracy Mean、Recall Mean、F1 Mean）结合起来，
# 为每个模型计算一个总体评分。
import pandas as pd
import os

def evaluate_models(file_path, output_folder, weights):
    # 读取模型性能指标
    df = pd.read_csv(file_path)

    # 计算加权评分
    df['Weighted Score'] = sum(df[key] * weight for key, weight in weights.items())

    # 找到加权评分最高的模型
    best_model = df.loc[df['Weighted Score'].idxmax()]

    print("最佳模型及其性能指标：")
    print(best_model)

    # 打印所有模型的评分
    print("\n所有模型的评分：")
    print(df[['Model', 'Weighted Score']].sort_values(by='Weighted Score', ascending=False))

    # 保存评分结果到CSV文件
    output_file_path = os.path.join(output_folder, "Weighted_model_evaluation_scores.csv") # 名称
    df[['Model', 'Weighted Score']].sort_values(by='Weighted Score', ascending=False).to_csv(output_file_path, index=False)

    print(f"\n评分结果已保存到：{output_file_path}")

# 模型性能指标文件路径
file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_TiaoCan_models\model_evaluation_results.csv"

# 输出文件夹路径
output_folder = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Train_Output\train_TiaoCan_models"

# 设置各性能指标的权重
weights = {
    'ROC AUC Mean': 0.4,
    'Accuracy Mean': 0.2,
    'Recall Mean': 0.2,
    'F1 Mean': 0.2,
}

# 调用函数
evaluate_models(file_path, output_folder, weights)

