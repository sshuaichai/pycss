''
'每个特征与目标变量的相关性,保留互关联特征之一与目标变量强相关的特征'

'考虑目标变量进行相关性分析，从而保留对目标变量相关性更高的特征：'

'status不要忘记手动添加！！！！！'

'改为Correlation与目标变量前10的特征，为了找到与目标变量 status相关性前10的特征，并将这些特征的相关性结果保存到Excel文件中，'
'我们需要对 cor_df DataFrame根据 Correlation列的绝对值进行降序排序，并选取前10行。'
'这样可以确保我们获得的是与目标变量最强相关的前10个特征，无论这些相关性是正的还是负的。'

import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

# 定义文件路径
file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\2_featureextractor.xlsx"
save_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\FeaturesOutput\3_Top10_Correlation_Features.xlsx"

# 读取数据
df = pd.read_excel(file_path)

# 确保 'status' 列为数值型，如果不是，根据你的数据进行适当的转换
if 'status' not in df.columns:
    print("DataFrame 中没有找到 'status' 列，请检查列名。")
else:
    try:
        df['status'] = pd.to_numeric(df['status'], errors='raise')
    except ValueError:
        print("无法将 'status' 列转换为数值型，请检查数据。")

# 选择的相关性分析方法
method = 'kendall'  # 可以更改为 'pearson', 'spearman' 或 'kendall'

# 初始化一个空列表来存储相关性结果
cor_results = []

# 定义一个函数来选择相关性测试方法
def compute_correlation(x, y, method='kendall'):
    if method == 'pearson':
        return pearsonr(x, y)[0]  # 只获取相关系数
    elif method == 'spearman':
        return spearmanr(x, y)[0]  # 只获取相关系数
    elif method == 'kendall':
        return kendalltau(x, y)[0]  # 只获取相关系数

# 遍历DataFrame中的所有特征列，除了 'status' 列
for column in df.columns[:-1]:  # 假设 'status' 列是最后一列
    if df[column].dtype in [float, int]:  # 确保列是数值型
        cor = compute_correlation(df[column], df['status'], method=method)
        cor_results.append([column, cor])

# 将结果转换为 DataFrame
cor_df = pd.DataFrame(cor_results, columns=['Feature', 'Correlation'])

# 根据 'Correlation' 的绝对值降序排序，选取相关性前10的特征
top10_features = cor_df.reindex(cor_df.Correlation.abs().sort_values(ascending=False).index).head(10)#Top 10 features
# 保存相关性前10的特征结果到 Excel 文件
top10_features.to_excel(save_path, index=False)

print(f"相关性分析（使用{method}方法）完成，并已保存前10个最强相关的特征至: {save_path}")

