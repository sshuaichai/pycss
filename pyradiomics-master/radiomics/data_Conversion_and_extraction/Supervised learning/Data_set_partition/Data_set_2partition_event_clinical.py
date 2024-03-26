import pandas as pd
from sklearn.model_selection import train_test_split
import chardet


# 设置随机种子以确保结果的可重复性
random_seed = 42

# 加载数据
data_path = r"D:\zhuomian\all_shuzhi.xlsx"
# 使用 Latin-1 编码读取 CSV 文件
df = pd.read_excel(data_path)


# 分层抽样分割数据集
# 由于您的目标是按照7:3的比例分割数据集，test_size设置为0.3
train_df, val_df = train_test_split(df, test_size=0.3, random_state=random_seed, stratify=df['event'])

# 保存训练集和验证集
train_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_clinical_set.csv", index=False)
val_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_clinical_set.csv", index=False)

print("训练集和验证集已成功保存。")
