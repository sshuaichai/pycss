import pandas as pd
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果的可重复性
random_seed = 42

# 从预处理开始，就不能使用验证组
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_RB_ID_updated.csv"
df = pd.read_csv(data_path)
# 分层抽样分组
# 将数据集按照7:3的比例划分为训练集和临时集
train_val_df, test_df = train_test_split(df, test_size=0.3, random_state=random_seed, stratify=df['event'])

# 将临时集按照1:1的比例划分为验证集和测试集
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=random_seed, stratify=test_df['event'])

# 保存训练集、验证集和测试集
train_val_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\train_set_RADIOMICS.csv", index=False)
val_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\val_set_RADIOMIC.csv", index=False)
test_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\Data_set_partition\test_set_RADIOMIC.csv", index=False)

print("训练集、验证集和测试集已成功保存。")
