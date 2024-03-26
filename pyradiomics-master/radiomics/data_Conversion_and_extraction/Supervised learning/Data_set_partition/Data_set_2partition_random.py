import pandas as pd
from sklearn.model_selection import train_test_split

# 数据文件路径
data_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\selected_features_cox.csv"

# 读取数据
df = pd.read_csv(data_path)

# 分割数据为训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)  # 设置随机种子为42

# 保存训练集和验证集到指定目录
train_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\train_set.csv", index=False)
val_df.to_csv(r"D:\zhuomian\pyradiomics\pyradiomics-master\radiomics\data_Conversion_and_extraction\Supervised learning\FeaturesOutput\raw_data\val_set.csv", index=False)

print("训练集和验证集已成功分割并保存。")
