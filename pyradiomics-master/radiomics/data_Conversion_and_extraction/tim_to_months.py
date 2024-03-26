import pandas as pd

# 加载Excel文件
data_path = r"D:\zhuomian\all_transformed.xlsx"
df = pd.read_excel(data_path)

# 假设'time'列以天为单位，将其转换为月（以30天为一个月）
df['time_months'] = df['time'] / 30

# 查看转换后的结果
print(df[['time', 'time_months']].head())

# 如果需要，可以将修改后的DataFrame保存回Excel或CSV文件
df.to_excel(data_path, index=False)  # 如果要保存回Excel
# 或者
# df.to_csv("D:\zhuomian\all_transformed_momths.csv", index=False)  # 如果要保存为CSV
