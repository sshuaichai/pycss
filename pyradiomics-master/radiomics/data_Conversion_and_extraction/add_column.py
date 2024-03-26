import pandas as pd

# 定义文件路径
excel_path = "D:\\zhuomian\\merged_imagename_with_id.xlsx"
csv_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_RB_ID.csv"
output_dir = "D:\\zhuomian\\pyradiomics\\pyradiomics-master\\examples\\output\\final\\"

# 读取Excel文件
excel_df = pd.read_excel(excel_path)

# 读取CSV文件
csv_df = pd.read_csv(csv_path)

# 如果CSV中没有'time'和'event'列，添加它们
if 'time' not in csv_df.columns:
    csv_df['time'] = pd.NA
if 'event' not in csv_df.columns:
    csv_df['event'] = pd.NA

# 设置ID为索引
excel_df.set_index('ID', inplace=True)
csv_df.set_index('ID', inplace=True)

# 更新CSV数据
for idx, row in excel_df.iterrows():
    if idx in csv_df.index:
        csv_df.at[idx, 'time'] = row['time']
        csv_df.at[idx, 'event'] = row['event']

# 重置索引
csv_df.reset_index(inplace=True)

# 保存更新后的CSV文件
csv_df.to_csv(output_dir + 'radiomics_RB_ID_updated.csv', index=False)
