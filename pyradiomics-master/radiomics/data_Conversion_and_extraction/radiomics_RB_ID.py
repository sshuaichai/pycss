import pandas as pd
import re

# 加载CSV文件
file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_features.csv"
df = pd.read_csv(file_path)

# 定义一个函数来从字符串中提取数字
def extract_number(s):
    # 使用正则表达式查找所有数字
    numbers = re.findall(r'\d+', s)
    # 将找到的数字连接成一个字符串（如果有多个数字的话）
    return ''.join(numbers)

# 应用函数到Image列，提取数字，并创建新的列
df['ID'] = df['Image'].apply(extract_number)

# 显示结果，查看是否正确添加了新列
print(df.head())

# 保存修改后的DataFrame回CSV文件
output_file_path = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output\final\radiomics_R3B12_ID.csv"
df.to_csv(output_file_path, index=False)
