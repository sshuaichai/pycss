import pandas as pd
import re

# 文件路径
input_file_path = "D:\\zhuomian\\Dataset021_lung\\image_names_with_id.csv"
output_file_path = "D:\\zhuomian\\Dataset021_lung\\image_names_with_updated_id.csv"

# 读取CSV文件
df = pd.read_csv(input_file_path)

# 提取Image Name列中的数字并赋值给ID列
df['ID'] = df['Image Name'].apply(lambda x: ''.join(re.findall(r'\d+', x)))

# 保存修改后的DataFrame为新的CSV文件
df.to_csv(output_file_path, index=False)

print("File has been saved to:", output_file_path)
