import pandas as pd

# 读取Excel文件
df = pd.read_excel("D:\\zhuomian\\all.xlsx")

# 转换sex列
df['sex'] = df['sex'].map({1: 'male', 2: 'female'})

# 转换position列
position_mapping = {
    1: 'Right Upper',
    2: 'Right Middle',
    3: 'Right Lower',
    4: 'Left Upper',
    5: 'Left Lower'
}
df['position'] = df['position'].map(position_mapping)

# 转换cli_stage列
cli_stage_mapping = {
    'ⅣB': 'IVB',
    'ⅢB': 'IIIB',
    'ⅣA': 'IVA',
    'ⅢA': 'IIIA',
    'ⅢC': 'IIIC'
}
df['cli_stage'] = df['cli_stage'].replace(cli_stage_mapping)

# 转换pth_type列
pth_type_mapping = {
    '肺腺癌': 'adenocarcinoma',
    '肺鳞癌': 'squamous cell carcinoma',
    '肺肉瘤样癌': 'sarcomatoid carcinoma',
    '小细胞肺癌': 'small cell cancer'
}
df['pth_type'] = df['pth_type'].replace(pth_type_mapping)

# 保存转换后的DataFrame到新的Excel文件
df.to_excel("D:\\zhuomian\\all_transformed.xlsx", index=False)

print("数据转换完成，已保存到新的Excel文件。")
