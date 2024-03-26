import os
import re
import shutil

"""
将文件名转换为大写。
从文件名中移除数字。
在文件名末尾添加后缀“_Segmentation.seg”
"""
def transform_and_copy_files(src_dir, dest_dir, suffix='_Segmentation.seg'):
  """
  转换指定目录中的.nrrd文件名称，并将它们以.nrrd格式复制到新的目录：
  1. 将名称转换为大写。
  2. 移除名称中的数字。
  3. 在名称末尾添加指定的后缀，保持.nrrd格式不变。
  4. 将转换后的文件复制到新的目录。

  参数：
  - src_dir: 源文件所在的目录。
  - dest_dir: 目标目录，转换后的文件将被保存在这里。
  - suffix: 要添加到转换后文件名末尾的后缀。
  """
  # 确保源目录存在
  if not os.path.exists(src_dir):
    print(f"源目录 {src_dir} 不存在。")
    return

  # 创建目标目录，如果它不存在的话
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    print(f"已创建目标目录 {dest_dir}")

  # 遍历源目录中的所有文件
  for filename in os.listdir(src_dir):
    if filename.endswith('.nrrd'):
      # 转换文件名，保留.nrrd格式
      new_name_base = re.sub(r'\d+', '', filename.split('.')[0]).upper() + suffix
      new_name = new_name_base + '.nrrd'
      # 定义源文件和目标文件的路径
      src_path = os.path.join(src_dir, filename)
      dest_path = os.path.join(dest_dir, new_name)
      # 复制并重命名文件
      shutil.copy(src_path, dest_path)
      print(f"已将 '{filename}' 复制并重命名为 '{new_name}'")


# 指定源目录和目标目录
src_dir = "D:\\zhuomian\\data\\Dataset018_Lung\\labels"
dest_dir = "D:\\zhuomian\\data\\Dataset018_Lung\\labels_newname"
transform_and_copy_files(src_dir, dest_dir)
