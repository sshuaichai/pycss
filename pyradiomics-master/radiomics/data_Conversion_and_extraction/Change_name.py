import os
import shutil  # 用于文件复制


def rename_and_copy_images(src_folder, dst_folder):
  # 确保目标文件夹存在，如果不存在，则创建
  if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

  # 初始化计数器用于生成新的文件名
  counter = 1

  # 获取指定文件夹内的所有文件
  files = sorted([f for f in os.listdir(src_folder) if f.endswith('.nii.gz')])

  for filename in files:
    # 构造新的文件名
    new_filename = f"lung_{counter}.nii.gz" #新的名稱

    # 定义源文件和目标文件的完整路径
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, new_filename)

    # 复制并重命名文件到目标文件夹
    shutil.copy2(src_path, dst_path)
    print(f"Copied and renamed '{filename}' to '{new_filename}'")

    # 更新计数器
    counter += 1 #name_+1.nii.gz


# 示例用法
src_folder = r"D:\zhuomian\140\labels_nii"
dst_folder = r"D:\zhuomian\Dataset021_lung\labelsTr"
rename_and_copy_images(src_folder, dst_folder)

'别忘了更改counter！！！  现在是 1 '
