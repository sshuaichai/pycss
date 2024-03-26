# 1.[data_conversion.py]
把nrrd格式转换为nii.gz格式
注意：
# 构建仿射变换矩阵
affine = np.eye(4)
affine[:3, :3] = space_directions  # 设置方向
affine[:3, 3] = space_origin  # 设置原点

# 调整仿射矩阵中的符号以解决颠倒问题
```调整1为旋转旋转180度，就是颠倒图像。
affine[0,0] *= -1  # X轴颠倒  X轴（矢状位）
affine[1,1] *= -1  # Y轴颠倒  Y轴（冠状位）
affine[2,2] *= 1  # Z轴颠倒 Z轴（横断位）
```

# 2.[Change_name.py]
按照图像存储顺序来重新命名。
注意！：image和label在文件夹中必须顺序一致，可以采取相同的命名前缀来达到要求！
