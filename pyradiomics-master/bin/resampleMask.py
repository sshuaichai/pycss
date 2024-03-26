''
"""
该脚本是如何将掩模图像重新采样为参考图像的几何形状的示例。
这使得在一个图像上生成的掩模能够用于另一图像的特征提取。
请注意，这仍然要求掩模具有与参考图像相似的物理空间。
"""
'''
这段代码用于将掩模图像重新采样为参考图像的几何形状的示例。
总体而言，该脚本的作用是对输入的掩模图像进行重采样，使其具有与指定参考图像相同的几何形状，以便在不同图像间进行特征提取等操作。需要注意的是，重采样仍然要求掩模具有与参考图像相似的物理空间。
'''
import argparse  # 导入参数解析模块

import SimpleITK as sitk  # 导入SimpleITK库

parser = argparse.ArgumentParser()  # 创建参数解析器
parser.add_argument('image', metavar='Image', help='将掩模重新采样为的参考图像')  # 添加图像参数
parser.add_argument('mask', metavar='Mask', help='输入掩码重新采样')  # 添加掩模参数
parser.add_argument('resMask', metavar='Out', help='存储重采样掩码的文件名')  # 添加输出参数

def main():  # 主函数
  args = parser.parse_args()  # 解析参数
  image = sitk.ReadImage(args.image)  # 读取参考图像
  mask = sitk.ReadImage(args.mask)  # 读取输入掩模

  rif = sitk.ResampleImageFilter()  # 创建重采样滤波器
  rif.SetReferenceImage(image)  # 设置参考图像
  rif.SetOutputPixelType(mask.GetPixelID())  # 设置输出像素类型
  rif.SetInterpolator(sitk.sitkNearestNeighbor)  # 设置插值方法为最近邻插值
  resMask = rif.Execute(mask)  # 执行掩模重采样

  sitk.WriteImage(resMask, args.resMask, True)  # 保存重采样后的掩模，True启用压缩

if __name__ == '__main__':
    main()  # 调用主函数
