import SimpleITK as sitk
import os

def adjust_mask(image, mask):
    """
    确保掩模完全位于图像的空间范围内。
    如果掩模超出图像范围，则尝试自动调整。
    """
    # 获取图像和掩模的信息
    image_size = image.GetSize()
    mask_size = mask.GetSize()
    image_spacing = image.GetSpacing()
    mask_spacing = mask.GetSpacing()

    # 计算图像和掩模的物理尺寸
    image_physical_size = [size * spacing for size, spacing in zip(image_size, image_spacing)]
    mask_physical_size = [size * spacing for size, spacing in zip(mask_size, mask_spacing)]

    # 检查掩模是否超出图像范围
    adjustment_needed = False
    for img_size, mask_size in zip(image_physical_size, mask_physical_size):
        if mask_size > img_size:
            adjustment_needed = True
            break

    if adjustment_needed:
        print("调整掩模以适应图像空间...")
        # 使用SimpleITK的Resample方法调整掩模大小
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetReferenceImage(image)
        resample_filter.SetSize(image.GetSize())
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        adjusted_mask = resample_filter.Execute(mask)
        return adjusted_mask
    else:
        return mask

def main(image_dir, mask_dir, output_image_dir, output_mask_dir):
    # 确保输出目录存在
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    for image_name in os.listdir(image_dir):
        if not image_name.endswith('.nii.gz'):
            continue  # 跳过非NII.GZ文件
        image_path = os.path.join(image_dir, image_name)
        mask_name = image_name  # 假设掩码文件名与图像文件名相同
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.isfile(mask_path):
            print(f"Skipping {image_name}: corresponding mask file not found.")
            continue

        print(f'Processing {image_name}...')
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            # 调整掩模
            adjusted_mask = adjust_mask(image, mask)

            # 保存调整后的掩模
            adjusted_mask_path = os.path.join(output_mask_dir, mask_name)
            sitk.WriteImage(adjusted_mask, adjusted_mask_path)
            print(f"Saved adjusted mask to {adjusted_mask_path}")

        except Exception as e:
            print(f"An error occurred while processing {image_name}: {str(e)}")

if __name__ == "__main__":
    image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_lung\imagesTr"
    mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_lung\labelsTr"
    output_image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_Lung\resampled_images"
    output_mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_Lung\resampled_masks"

    main(image_dir, mask_dir, output_image_dir, output_mask_dir)
