import SimpleITK as sitk
import os
from radiomics import imageoperations

def main(image_dir, mask_dir, output_image_dir, output_mask_dir):
    # 确保输出目录存在
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    for image_name in os.listdir(image_dir):
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

            # 对图像和掩码进行重采样
            resampled_image, resampled_mask = imageoperations.resampleImage(image, mask,
                                                                            resampledPixelSpacing=[1, 1, 1], #这里注意和参数yaml保持一致，重新采样像素间距，以统一图像分辨率
                                                                            interpolator=sitk.sitkBSpline,
                                                                            label=1,
                                                                            padDistance=5)

            output_resampled_image_path = os.path.join(output_image_dir, f"{image_name}")
            output_resampled_mask_path = os.path.join(output_mask_dir, f"{mask_name}")

            # 保存重采样后的图像和掩码
            sitk.WriteImage(resampled_image, output_resampled_image_path)
            sitk.WriteImage(resampled_mask, output_resampled_mask_path)

        except Exception as e:
            print(f"An error occurred while processing {image_name}: {str(e)}")

if __name__ == "__main__":
    image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_lung\imagesTr"
    mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_lung\labelsTr2_nii"
    output_image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset023_lung\imagesTr"
    output_mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset023_lung\labelsTr" # 新增输出掩码目录

    main(image_dir, mask_dir, output_image_dir, output_mask_dir)
