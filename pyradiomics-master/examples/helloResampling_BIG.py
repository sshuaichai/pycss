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


            # 对图像和掩码进行重采样,考虑原始分辨率：重采样体素间距不应该远小于图像的原始分辨率。例如，如果CT图像的原始体素间距是0.5mm x 0.5mm x 0.5mm，那么将其重采样到0.1mm x 0.1mm x 0.1mm可能没有实际意义，因为这不会增加图像的真实细节，反而会增加数据量和处理时间。
            resampled_image, resampled_mask = imageoperations.resampleImage(image, mask,
                                                                           # 对图像和掩码进行重采样，这里将体素间距设置为更小的值以增大图像尺寸
                                                                            resampledPixelSpacing=[0.5, 0.5, 0.5],## 减小体素间距以增大图像尺寸
                                                                            #重新采样像素间距，以统一图像分辨率
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
    mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset021_lung\labelsTr"
    output_image_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset023_lung\imagesTr"
    output_mask_dir = r"D:\zhuomian\pyradiomics\pyradiomics-master\data\Dataset023_lung\labelsTr" # 新增输出掩码目录

    main(image_dir, mask_dir, output_image_dir, output_mask_dir)
