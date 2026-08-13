import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


def create_background_mask_simple(image, max_search_distance=100, extend_pixels=3):
    """
    简单的背景mask创建 - 按每列的两端开始向内寻找最小值，并延伸背景

    参数:
    - image: 输入图像数组
    - max_search_distance: 最大搜索距离
    - extend_pixels: 背景延伸的像素数

    返回:
    - mask: 背景mask (背景为1, 中心为0)
    """
    height, width = image.shape
    min_val = np.min(image)

    # 初始化mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 对每一列进行处理
    for j in range(width):
        col = image[:, j]

        # 从上向下搜索
        for i in range(min(max_search_distance, height)):
            if col[i] != min_val:
                # 找到非最小值，停止并设置mask，同时延伸背景
                extend_position = min(i + extend_pixels, height - 1)
                mask[:extend_position, j] = 1
                break
        else:
            # 如果整列都是最小值，设置整个列
            mask[:, j] = 1

        # 从下向上搜索
        for i in range(min(max_search_distance, height)):
            if col[height - 1 - i] != min_val:
                # 找到非最小值，停止并设置mask，同时延伸背景
                extend_position = max(height - i - extend_pixels, 0)
                mask[extend_position:, j] = 1
                break
        else:
            # 如果整列都是最小值，设置整个列
            mask[:, j] = 1

    return mask


def process_dicom_remove_background(input_dir, output_dir_D, output_dir_M, show_comparison=True):
    """
    处理DICOM图像：去除边缘背景区域

    参数:
    - input_dir: 输入DICOM图像目录
    - output_dir_D: 输出DICOM图像目录
    - output_dir_M: 输出PNG图像目录
    - show_comparison: 是否显示处理前后对比
    """
    os.makedirs(output_dir_D, exist_ok=True)
    os.makedirs(output_dir_M, exist_ok=True)

    # 获取所有DICOM文件
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    processed_count = 0

    for filename in dicom_files:
        try:
            file_path = os.path.join(input_dir, filename)

            # 读取DICOM文件
            ds = pydicom.dcmread(file_path)

            # 获取像素数据
            image_data = ds.pixel_array

            print(f"处理图像: {filename}, 形状: {image_data.shape}")

            # 创建背景mask
            mask = create_background_mask_simple(image_data)

            # 处理图像：将背景区域设为最大值
            processed_image = image_data.copy()
            max_val = np.max(processed_image)
            processed_image[mask == 1] = max_val

            # 保存处理后的DICOM文件（不带后缀）
            output_dicom_path = os.path.join(output_dir_D, filename)

            # 更新DICOM文件的像素数据
            ds.PixelData = processed_image.tobytes()
            ds.Rows, ds.Columns = processed_image.shape

            # 保存DICOM文件
            pydicom.dcmwrite(output_dicom_path, ds)

            # 保存PNG文件
            output_png_path = os.path.join(output_dir_M, f"{filename}.png")

            # 归一化图像数据用于显示
            if np.max(processed_image) > np.min(processed_image):
                normalized_image = (processed_image - np.min(processed_image)) / (
                        np.max(processed_image) - np.min(processed_image))
                normalized_image = (normalized_image * 255).astype(np.uint8)
            else:
                normalized_image = processed_image.astype(np.uint8)

            # 使用PIL保存PNG
            png_image = Image.fromarray(normalized_image)
            png_image.save(output_png_path)

            # 显示处理前后对比
            if show_comparison and processed_count < 3:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 原始图像
                axes[0].imshow(image_data, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Mask
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Background Mask')
                axes[1].axis('off')

                # 处理后的图像
                axes[2].imshow(processed_image, cmap='gray')
                axes[2].set_title('Processed Image')
                axes[2].axis('off')

                plt.tight_layout()
                plt.show()

            processed_count += 1
            print(f"处理完成: {filename} ({processed_count}/{len(dicom_files)})")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue


def main():
    input_directory = r"D:/med_data/ANY/data1"
    output_D = r"D:/med_data/ANY/processed_data\D"
    output_M = r"D:/med_data/ANY/processed_data\PNG"

    print("开始处理DICOM图像...")
    print(f"输入目录: {input_directory}")
    print(f"输出DICOM目录: {output_D}")
    print(f"输出PNG目录: {output_M}")

    process_dicom_remove_background(input_directory, output_D, output_M, show_comparison=True)

    print("处理完成！")


if __name__ == "__main__":
    main()