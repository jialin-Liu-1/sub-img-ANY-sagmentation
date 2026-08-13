import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


def create_background_mask(image, max_search_distance=50, extend_distance=5):
    """
    创建背景mask

    参数:
    - image: 输入图像数组
    - max_search_distance: 最大搜索距离
    - extend_distance: 延长搜索距离

    返回:
    - mask: 背景mask (背景为1, 中心为0)
    """
    height, width = image.shape
    min_val = np.min(image)
    max_val = np.max(image)

    # 初始化mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 从四个方向搜索背景边界
    boundaries = {}

    # 从上边缘向下搜索
    top_boundary = height
    for i in range(min(max_search_distance, height)):
        row = image[i, :]
        if not np.all(row == min_val):
            top_boundary = i + extend_distance
            break

    # 从下边缘向上搜索
    bottom_boundary = 0
    for i in range(min(max_search_distance, height)):
        row = image[height - 1 - i, :]
        if not np.all(row == min_val):
            bottom_boundary = height - 1 - i - extend_distance
            break

    # 从左边缘向右搜索
    left_boundary = width
    for j in range(min(max_search_distance, width)):
        col = image[:, j]
        if not np.all(col == min_val):
            left_boundary = j + extend_distance
            break

    # 从右边缘向左搜索
    right_boundary = 0
    for j in range(min(max_search_distance, width)):
        col = image[:, width - 1 - j]
        if not np.all(col == min_val):
            right_boundary = width - 1 - j - extend_distance
            break

    # 创建mask - 边界外的区域设为背景
    mask[:top_boundary, :] = 1  # 上背景
    mask[bottom_boundary:, :] = 1  # 下背景
    mask[:, :left_boundary] = 1  # 左背景
    mask[:, right_boundary:] = 1  # 右背景

    return mask


def process_dicom_remove_background(input_dir, output_dir, show_comparison=True):
    """
    处理DICOM图像：去除边缘背景区域

    参数:
    - input_dir: 输入DICOM图像目录
    - output_dir: 输出图像目录
    - show_comparison: 是否显示处理前后对比
    """
    os.makedirs(output_dir, exist_ok=True)

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

            # 创建背景mask
            mask = create_background_mask(image_data)

            # 处理图像：将背景区域设为最大值
            processed_image = image_data.copy()
            processed_image[mask == 1] = np.max(processed_image)

            # 保存处理后的DICOM文件（不带后缀）
            output_dicom_path = os.path.join(output_dir, filename)
            ds.PixelData = processed_image.tobytes()
            ds.Rows, ds.Columns = processed_image.shape
            pydicom.dcmwrite(output_dicom_path, ds)

            # 保存PNG文件
            output_png_path = os.path.join(output_dir, f"{filename}.png")

            # 归一化图像数据用于显示
            normalized_image = (processed_image - np.min(processed_image)) / (
                        np.max(processed_image) - np.min(processed_image))
            normalized_image = (normalized_image * 255).astype(np.uint8)

            # 使用PIL保存PNG
            png_image = Image.fromarray(normalized_image)
            png_image.save(output_png_path)

            # 显示处理前后对比
            if show_comparison and processed_count < 3:  # 只显示前3张的对比
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
    input_directory = r"D:\ai\test_0"
    output_directory = r"D:\ai\processed_data"

    print("开始处理DICOM图像...")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")

    process_dicom_remove_background(input_directory, output_directory, show_comparison=True)

    print("处理完成！")


if __name__ == "__main__":
    main()