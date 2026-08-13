import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def process_dicom_images(input_dir, output_dir):
    """
    处理DICOM图像：将像素值为1的像素设为该图像的最大值

    参数:
    - input_dir: 输入DICOM图像目录
    - output_dir: 输出图像目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有DICOM文件
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and
                   f.lower().endswith(('.dcm', '.dicom'))]

    # 如果没有找到DICOM文件，尝试所有文件
    if len(dicom_files) == 0:
        dicom_files = [f for f in os.listdir(input_dir)
                       if os.path.isfile(os.path.join(input_dir, f))]
        print(f"未找到标准DICOM文件，尝试处理所有 {len(dicom_files)} 个文件")
    else:
        print(f"找到 {len(dicom_files)} 个DICOM文件")

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有文件!")
        return

    # 处理统计
    processed_count = 0
    error_count = 0

    print("开始处理图像...")

    for filename in tqdm(dicom_files, desc="处理DICOM图像"):
        try:
            # 读取DICOM文件
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)

            # 打印原始图像信息
            print(f"\n处理文件: {filename}")
            print(f"  原始图像形状: {image.shape}")
            print(f"  原始图像值范围: [{image.min():.2f}, {image.max():.2f}]")

            # 统计像素值为1的像素数量
            pixels_equal_1 = np.sum(image == 1)
            print(f"  像素值=1的像素数量: {pixels_equal_1}")

            # 处理图像：将像素值为1的像素设为图像最大值
            image_max = image.max()
            if pixels_equal_1 > 0:
                # 创建掩码
                mask = (image == 1)
                # 将值为1的像素设为最大值
                image[mask] = image_max

                print(f"  已将 {pixels_equal_1} 个像素值从1改为最大值 {image_max:.2f}")
            else:
                print(f"  没有找到像素值为1的像素，图像保持不变")

            # 归一化到0-255范围用于保存
            image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
            image_uint8 = image_normalized.astype(np.uint8)

            # 保存处理后的图像
            output_filename = os.path.splitext(filename)[0] + '_processed.png'
            output_path = os.path.join(output_dir, output_filename)

            # 使用PIL保存图像
            pil_image = Image.fromarray(image_uint8)
            pil_image.save(output_path)

            print(f"  已保存处理后的图像: {output_filename}")
            print(f"  处理后图像值范围: [{image.min():.2f}, {image.max():.2f}]")

            processed_count += 1

        except Exception as e:
            print(f"错误处理文件 {filename}: {e}")
            error_count += 1

    # 打印处理总结
    print(f"\n{'=' * 50}")
    print("处理完成!")
    print(f"{'=' * 50}")
    print(f"总文件数: {len(dicom_files)}")
    print(f"成功处理: {processed_count}")
    print(f"处理失败: {error_count}")
    print(f"输出目录: {output_dir}")


def process_with_visualization(input_dir, output_dir, show_images=False):
    """
    处理DICOM图像并可选显示处理前后的对比

    参数:
    - input_dir: 输入DICOM图像目录
    - output_dir: 输出图像目录
    - show_images: 是否显示处理前后的图像对比
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有DICOM文件
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有文件!")
        return

    print(f"找到 {len(dicom_files)} 个文件")

    processed_count = 0

    for filename in tqdm(dicom_files, desc="处理并可视化"):
        try:
            # 读取原始DICOM
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            original_image = dicom_data.pixel_array.astype(np.float32)

            # 创建处理后的图像副本
            processed_image = original_image.copy()

            # 处理：将像素值为1的像素设为最大值
            image_max = original_image.max()
            mask = (original_image == 1)
            pixels_to_change = np.sum(mask)

            if pixels_to_change > 0:
                processed_image[mask] = image_max

            # 保存处理后的图像
            output_filename = os.path.splitext(filename)[0] + '_processed.png'
            output_path = os.path.join(output_dir, output_filename)

            # 归一化并保存
            processed_normalized = (processed_image - processed_image.min()) / (
                        processed_image.max() - processed_image.min() + 1e-8) * 255
            Image.fromarray(processed_normalized.astype(np.uint8)).save(output_path)

            # 如果需要显示图像
            if show_images and processed_count < 5:  # 只显示前5个
                display_comparison(original_image, processed_image, filename, pixels_to_change)

            processed_count += 1

        except Exception as e:
            print(f"错误处理文件 {filename}: {e}")

    print(f"\n处理完成! 共处理 {processed_count} 个文件")


def display_comparison(original, processed, filename, changed_pixels):
    """
    显示处理前后的图像对比
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    im1 = ax1.imshow(original, cmap='gray')
    ax1.set_title(f'原始图像\n{filename}\n范围: [{original.min():.1f}, {original.max():.1f}]')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 处理后的图像
    im2 = ax2.imshow(processed, cmap='gray')
    ax2.set_title(f'处理后图像\n改变的像素: {changed_pixels}\n范围: [{processed.min():.1f}, {processed.max():.1f}]')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 差异图像
    difference = processed - original
    im3 = ax3.imshow(difference, cmap='coolwarm', vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
    ax3.set_title('差异图像\n(处理后 - 原始)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    plt.tight_layout()
    plt.show()


def batch_statistics(input_dir):
    """
    批量统计DICOM图像中像素值为1的情况
    """
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有文件!")
        return

    print(f"分析 {len(dicom_files)} 个文件...")

    stats = {
        'total_files': len(dicom_files),
        'files_with_pixel_1': 0,
        'total_pixels_1': 0,
        'max_pixels_1_per_file': 0,
        'files_info': []
    }

    for filename in tqdm(dicom_files, desc="统计分析"):
        try:
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)

            pixels_1 = np.sum(image == 1)
            file_info = {
                'filename': filename,
                'shape': image.shape,
                'value_range': (image.min(), image.max()),
                'pixels_equal_1': pixels_1,
                'percentage_1': pixels_1 / image.size * 100
            }

            stats['files_info'].append(file_info)

            if pixels_1 > 0:
                stats['files_with_pixel_1'] += 1
                stats['total_pixels_1'] += pixels_1
                stats['max_pixels_1_per_file'] = max(stats['max_pixels_1_per_file'], pixels_1)

        except Exception as e:
            print(f"错误分析文件 {filename}: {e}")

    # 打印统计结果
    print(f"\n{'=' * 50}")
    print("统计分析结果")
    print(f"{'=' * 50}")
    print(f"总文件数: {stats['total_files']}")
    print(f"包含像素值=1的文件数: {stats['files_with_pixel_1']}")
    print(f"总像素值=1的像素数: {stats['total_pixels_1']}")
    print(f"单个文件中最大像素值=1的像素数: {stats['max_pixels_1_per_file']}")

    if stats['files_with_pixel_1'] > 0:
        avg_pixels_per_file = stats['total_pixels_1'] / stats['files_with_pixel_1']
        print(f"平均每个文件像素值=1的像素数: {avg_pixels_per_file:.1f}")

    # 显示前几个包含像素值=1的文件
    files_with_1 = [f for f in stats['files_info'] if f['pixels_equal_1'] > 0]
    if files_with_1:
        print(f"\n前5个包含像素值=1的文件:")
        for i, file_info in enumerate(files_with_1[:5]):
            print(f"  {i + 1}. {file_info['filename']}: {file_info['pixels_equal_1']} 个像素")

    return stats


def main():
    """
    主函数：处理DICOM图像
    """
    # 设置路径
    input_dir = "D:/ai/test_0"  # 输入DICOM图像目录
    output_dir = "D:/ai/processed_images"  # 输出目录

    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 选择处理模式
    print("\n选择处理模式:")
    print("1. 简单处理（仅处理图像）")
    print("2. 处理并显示对比（显示前5个图像）")
    print("3. 统计分析（不处理，仅分析）")

    try:
        choice = input("请输入选择 (1/2/3, 默认1): ").strip()
        if choice == '':
            choice = '1'
    except:
        choice = '1'

    if choice == '1':
        # 简单处理模式
        process_dicom_images(input_dir, output_dir)

    elif choice == '2':
        # 处理并显示对比
        process_with_visualization(input_dir, output_dir, show_images=True)

    elif choice == '3':
        # 统计分析模式
        batch_statistics(input_dir)

    else:
        print("无效选择，使用默认模式")
        process_dicom_images(input_dir, output_dir)


if __name__ == "__main__":
    main()
