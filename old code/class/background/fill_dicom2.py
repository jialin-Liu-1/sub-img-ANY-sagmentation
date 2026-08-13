import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy import ndimage
import shutil


def detect_edge_background_advanced(image, margin_ratio=0.15, intensity_threshold_ratio=0.1):
    """
    高级边缘背景检测方法

    参数:
    - image: 输入图像
    - margin_ratio: 边缘检测区域比例
    - intensity_threshold_ratio: 强度阈值比例

    返回:
    - left_crop: 左侧裁剪位置
    - right_crop: 右侧裁剪位置
    - background_mask: 背景掩码
    """
    height, width = image.shape

    # 计算边缘区域
    margin_width = int(width * margin_ratio)

    # 方法1: 基于边缘强度分析
    left_edge = image[:, :margin_width]
    right_edge = image[:, -margin_width:]

    # 计算边缘区域的平均强度
    left_mean = np.mean(left_edge)
    right_mean = np.mean(right_edge)
    center_mean = np.mean(image[:, margin_width:-margin_width])

    # 计算全局阈值
    global_threshold = np.percentile(image, 5)  # 使用5%分位数作为背景阈值

    # 方法2: 基于梯度分析
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = np.abs(sobel_x)

    # 边缘区域的梯度强度
    left_gradient = np.mean(sobel_x_abs[:, :margin_width])
    right_gradient = np.mean(sobel_x_abs[:, -margin_width:])
    center_gradient = np.mean(sobel_x_abs[:, margin_width:-margin_width])

    # 方法3: 基于连通组件分析
    binary_image = image > global_threshold
    labeled_array, num_features = ndimage.label(binary_image)

    # 找到最大的连通组件（主要组织区域）
    component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
    if component_sizes:
        main_component = np.argmax(component_sizes) + 1
        main_component_mask = (labeled_array == main_component)

        # 找到主要组件的边界
        rows, cols = np.where(main_component_mask)
        left_bound = np.min(cols) if len(cols) > 0 else 0
        right_bound = np.max(cols) if len(cols) > 0 else width
    else:
        left_bound = 0
        right_bound = width

    # 综合判断
    left_crop = 0
    right_crop = width

    # 如果边缘区域强度明显低于中心区域且梯度较低，则认为是背景
    left_is_background = (left_mean < center_mean * 0.6 and
                          left_gradient < center_gradient * 0.4)
    right_is_background = (right_mean < center_mean * 0.6 and
                           right_gradient < center_gradient * 0.4)

    if left_is_background:
        left_crop = max(0, left_bound - 5)  # 稍微扩大一点确保完全覆盖背景
    if right_is_background:
        right_crop = min(width, right_bound + 5)

    # 创建背景掩码
    background_mask = np.zeros_like(image, dtype=bool)
    if left_is_background:
        background_mask[:, :left_crop] = True
    if right_is_background:
        background_mask[:, right_crop:] = True

    return left_crop, right_crop, background_mask


def fill_background_with_max_value(image, background_mask):
    """
    将背景区域设为图像的最大值

    参数:
    - image: 输入图像
    - background_mask: 背景掩码

    返回:
    - processed_image: 处理后的图像
    """
    processed_image = image.copy()
    image_max = np.max(image)

    # 将背景区域设为最大值
    processed_image[background_mask] = image_max

    return processed_image


def save_as_dicom(processed_image, original_dicom, output_path):
    """
    将处理后的图像保存为DICOM格式

    参数:
    - processed_image: 处理后的图像数据
    - original_dicom: 原始DICOM对象
    - output_path: 输出DICOM文件路径
    """
    try:
        # 创建新的DICOM对象副本
        new_dicom = original_dicom.copy()

        # 确保图像数据格式正确
        if processed_image.dtype != original_dicom.pixel_array.dtype:
            # 转换为原始DICOM的数据类型
            processed_image = processed_image.astype(original_dicom.pixel_array.dtype)

        # 更新像素数据
        new_dicom.PixelData = processed_image.tobytes()

        # 更新图像尺寸相关信息
        new_dicom.Rows, new_dicom.Columns = processed_image.shape

        # 添加处理信息到DICOM头文件
        if hasattr(new_dicom, 'ImageComments'):
            new_dicom.ImageComments = f"Processed: Background set to max value - {original_dicom.ImageComments}"
        else:
            new_dicom.ImageComments = "Processed: Background regions set to maximum intensity value"

        # 添加处理标签
        new_dicom.SeriesDescription = f"Processed_{original_dicom.SeriesDescription}"

        # 保存DICOM文件
        new_dicom.save_as(output_path)
        return True

    except Exception as e:
        print(f"保存DICOM文件失败: {e}")
        return False


def save_as_png(processed_image, output_path):
    """
    将处理后的图像保存为PNG格式

    参数:
    - processed_image: 处理后的图像数据
    - output_path: 输出PNG文件路径
    """
    try:
        # 归一化到0-255范围
        if processed_image.dtype != np.uint8:
            image_normalized = (processed_image - processed_image.min()) / (
                        processed_image.max() - processed_image.min() + 1e-8) * 255
            image_uint8 = image_normalized.astype(np.uint8)
        else:
            image_uint8 = processed_image

        # 使用PIL保存为PNG
        pil_image = Image.fromarray(image_uint8)
        pil_image.save(output_path, 'PNG')
        return True

    except Exception as e:
        print(f"保存PNG文件失败: {e}")
        return False


def process_dicom_background_to_max(input_dir, output_dir, show_comparison=True):
    """
    处理DICOM图像：将背景区域设为最大值，保存PNG和DICOM格式

    参数:
    - input_dir: 输入DICOM图像目录
    - output_dir: 输出图像目录
    - show_comparison: 是否显示处理前后对比
    """
    # 创建输出目录
    png_dir = os.path.join(output_dir, "PNG")
    dicom_dir = os.path.join(output_dir, "DICOM")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(dicom_dir, exist_ok=True)

    # 获取所有DICOM文件
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有文件!")
        return

    print(f"找到 {len(dicom_files)} 个文件")
    print(f"PNG输出目录: {png_dir}")
    print(f"DICOM输出目录: {dicom_dir}")

    processed_count = 0
    error_count = 0

    for filename in tqdm(dicom_files, desc="背景转最大值处理"):
        try:
            # 读取DICOM文件
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            original_image = dicom_data.pixel_array.astype(np.float32)

            # 归一化到0-1范围以便处理
            image_normalized = (original_image - original_image.min()) / (
                        original_image.max() - original_image.min() + 1e-8)

            # 检测边缘背景
            left_crop, right_crop, background_mask = detect_edge_background_advanced(image_normalized)

            # 统计背景区域
            background_pixels = np.sum(background_mask)
            total_pixels = original_image.size
            background_ratio = background_pixels / total_pixels

            print(f"\n处理: {filename}")
            print(f"  检测到背景区域: {background_pixels} 像素 ({background_ratio * 100:.1f}%)")
            print(f"  有效区域: [{left_crop}, {right_crop}]")
            print(f"  图像最大值: {original_image.max():.2f}")

            # 将背景区域设为最大值
            if background_pixels > 0:
                processed_image = fill_background_with_max_value(original_image, background_mask)

                # 生成输出文件名
                base_name = os.path.splitext(filename)[0]

                # 保存PNG格式
                png_filename = f"{base_name}_processed.png"
                png_path = os.path.join(png_dir, png_filename)
                png_success = save_as_png(processed_image, png_path)

                # 保存DICOM格式
                dicom_filename = f"{base_name}_processed.dcm"
                dicom_path_output = os.path.join(dicom_dir, dicom_filename)
                dicom_success = save_as_dicom(processed_image, dicom_data, dicom_path_output)

                if png_success and dicom_success:
                    print(f"  ✓ 已保存PNG: {png_filename}")
                    print(f"  ✓ 已保存DICOM: {dicom_filename}")
                    processed_count += 1
                else:
                    print(f"  ✗ 文件保存失败")
                    error_count += 1

                # 显示对比
                if show_comparison and processed_count <= 3:
                    display_processing_comparison(original_image, processed_image, background_mask, filename)
            else:
                print(f"  未检测到明显背景区域，跳过处理")

                # 即使没有背景也保存文件
                base_name = os.path.splitext(filename)[0]

                # 保存PNG
                png_filename = f"{base_name}_processed.png"
                png_path = os.path.join(png_dir, png_filename)
                save_as_png(original_image, png_path)

                # 保存DICOM
                dicom_filename = f"{base_name}_processed.dcm"
                dicom_path_output = os.path.join(dicom_dir, dicom_filename)
                save_as_dicom(original_image, dicom_data, dicom_path_output)

                print(f"  ✓ 已保存文件（无背景处理）")
                processed_count += 1

        except Exception as e:
            print(f"错误处理文件 {filename}: {e}")
            error_count += 1

    # 打印处理总结
    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"{'=' * 60}")
    print(f"总文件数: {len(dicom_files)}")
    print(f"成功处理: {processed_count}")
    print(f"处理失败: {error_count}")
    print(f"PNG文件位置: {png_dir}")
    print(f"DICOM文件位置: {dicom_dir}")


def display_processing_comparison(original, processed, background_mask, filename):
    """
    显示处理前后的对比
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 原始图像
    im1 = ax1.imshow(original, cmap='gray')
    ax1.set_title(f'原始图像\n{filename}\n范围: [{original.min():.1f}, {original.max():.1f}]')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 处理后的图像
    im2 = ax2.imshow(processed, cmap='gray')
    ax2.set_title(f'背景设为最大值\n范围: [{processed.min():.1f}, {processed.max():.1f}]')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 背景掩码
    im3 = ax3.imshow(background_mask, cmap='Reds', alpha=0.7)
    ax3.imshow(original, cmap='gray', alpha=0.3)
    ax3.set_title('检测到的背景区域(红色覆盖)')
    ax3.axis('off')

    # 差异图像（突出显示变化）
    difference = processed - original
    im4 = ax4.imshow(difference, cmap='coolwarm',
                     vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
    ax4.set_title('差异图像\n(红色: 增加值, 蓝色: 减少值)')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    plt.tight_layout()
    plt.show()


def verify_output_files(output_dir):
    """
    验证输出文件
    """
    png_dir = os.path.join(output_dir, "PNG")
    dicom_dir = os.path.join(output_dir, "DICOM")

    png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]

    print(f"\n输出文件验证:")
    print(f"PNG文件数: {len(png_files)}")
    print(f"DICOM文件数: {len(dicom_files)}")

    if png_files and dicom_files:
        print("✓ 文件保存成功")

        # 显示示例文件
        print(f"\n示例文件:")
        print(f"PNG: {png_files[0]}")
        print(f"DICOM: {dicom_files[0]}")
    else:
        print("✗ 文件保存可能有问题")


def main():
    """
    主函数：将DICOM图像背景设为最大值并保存两种格式
    """
    # 设置路径
    input_dir = "D:/ai/test_0"  # 输入DICOM图像目录
    output_dir = "D:/ai/background_to_max"  # 输出目录

    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 选择处理模式
    print("\n选择处理模式:")
    print("1. 处理并保存PNG+DICOM")
    print("2. 仅验证输出文件")

    try:
        choice = input("请输入选择 (1/2, 默认1): ").strip()
        if choice == '':
            choice = '1'
    except:
        choice = '1'

    if choice == '1':
        # 处理并保存两种格式
        process_dicom_background_to_max(input_dir, output_dir, show_comparison=True)

        # 验证输出
        verify_output_files(output_dir)

    elif choice == '2':
        # 仅验证输出
        if os.path.exists(output_dir):
            verify_output_files(output_dir)
        else:
            print("输出目录不存在，请先运行处理程序")

    else:
        print("无效选择，使用默认模式")
        process_dicom_background_to_max(input_dir, output_dir)


if __name__ == "__main__":
    main()