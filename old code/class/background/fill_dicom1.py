import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy import ndimage


def detect_edge_background_advanced(image, margin_ratio=0.1, intensity_threshold_ratio=0.1):
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
    left_is_background = (left_mean < center_mean * 0.7 and
                          left_gradient < center_gradient * 0.5)
    right_is_background = (right_mean < center_mean * 0.7 and
                           right_gradient < center_gradient * 0.5)

    if left_is_background:
        left_crop = left_bound
    if right_is_background:
        right_crop = right_bound

    # 创建背景掩码
    background_mask = np.zeros_like(image, dtype=bool)
    if left_is_background:
        background_mask[:, :left_crop] = True
    if right_is_background:
        background_mask[:, right_crop:] = True

    return left_crop, right_crop, background_mask


def fill_background_with_interpolation(image, background_mask, method='inpaint'):
    """
    使用插值方法填充背景区域

    参数:
    - image: 输入图像
    - background_mask: 背景掩码
    - method: 填充方法 ('inpaint', 'mean', 'median')

    返回:
    - filled_image: 填充后的图像
    """
    if method == 'inpaint' and cv2.__version__:
        # 使用OpenCV图像修复
        mask_uint8 = (background_mask * 255).astype(np.uint8)
        filled_image = cv2.inpaint(image.astype(np.float32), mask_uint8, 3, cv2.INPAINT_TELEA)
        return filled_image

    elif method == 'mean':
        # 使用邻域均值填充
        filled_image = image.copy()
        foreground_mean = np.mean(image[~background_mask])
        filled_image[background_mask] = foreground_mean
        return filled_image

    elif method == 'median':
        # 使用邻域中值填充
        filled_image = image.copy()
        foreground_median = np.median(image[~background_mask])
        filled_image[background_mask] = foreground_median
        return filled_image

    else:
        # 默认使用线性插值
        return fill_with_linear_interpolation(image, background_mask)


def fill_with_linear_interpolation(image, background_mask):
    """
    使用线性插值填充背景区域
    """
    filled_image = image.copy()
    height, width = image.shape

    for row in range(height):
        # 找到该行中的前景像素
        foreground_cols = np.where(~background_mask[row, :])[0]

        if len(foreground_cols) > 0:
            # 对于左侧背景
            left_foreground = foreground_cols[0]
            if left_foreground > 0:
                # 使用最近的右侧前景像素值填充左侧
                filled_image[row, :left_foreground] = image[row, left_foreground]

            # 对于右侧背景
            right_foreground = foreground_cols[-1]
            if right_foreground < width - 1:
                # 使用最近的左侧前景像素值填充右侧
                filled_image[row, right_foreground + 1:] = image[row, right_foreground]

    return filled_image


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

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有文件!")
        return

    print(f"找到 {len(dicom_files)} 个文件")

    processed_count = 0

    for filename in tqdm(dicom_files, desc="去除背景处理"):
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

            # 填充背景区域
            if background_pixels > 0:
                # 使用图像修复方法填充背景
                filled_image = fill_background_with_interpolation(original_image, background_mask, method='inpaint')

                # 保存处理后的图像
                output_filename = os.path.splitext(filename)[0] + '_nobg.png'
                output_path = os.path.join(output_dir, output_filename)

                # 归一化保存
                filled_normalized = (filled_image - filled_image.min()) / (
                            filled_image.max() - filled_image.min() + 1e-8) * 255
                Image.fromarray(filled_normalized.astype(np.uint8)).save(output_path)

                print(f"  已保存: {output_filename}")

                # 显示对比
                if show_comparison and processed_count < 3:
                    display_background_removal_comparison(original_image, filled_image, background_mask, filename)

                processed_count += 1
            else:
                print(f"  未检测到明显背景区域，跳过处理")

        except Exception as e:
            print(f"错误处理文件 {filename}: {e}")

    print(f"\n处理完成! 共处理 {processed_count} 个文件")


def display_background_removal_comparison(original, processed, background_mask, filename):
    """
    显示背景去除前后的对比
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 原始图像
    im1 = ax1.imshow(original, cmap='gray')
    ax1.set_title(f'原始图像\n{filename}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 处理后的图像
    im2 = ax2.imshow(processed, cmap='gray')
    ax2.set_title('背景去除后')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 背景掩码
    im3 = ax3.imshow(background_mask, cmap='Reds')
    ax3.set_title('检测到的背景区域(红色)')
    ax3.axis('off')

    # 差异图像
    difference = processed - original
    im4 = ax4.imshow(difference, cmap='coolwarm',
                     vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
    ax4.set_title('差异图像\n(处理后 - 原始)')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    plt.tight_layout()
    plt.show()


def analyze_background_distribution(input_dir):
    """
    分析图像中背景区域的分布情况
    """
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有文件!")
        return

    print(f"分析 {len(dicom_files)} 个文件的背景分布...")

    background_stats = []

    for filename in tqdm(dicom_files[:20], desc="背景分析"):  # 只分析前20个以节省时间
        try:
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)

            # 归一化
            image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # 检测背景
            left_crop, right_crop, background_mask = detect_edge_background_advanced(image_normalized)

            background_ratio = np.sum(background_mask) / image.size

            background_stats.append({
                'filename': filename,
                'background_ratio': background_ratio,
                'left_crop': left_crop,
                'right_crop': right_crop,
                'image_width': image.shape[1]
            })

        except Exception as e:
            print(f"错误分析文件 {filename}: {e}")

    # 显示分析结果
    if background_stats:
        background_ratios = [stat['background_ratio'] for stat in background_stats]

        print(f"\n背景分析结果:")
        print(f"平均背景比例: {np.mean(background_ratios) * 100:.2f}%")
        print(f"最大背景比例: {np.max(background_ratios) * 100:.2f}%")
        print(f"最小背景比例: {np.min(background_ratios) * 100:.2f}%")

        # 显示背景比例最高的文件
        sorted_stats = sorted(background_stats, key=lambda x: x['background_ratio'], reverse=True)
        print(f"\n背景比例最高的5个文件:")
        for i, stat in enumerate(sorted_stats[:5]):
            print(f"  {i + 1}. {stat['filename']}: {stat['background_ratio'] * 100:.1f}%")

    return background_stats


def main():
    """
    主函数：去除DICOM图像边缘背景
    """
    # 设置路径
    input_dir = "D:/ai/test_0"  # 输入DICOM图像目录
    output_dir = "D:/ai/processed_no_background"  # 输出目录

    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 选择处理模式
    print("\n选择处理模式:")
    print("1. 去除背景处理")
    print("2. 背景分布分析")

    try:
        choice = input("请输入选择 (1/2, 默认1): ").strip()
        if choice == '':
            choice = '1'
    except:
        choice = '1'

    if choice == '1':
        # 去除背景处理
        process_dicom_remove_background(input_dir, output_dir, show_comparison=True)

    elif choice == '2':
        # 背景分析
        analyze_background_distribution(input_dir)

    else:
        print("无效选择，使用默认模式")
        process_dicom_remove_background(input_dir, output_dir)


if __name__ == "__main__":
    main()