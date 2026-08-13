import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json


def read_dicom_file(filepath):
    """读取无后缀的DICOM文件"""
    try:
        dicom_data = pydicom.dcmread(filepath)
        image_array = dicom_data.pixel_array
        return image_array
    except Exception as e:
        print(f"读取DICOM文件失败 {filepath}: {e}")
        return None


def normalize_image(image):
    """将图像归一化到0-255范围"""
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)
    return image


def calculate_distributions():
    # 输入路径
    dicom_dir = r"D:\ai1\processed_data\D"  # DICOM医学图像目录
    mask_dir = r"D:\ai1\mask2"  # TIF mask图像目录

    # 输出路径
    output_dir = r"D:\ai\Distribution"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化统计变量
    pixel_values = []  # 所有DICOM图像的像素值
    aneurysm_areas = []  # 动脉瘤面积占比
    aneurysm_centers = []  # 动脉瘤中心点坐标
    aneurysm_pixel_values = []  # 动脉瘤区域的像素值

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print("开始计算分布图...")

    processed_count = 0

    for dicom_file in dicom_files:
        try:
            # 构建文件路径
            dicom_path = os.path.join(dicom_dir, dicom_file)
            mask_path = os.path.join(mask_dir, dicom_file + ".tif")

            # 检查mask文件是否存在
            if not os.path.exists(mask_path):
                print(f"警告: 找不到对应的mask文件 {mask_path}")
                continue

            # 读取DICOM文件
            dicom_image = read_dicom_file(dicom_path)
            if dicom_image is None:
                continue

            # 读取mask文件
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                print(f"读取mask文件失败: {mask_path}")
                continue

            # 确保mask是二值图像
            mask_binary = (mask_image > 0).astype(np.uint8)

            # 1. 收集DICOM图像像素值
            pixel_values.extend(dicom_image.flatten())

            # 2. 计算动脉瘤面积占比
            total_pixels = mask_binary.size
            aneurysm_pixels = np.sum(mask_binary)
            aneurysm_ratio = aneurysm_pixels / total_pixels
            aneurysm_areas.append(aneurysm_ratio)

            # 3. 计算动脉瘤中心点
            if aneurysm_pixels > 0:
                # 找到所有非零像素的坐标
                y_coords, x_coords = np.where(mask_binary > 0)
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                aneurysm_centers.append((center_x, center_y))

            # 4. 计算动脉瘤区域的像素值
            if aneurysm_pixels > 0:
                # 将DICOM图像与mask相乘，提取动脉瘤区域
                aneurysm_region = dicom_image * mask_binary
                # 只取非零值（动脉瘤区域）
                aneurysm_pixels_values = aneurysm_region[aneurysm_region > 0]
                aneurysm_pixel_values.extend(aneurysm_pixels_values)

            processed_count += 1
            print(f"已处理 {processed_count}/{len(dicom_files)}: {dicom_file}")

            # 每处理100个文件就输出一次中间结果（可选）
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个文件，生成中间分布图...")
                generate_distribution_plots(pixel_values, aneurysm_areas,
                                            aneurysm_centers, aneurysm_pixel_values,
                                            output_dir, suffix=f"_intermediate_{processed_count}")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            continue

    print(f"\n处理完成! 成功处理 {processed_count} 组图像")

    # 生成最终的分布图
    generate_distribution_plots(pixel_values, aneurysm_areas,
                                aneurysm_centers, aneurysm_pixel_values,
                                output_dir)

    # 保存统计信息到文本文件
    save_statistics(pixel_values, aneurysm_areas, aneurysm_centers,
                    aneurysm_pixel_values, output_dir, processed_count)


def generate_distribution_plots(pixel_values, aneurysm_areas, aneurysm_centers,
                                aneurysm_pixel_values, output_dir, suffix=""):
    """生成并保存所有分布图"""

    # 设置中文字体（如果需要显示中文标签）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. DICOM图像的像素平均值分布图
    if pixel_values:
        plt.figure(figsize=(10, 6))
        plt.hist(pixel_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('像素值')
        plt.ylabel('频数')
        plt.title('DICOM图像像素值分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'pixel_value_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 2. 动脉瘤面积占比分布图
    if aneurysm_areas:
        plt.figure(figsize=(10, 6))
        plt.hist(aneurysm_areas, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('动脉瘤面积占比')
        plt.ylabel('频数')
        plt.title('动脉瘤面积占比分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_area_ratio_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 动脉瘤位置分布图（中心点叠加）
    if aneurysm_centers:
        # 创建一个空的热力图
        fig, ax = plt.subplots(figsize=(10, 10))

        # 提取所有中心点的x和y坐标
        x_coords = [center[0] for center in aneurysm_centers]
        y_coords = [center[1] for center in aneurysm_centers]

        # 创建2D直方图（热力图）
        hb = ax.hist2d(x_coords, y_coords, bins=50, cmap='hot')
        plt.colorbar(hb[3], ax=ax, label='频数')

        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title('动脉瘤位置分布热力图')
        ax.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, f'aneurysm_location_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 另外保存一个散点图版本
        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, alpha=0.5, s=1)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('动脉瘤中心点分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_center_scatter{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 4. 动脉瘤处像素平均值分布图
    if aneurysm_pixel_values:
        plt.figure(figsize=(10, 6))
        plt.hist(aneurysm_pixel_values, bins=100, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('像素值')
        plt.ylabel('频数')
        plt.title('动脉瘤区域像素值分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_pixel_value_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def save_statistics(pixel_values, aneurysm_areas, aneurysm_centers,
                    aneurysm_pixel_values, output_dir, processed_count):
    """保存统计信息到文本文件"""

    stats = {
        "processed_images": processed_count,
        "total_pixels_analyzed": len(pixel_values),
        "images_with_aneurysm": len(aneurysm_areas),
        "aneurysm_centers_found": len(aneurysm_centers),
        "aneurysm_pixels_analyzed": len(aneurysm_pixel_values)
    }

    if pixel_values:
        stats["dicom_pixel_stats"] = {
            "mean": float(np.mean(pixel_values)),
            "std": float(np.std(pixel_values)),
            "min": float(np.min(pixel_values)),
            "max": float(np.max(pixel_values)),
            "median": float(np.median(pixel_values))
        }

    if aneurysm_areas:
        stats["aneurysm_area_stats"] = {
            "mean_ratio": float(np.mean(aneurysm_areas)),
            "std_ratio": float(np.std(aneurysm_areas)),
            "min_ratio": float(np.min(aneurysm_areas)),
            "max_ratio": float(np.max(aneurysm_areas)),
            "median_ratio": float(np.median(aneurysm_areas))
        }

    if aneurysm_pixel_values:
        stats["aneurysm_pixel_stats"] = {
            "mean": float(np.mean(aneurysm_pixel_values)),
            "std": float(np.std(aneurysm_pixel_values)),
            "min": float(np.min(aneurysm_pixel_values)),
            "max": float(np.max(aneurysm_pixel_values)),
            "median": float(np.median(aneurysm_pixel_values))
        }

    # 保存为JSON文件
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    # 保存为文本文件
    with open(os.path.join(output_dir, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("DICOM图像统计分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理图像数量: {stats['processed_images']}\n")
        f.write(f"分析像素总数: {stats['total_pixels_analyzed']}\n")
        f.write(f"包含动脉瘤的图像数量: {stats['images_with_aneurysm']}\n")
        f.write(f"找到的动脉瘤中心点数量: {stats['aneurysm_centers_found']}\n")
        f.write(f"分析的动脉瘤区域像素数量: {stats['aneurysm_pixels_analyzed']}\n\n")

        if 'dicom_pixel_stats' in stats:
            f.write("DICOM图像像素统计:\n")
            for key, value in stats['dicom_pixel_stats'].items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write("\n")

        if 'aneurysm_area_stats' in stats:
            f.write("动脉瘤面积占比统计:\n")
            for key, value in stats['aneurysm_area_stats'].items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

        if 'aneurysm_pixel_stats' in stats:
            f.write("动脉瘤区域像素统计:\n")
            for key, value in stats['aneurysm_pixel_stats'].items():
                f.write(f"  {key}: {value:.2f}\n")


if __name__ == "__main__":
    calculate_distributions()