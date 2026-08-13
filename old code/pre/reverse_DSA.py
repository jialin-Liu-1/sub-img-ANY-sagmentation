import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
import cv2
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# 配置路径
INPUT_DIR = r"D:/med_data/multi/min_DSA"
OUTPUT_DCM_DIR = r"D:/med_data/multi/mix_DSA"
OUTPUT_PNG_DIR = r"D:/med_data/multi/min_location"

# 创建输出目录
Path(OUTPUT_DCM_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PNG_DIR).mkdir(parents=True, exist_ok=True)


def reverse_pixel_values(pixel_array):
    """
    反转像素灰度值：最小值变最大值，最大值变最小值
    基于图像本身的像素值范围
    """
    if pixel_array.size == 0:
        return pixel_array

    # 获取图像的实际像素值范围
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)

    # 如果已经是反转过的或者图像只有一个灰度值，直接返回
    if min_val == max_val:
        return pixel_array

    # 反转公式：new_pixel = min_val + max_val - old_pixel
    reversed_array = min_val + max_val - pixel_array

    return reversed_array.astype(pixel_array.dtype)


def save_as_png(pixel_array, output_path):
    """将像素数组保存为PNG文件（8位或16位自适应）"""
    # 归一化到0-255范围以便保存为PNG
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)

    if max_val > min_val:
        normalized = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)

    cv2.imwrite(output_path, normalized)


def process_single_file(file_path, filename):
    """处理单个DICOM文件：读取、反转、保存"""
    try:
        # 读取DICOM文件
        ds = pydicom.dcmread(file_path, force=True)

        # 获取像素数组
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
        else:
            # 有些DICOM需要先解压
            ds.decompress()
            pixel_array = ds.pixel_array

        # 反转像素值
        reversed_array = reverse_pixel_values(pixel_array)

        # 保存为DICOM格式（保持原文件名，无后缀）
        dcm_output_path = os.path.join(OUTPUT_DCM_DIR, filename)

        # 创建新的DICOM数据集（保留原元数据但更新像素数据）
        new_ds = Dataset()
        new_ds.update(ds)  # 复制所有元数据

        # 更新像素数据
        new_ds.PixelData = reversed_array.tobytes()
        new_ds['PixelData'].VR = 'OW'  # 或 'OB'，根据数据类型

        # 更新必要的DICOM字段
        new_ds.file_meta = ds.file_meta
        new_ds.is_little_endian = ds.is_little_endian
        new_ds.is_implicit_VR = ds.is_implicit_VR

        # 保存DICOM文件
        new_ds.save_as(dcm_output_path, write_like_original=False)

        # 保存为PNG格式
        png_output_path = os.path.join(OUTPUT_PNG_DIR, f"{filename}.png")
        save_as_png(reversed_array, png_output_path)

        return True, filename

    except Exception as e:
        return False, f"{filename}: {str(e)}"


def main():
    """主函数：批量处理所有DICOM文件"""

    # 获取所有文件（没有后缀的文件）
    all_files = []
    for item in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, item)
        if os.path.isfile(file_path) and not item.startswith('.'):
            # 检查文件是否包含病例名格式（可选）
            all_files.append((file_path, item))

    print(f"找到 {len(all_files)} 个文件待处理")

    # 方式1：使用多线程并行处理（推荐，速度快）
    print("开始处理（多线程模式）...")
    successful = 0
    failed = 0

    # 使用ThreadPoolExecutor进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, file_path, filename): filename
            for file_path, filename in all_files
        }

        # 使用tqdm显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_file),
                           total=len(all_files),
                           desc="处理进度"):
            success, result = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                tqdm.write(f"错误: {result}")

    print(f"\n处理完成！")
    print(f"成功: {successful} 个文件")
    print(f"失败: {failed} 个文件")
    print(f"DICOM保存位置: {OUTPUT_DCM_DIR}")
    print(f"PNG保存位置: {OUTPUT_PNG_DIR}")

    # 可选：方式2 - 顺序处理（如果内存有限，取消注释以下代码）
    """
    print("开始处理（顺序模式）...")
    successful = 0
    failed = 0
    for file_path, filename in tqdm(all_files, desc="处理进度"):
        success, result = process_single_file(file_path, filename)
        if success:
            successful += 1
        else:
            failed += 1
            print(f"错误: {result}")
    print(f"\n处理完成！成功: {successful}, 失败: {failed}")
    """


if __name__ == "__main__":
    main()