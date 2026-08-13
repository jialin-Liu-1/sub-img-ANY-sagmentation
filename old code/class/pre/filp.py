import os
import pydicom
import numpy as np
from PIL import Image
import cv2


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


def process_images():
    # 输入路径
    dicom_dir = r"D:/med_data/ai/background/D"  # DICOM医学图像目录
    mask_dir = r"D:/med_data/ai/mask1"  # TIF mask图像目录

    # 输出路径
    output_dicom_dir = r"D:/med_data/ai/flip_dicom"  # 翻转后的DICOM图像
    output_png_dir = r"D:/med_data/ai/flip_png"  # 翻转后的PNG医学图像
    output_mask_dir = r"D:/med_data/ai/flip_mask"  # 翻转后的TIF mask图像

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"找到 {len(dicom_files)} 个DICOM文件")

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

            # 执行左右翻转
            flipped_dicom = cv2.flip(dicom_image, 1)  # 1表示水平翻转
            flipped_mask = cv2.flip(mask_image, 1)

            # 归一化DICOM图像用于PNG保存
            normalized_dicom = normalize_image(flipped_dicom)

            # 生成新的文件名（在原文件名后添加"_1"）
            base_name = dicom_file  # 原文件名无后缀
            new_dicom_name = base_name + "_1"  # 无后缀DICOM
            new_png_name = base_name + "_1.png"  # PNG格式
            new_mask_name = base_name + "_1.tif"  # TIF格式

            # 保存翻转后的DICOM文件（保持原始格式）
            # 由于原始DICOM无后缀，我们也不加后缀保存
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)

            # 对于DICOM文件，我们需要使用pydicom保存
            try:
                original_dicom = pydicom.dcmread(dicom_path)
                original_dicom.PixelData = flipped_dicom.tobytes()
                original_dicom.Rows, original_dicom.Columns = flipped_dicom.shape
                original_dicom.save_as(output_dicom_path)
            except Exception as e:
                print(f"保存DICOM文件失败 {output_dicom_path}: {e}")
                # 如果DICOM保存失败，保存为RAW格式
                flipped_dicom.tofile(output_dicom_path + ".raw")
                print(f"已保存为RAW格式: {output_dicom_path}.raw")

            # 保存PNG格式的医学图像
            output_png_path = os.path.join(output_png_dir, new_png_name)
            cv2.imwrite(output_png_path, normalized_dicom)

            # 保存TIF格式的mask图像
            output_mask_path = os.path.join(output_mask_dir, new_mask_name)
            cv2.imwrite(output_mask_path, flipped_mask)

            processed_count += 1
            print(f"已处理: {dicom_file} -> 翻转图像已保存")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            continue

    print(f"\n处理完成! 成功处理 {processed_count} 组图像")
    print(f"DICOM文件保存在: {output_dicom_dir}")
    print(f"PNG文件保存在: {output_png_dir}")
    print(f"Mask文件保存在: {output_mask_dir}")


if __name__ == "__main__":

    process_images()