import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import math


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


def ensure_numeric_fill_value(image, fill_value=None):
    """
    确保填充值是适合图像数据类型的数值
    """
    if fill_value is None:
        fill_value = image.max()

    # 根据图像数据类型转换填充值
    if image.dtype == np.uint8:
        fill_value = np.uint8(fill_value)
    elif image.dtype == np.uint16:
        fill_value = np.uint16(fill_value)
    elif image.dtype == np.int16:
        fill_value = np.int16(fill_value)
    elif image.dtype == np.float32:
        fill_value = np.float32(fill_value)
    elif image.dtype == np.float64:
        fill_value = np.float64(fill_value)

    return fill_value


def rotate_image_center_fixed_size(image, angle_degrees, fill_value=None):
    """
    以图像中心为中心进行旋转，保持原始图像大小
    angle_degrees: 顺时针旋转角度（度）
    fill_value: 填充值，如果为None则使用图像最大值
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 确保填充值是合适的数据类型
    fill_value = ensure_numeric_fill_value(image, fill_value)

    # 计算旋转中心
    center = (width // 2, height // 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # 执行旋转，保持原始图像大小，用指定值填充空白区域
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=float(fill_value))  # 确保转换为float

    return rotated_image


def process_images():
    # 输入路径
    dicom_dir = r"D:\med_data\ai\preprocess\aug\1"  # DICOM医学图像目录
    mask_dir = r"D:\med_data\ai\preprocess\aug\1(1)"  # TIF mask图像目录

    # 输出路径
    output_dicom_dir = r"D:\med_data\ai\preprocess\aug\1(2)"  # 旋转后的DICOM图像
    output_png_dir = r"D:\med_data\ai\preprocess\aug\1png"  # 旋转后的PNG医学图像
    output_mask_dir = r"D:\med_data\ai\preprocess\aug\1(3)"  # 旋转后的TIF mask图像

    # 旋转角度（顺时针20度）
    rotation_angle = 5

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print(f"旋转角度: {rotation_angle}° (顺时针)")
    print(f"空白区域填充: 图像最大值")

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

            # 打印图像数据类型信息
            print(f"处理 {dicom_file}:")
            print(f"  DICOM - 尺寸: {dicom_image.shape}, 数据类型: {dicom_image.dtype}, 最大值: {dicom_image.max()}")
            print(f"  Mask - 尺寸: {mask_image.shape}, 数据类型: {mask_image.dtype}, 最大值: {mask_image.max()}")

            # 计算DICOM图像的最大值（用于填充空白区域）
            dicom_max_value = float(dicom_image.max())  # 转换为float确保兼容性
            # mask图像使用0填充（因为mask通常是二值图像）
            mask_fill_value = 0.0  # 使用float

            # 执行旋转操作，保持原始尺寸，用最大值填充空白区域
            rotated_dicom = rotate_image_center_fixed_size(dicom_image, -rotation_angle, dicom_max_value)
            rotated_mask = rotate_image_center_fixed_size(mask_image, -rotation_angle, mask_fill_value)

            # 检查旋转后的图像尺寸是否与原始相同
            if rotated_dicom.shape != dicom_image.shape:
                print(f"警告: DICOM图像尺寸改变 {dicom_image.shape} -> {rotated_dicom.shape}")
            if rotated_mask.shape != mask_image.shape:
                print(f"警告: Mask图像尺寸改变 {mask_image.shape} -> {rotated_mask.shape}")

            # 归一化DICOM图像用于PNG保存（注意：这里只用于可视化）
            normalized_dicom = normalize_image(rotated_dicom.copy())  # 使用copy避免修改原始数据

            # 生成新的文件名（在原文件名后添加"_rot20"）
            base_name = dicom_file  # 原文件名无后缀
            new_dicom_name = base_name + "_2"  # 无后缀DICOM
            new_png_name = base_name + "_2.png"  # PNG格式
            new_mask_name = base_name + "_2.tif"  # TIF格式

            # 保存旋转后的DICOM文件（保持原始格式）
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)

            # 对于DICOM文件，我们需要使用pydicom保存
            try:
                original_dicom = pydicom.dcmread(dicom_path)
                # 更新图像数据
                original_dicom.PixelData = rotated_dicom.tobytes()
                original_dicom.Rows, original_dicom.Columns = rotated_dicom.shape
                # 保存为无后缀文件
                original_dicom.save_as(output_dicom_path)
                print(f"  已保存DICOM: {output_dicom_path}")
            except Exception as e:
                print(f"  保存DICOM文件失败 {output_dicom_path}: {e}")
                # 如果DICOM保存失败，保存为RAW格式
                rotated_dicom.tofile(output_dicom_path + ".raw")
                print(f"  已保存为RAW格式: {output_dicom_path}.raw")

            # 保存PNG格式的医学图像
            output_png_path = os.path.join(output_png_dir, new_png_name)
            # 使用PIL保存PNG以获得更好的质量
            Image.fromarray(normalized_dicom).save(output_png_path, 'PNG')
            print(f"  已保存PNG: {output_png_path}")

            # 保存TIF格式的mask图像
            output_mask_path = os.path.join(output_mask_dir, new_mask_name)
            cv2.imwrite(output_mask_path, rotated_mask)
            print(f"  已保存Mask: {output_mask_path}")

            processed_count += 1
            print(f"  完成处理: {dicom_file} -> 旋转{rotation_angle}°\n")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            continue

    print(f"\n处理完成! 成功处理 {processed_count} 组图像")
    print(f"旋转后的DICOM文件保存在: {output_dicom_dir}")
    print(f"旋转后的PNG文件保存在: {output_png_dir}")
    print(f"旋转后的Mask文件保存在: {output_mask_dir}")


if __name__ == "__main__":
    process_images()