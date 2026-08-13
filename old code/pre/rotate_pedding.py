import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import math
import re


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
        fill_value = 0  # 改为0作为默认填充值

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


def rotate_image_center_fixed_size(image, angle_degrees, fill_value=0):
    """
    以图像中心为中心进行旋转，保持原始图像大小
    旋转后的空白部分用指定的fill_value填充

    Args:
        image: 输入图像
        angle_degrees: 顺时针旋转角度（度）
        fill_value: 填充值，默认为0

    Returns:
        旋转后的图像
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
                                   borderValue=float(fill_value))

    return rotated_image


def generate_new_filename(original_filename):
    """
    根据原始文件名生成新的文件名
    格式：ANY_204_1 -> ANY_704_1 (中间数字加500)
          ANY_216_0 -> ANY_716_0 (中间数字加500)
    """
    # 使用正则表达式匹配文件名模式
    # 匹配 ANY_数字_数字 的格式
    pattern = r'^(.*?)(\d+)(_\d+)$'
    match = re.match(pattern, original_filename)

    if match:
        prefix = match.group(1)  # "ANY_"
        middle_num = int(match.group(2))  # 204 或 216
        suffix = match.group(3)  # "_1" 或 "_0"

        # 中间数字加500
        new_middle_num = (middle_num)  # 如需加数字可以修改这里
        # + 3000

        # 生成新文件名
        new_filename = f"{prefix}{new_middle_num}{suffix}"
        print(f"  文件名转换: {original_filename} -> {new_filename}")
        return new_filename
    else:
        # 如果文件名格式不匹配，在原文件名后添加"_2"作为后备方案
        print(f"  警告: 文件名格式不匹配 {original_filename}，使用后备命名方案")
        return original_filename + "_2"


def create_white_mask_from_rotated(image_shape, rotation_matrix, angle_degrees):
    """
    创建旋转后空白区域的掩码（用于验证）
    """
    height, width = image_shape[:2]

    # 创建全1图像
    test_image = np.ones((height, width), dtype=np.uint8) * 255

    # 执行相同的旋转
    rotated_test = cv2.warpAffine(test_image, rotation_matrix, (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

    # 返回空白区域掩码（True表示空白区域）
    blank_mask = (rotated_test == 0)
    return blank_mask


def process_images():
    # 输入路径
    dicom_dir = r"D:\med_data\multi\test"  # DICOM医学图像目录
    mask_dir = r"D:\med_data\multi\test2_tif"  # TIF mask图像目录

    # 输出路径
    output_dicom_dir = r"D:\med_data\multi\test_rot"  # 旋转后的DICOM图像
    output_png_dir = r"D:\med_data\multi\test_rot\2png"  # 旋转后的PNG医学图像
    output_mask_dir = r"D:\med_data\multi\test_rot\mask2png"  # 旋转后的TIF mask图像

    # 旋转角度（顺时针旋转）
    rotation_angle = -17  # 顺时针旋转33度
    # 如果希望逆时针旋转，使用正角度

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print(f"旋转角度: {rotation_angle}° (顺时针)")
    print(f"空白区域填充值: 0")
    print(f"文件命名方式: 保持原文件名\n")

    processed_count = 0
    total_rotation_count = 0

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

            # 打印图像信息
            print(f"\n处理 {dicom_file}:")
            print(f"  DICOM - 尺寸: {dicom_image.shape}, 数据类型: {dicom_image.dtype}, 最大值: {dicom_image.max()}")
            print(f"  Mask - 尺寸: {mask_image.shape}, 数据类型: {mask_image.dtype}, 最大值: {mask_image.max()}")

            # 计算旋转矩阵（用于验证）
            height, width = dicom_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

            # 执行旋转操作，保持原始尺寸，用0填充空白区域
            rotated_dicom = rotate_image_center_fixed_size(dicom_image, -rotation_angle, fill_value=0)
            rotated_mask = rotate_image_center_fixed_size(mask_image, -rotation_angle, fill_value=0)

            # 验证空白区域是否被正确填充为0
            blank_mask = create_white_mask_from_rotated(dicom_image.shape, rotation_matrix, rotation_angle)

            # 统计空白区域和有效区域
            total_pixels = rotated_dicom.size
            blank_pixels = np.sum(blank_mask)
            valid_pixels = total_pixels - blank_pixels

            # 验证旋转后的图像空白区域是否为0
            blank_region_values = rotated_dicom[blank_mask]
            is_blank_zero = np.all(blank_region_values == 0) if blank_region_values.size > 0 else True

            print(f"  旋转信息:")
            print(f"    总像素数: {total_pixels}")
            print(f"    空白像素数: {blank_pixels} ({blank_pixels / total_pixels * 100:.1f}%)")
            print(f"    有效像素数: {valid_pixels} ({valid_pixels / total_pixels * 100:.1f}%)")
            print(f"    空白区域填充值验证: {'✓ 正确(均为0)' if is_blank_zero else '✗ 异常'}")

            # 检查旋转后的图像尺寸是否与原始相同
            if rotated_dicom.shape != dicom_image.shape:
                print(f"警告: DICOM图像尺寸改变 {dicom_image.shape} -> {rotated_dicom.shape}")
            if rotated_mask.shape != mask_image.shape:
                print(f"警告: Mask图像尺寸改变 {mask_image.shape} -> {rotated_mask.shape}")

            # 归一化DICOM图像用于PNG保存
            normalized_dicom = normalize_image(rotated_dicom.copy())

            # 生成新的文件名
            base_name = dicom_file  # 原文件名无后缀
            new_base_name = generate_new_filename(base_name)

            # 为不同格式生成文件名
            new_dicom_name = new_base_name  # 无后缀DICOM
            new_png_name = new_base_name + ".png"  # PNG格式
            new_mask_name = new_base_name + ".tif"  # TIF格式

            # 保存旋转后的DICOM文件
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)
            try:
                original_dicom = pydicom.dcmread(dicom_path)
                # 更新图像数据
                original_dicom.PixelData = rotated_dicom.tobytes()
                original_dicom.Rows, original_dicom.Columns = rotated_dicom.shape
                # 保存为无后缀文件
                original_dicom.save_as(output_dicom_path)
                print(f"  ✓ 已保存DICOM: {output_dicom_path}")
            except Exception as e:
                print(f"  ✗ 保存DICOM文件失败 {output_dicom_path}: {e}")
                # 如果DICOM保存失败，保存为RAW格式
                rotated_dicom.tofile(output_dicom_path + ".raw")
                print(f"  ✓ 已保存为RAW格式: {output_dicom_path}.raw")

            # 保存PNG格式的医学图像
            output_png_path = os.path.join(output_png_dir, new_png_name)
            Image.fromarray(normalized_dicom).save(output_png_path, 'PNG')
            print(f"  ✓ 已保存PNG: {output_png_path}")

            # 保存TIF格式的mask图像
            output_mask_path = os.path.join(output_mask_dir, new_mask_name)
            cv2.imwrite(output_mask_path, rotated_mask)
            print(f"  ✓ 已保存Mask: {output_mask_path}")

            processed_count += 1
            total_rotation_count += 1
            print(f"  ✅ 完成处理: {dicom_file} -> {new_base_name} (旋转{rotation_angle}°)\n")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print(f"处理完成!")
    print(f"成功处理: {processed_count} 组图像")
    print(f"旋转操作: {total_rotation_count} 次")
    print(f"填充值: 0")
    print(f"旋转后的DICOM文件保存在: {output_dicom_dir}")
    print(f"旋转后的PNG文件保存在: {output_png_dir}")
    print(f"旋转后的Mask文件保存在: {output_mask_dir}")
    print(f"{'=' * 60}")


def test_rotation_fill_value():
    """
    测试函数：验证旋转后的填充值是否为0
    """
    print("\n运行测试: 验证旋转填充值")
    print("-" * 40)

    # 创建测试图像
    test_image = np.ones((100, 100), dtype=np.uint8) * 255

    # 旋转45度
    rotated = rotate_image_center_fixed_size(test_image, 45, fill_value=0)

    # 检查是否有0值
    zero_count = np.sum(rotated == 0)
    non_zero_count = np.sum(rotated > 0)

    print(f"测试图像尺寸: {test_image.shape}")
    print(f"旋转后图像尺寸: {rotated.shape}")
    print(f"零值像素数: {zero_count}")
    print(f"非零像素数: {non_zero_count}")
    print(f"总像素数: {rotated.size}")

    if zero_count > 0:
        print("✓ 测试通过: 旋转后存在填充的零值区域")
    else:
        print("✗ 测试失败: 旋转后没有填充区域")

    return rotated


if __name__ == "__main__":
    # 可选：运行测试验证功能
    # test_rotation_fill_value()

    # 主处理流程
    process_images()