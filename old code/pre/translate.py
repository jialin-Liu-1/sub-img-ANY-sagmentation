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


def translate_image_vertical(image, shift_pixels, fill_value=None):
    """
    上下平移图像，保持原始图像大小
    shift_pixels: 平移像素数，正数向上平移，负数向下平移
    fill_value: 填充值
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 确保填充值是合适的数据类型
    fill_value = ensure_numeric_fill_value(image, fill_value)

    # 创建平移矩阵 [1, 0, tx; 0, 1, ty]
    # 对于垂直平移，tx=0，ty=shift_pixels
    translation_matrix = np.float32([[1, 0, 0], [0, 1, -shift_pixels]])  # 负号是因为OpenCV的坐标系：正y向下

    # 执行平移，保持原始图像大小，用指定值填充空白区域
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=float(fill_value))

    return translated_image


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
        new_middle_num = middle_num + 10000

        # 生成新文件名
        new_filename = f"{prefix}{new_middle_num}{suffix}"
        print(f"  文件名转换: {original_filename} -> {new_filename}")
        return new_filename
    else:
        # 如果文件名格式不匹配，在原文件名后添加"_2"作为后备方案
        print(f"  警告: 文件名格式不匹配 {original_filename}，使用后备命名方案")
        return original_filename + "_2"


def process_images():
    # 输入路径
    dicom_dir = r"D:\med_data\ai\train11"  # DICOM医学图像目录
    tif_dir = r"D:\med_data\ai\train22"  # TIF图像目录

    # 输出路径
    output_dicom_dir = r"D:\med_data\ai\train1(1)"  # 平移后的DICOM文件
    output_png_dir = r"D:\med_data\ai\train1PNG"  # 平移后的PNG文件
    output_tif_dir = r"D:\med_data\ai\train2(1)"  # 平移后的TIF文件

    # 平移参数
    # 所有图像尺寸为[512, 512]，平移15%为77像素
    shift_percentage = 15  # 平移比例（%）
    shift_pixels = int(512 * shift_percentage / 100)  # 计算平移像素数

    # 设置平移方向：正数向上平移，负数向下平移
    # 默认向上平移77像素
    shift_direction = 1  # 1表示向上，-1表示向下

    # 计算最终的平移像素数
    final_shift_pixels = shift_direction * shift_pixels

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_tif_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print(f"图像尺寸: 512×512")
    print(f"平移比例: {shift_percentage}% ({shift_pixels}像素)")
    print(f"平移方向: {'向上' if final_shift_pixels > 0 else '向下'} ({abs(final_shift_pixels)}像素)")
    print(f"DICOM空白填充: 图像最大值")
    print(f"TIF空白填充: 0")
    print(f"文件命名方式: 原文件名中间数字 + 500\n")

    processed_count = 0

    for dicom_file in dicom_files:
        try:
            # 构建文件路径
            dicom_path = os.path.join(dicom_dir, dicom_file)
            tif_path = os.path.join(tif_dir, dicom_file + ".tif")

            # 检查TIF文件是否存在
            if not os.path.exists(tif_path):
                print(f"警告: 找不到对应的TIF文件 {tif_path}")
                continue

            # 读取DICOM文件
            dicom_image = read_dicom_file(dicom_path)
            if dicom_image is None:
                continue

            # 读取TIF文件
            tif_image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
            if tif_image is None:
                print(f"读取TIF文件失败: {tif_path}")
                continue

            # 打印图像数据类型信息
            print(f"处理 {dicom_file}:")
            print(f"  DICOM - 尺寸: {dicom_image.shape}, 数据类型: {dicom_image.dtype}, 最大值: {dicom_image.max()}")
            print(f"  TIF - 尺寸: {tif_image.shape}, 数据类型: {tif_image.dtype}, 最大值: {tif_image.max()}")

            # 计算DICOM图像的最大值（用于填充空白区域）
            dicom_max_value = float(dicom_image.max())
            # TIF图像使用0填充
            tif_fill_value = 0.0

            # 执行平移操作
            translated_dicom = translate_image_vertical(dicom_image, final_shift_pixels, dicom_max_value)
            translated_tif = translate_image_vertical(tif_image, final_shift_pixels, tif_fill_value)

            # 检查平移后的图像尺寸是否与原始相同
            if translated_dicom.shape != dicom_image.shape:
                print(f"警告: DICOM图像尺寸改变 {dicom_image.shape} -> {translated_dicom.shape}")
            if translated_tif.shape != tif_image.shape:
                print(f"警告: TIF图像尺寸改变 {tif_image.shape} -> {translated_tif.shape}")

            # 归一化DICOM图像用于PNG保存
            normalized_dicom = normalize_image(translated_dicom.copy())

            # 生成新的文件名（中间数字加500）
            base_name = dicom_file  # 原文件名无后缀
            new_base_name = generate_new_filename(base_name)

            # 为不同格式生成文件名
            new_dicom_name = new_base_name  # 无后缀DICOM
            new_png_name = new_base_name + ".png"  # PNG格式
            new_tif_name = new_base_name + ".tif"  # TIF格式

            # 保存平移后的DICOM文件（保持原始格式）
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)

            # 对于DICOM文件，使用pydicom保存
            try:
                original_dicom = pydicom.dcmread(dicom_path)
                # 更新图像数据
                original_dicom.PixelData = translated_dicom.tobytes()
                original_dicom.Rows, original_dicom.Columns = translated_dicom.shape
                # 保存为无后缀文件
                original_dicom.save_as(output_dicom_path)
                print(f"  已保存DICOM: {output_dicom_path}")
            except Exception as e:
                print(f"  保存DICOM文件失败 {output_dicom_path}: {e}")
                # 如果DICOM保存失败，保存为RAW格式
                translated_dicom.tofile(output_dicom_path + ".raw")
                print(f"  已保存为RAW格式: {output_dicom_path}.raw")

            # 保存PNG格式的医学图像
            output_png_path = os.path.join(output_png_dir, new_png_name)
            Image.fromarray(normalized_dicom).save(output_png_path, 'PNG')
            print(f"  已保存PNG: {output_png_path}")

            # 保存TIF格式的图像
            output_tif_path = os.path.join(output_tif_dir, new_tif_name)
            cv2.imwrite(output_tif_path, translated_tif)
            print(f"  已保存TIF: {output_tif_path}")

            processed_count += 1
            print(f"  完成处理: {dicom_file} -> {new_base_name} (平移{final_shift_pixels}像素)\n")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n处理完成! 成功处理 {processed_count} 组图像")
    print(f"平移后的DICOM文件保存在: {output_dicom_dir}")
    print(f"平移后的PNG文件保存在: {output_png_dir}")
    print(f"平移后的TIF文件保存在: {output_tif_dir}")


if __name__ == "__main__":
    process_images()