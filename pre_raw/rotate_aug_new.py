import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import re
import logging
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_dicom_file(filepath):
    """读取无后缀的DICOM文件"""
    try:
        dicom_data = pydicom.dcmread(filepath)
        image_array = dicom_data.pixel_array
        return image_array, dicom_data
    except Exception as e:
        logger.error(f"读取DICOM文件失败 {filepath}: {e}")
        return None, None


def normalize_image(image):
    """将图像归一化到0-255范围"""
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min) * 255
        else:
            image = np.zeros_like(image)
        image = image.astype(np.uint8)
    return image


def ensure_numeric_fill_value(image, fill_value=None):
    """确保填充值是适合图像数据类型的数值"""
    if fill_value is None:
        fill_value = float(image.max())

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
    else:
        fill_value = float(fill_value)

    return fill_value


def rotate_image_center_fixed_size(image, angle_degrees, fill_value=None):
    """
    以图像中心为中心进行旋转，保持原始图像大小
    angle_degrees: 顺时针旋转角度（度）
    fill_value: 填充值，如果为None则使用图像最大值
    """
    height, width = image.shape[:2]
    fill_value = ensure_numeric_fill_value(image, fill_value)
    center = (width // 2, height // 2)

    # 注意：OpenCV中正角度表示逆时针旋转，所以需要取负值来实现顺时针旋转
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)

    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value)
    )

    return rotated_image


def parse_filename(filename):
    """
    解析文件名，支持多种格式
    格式1: ANY_204_1 (旧格式)
    格式2: ANY_204_1_0 (新格式，_0表示最小值投影，_1表示最大值投影等)
    格式3: ANY_204_1_0.tif (带扩展名的新格式)

    返回:
        prefix: 前缀 (如 "ANY")
        case_num: 病例编号 (如 204)
        angle_num: 角度编号 (如 1)
        projection_type: 投影类型 (如 "0" 或 "1")，旧格式返回None
        extension: 文件扩展名，无后缀返回空字符串
    """
    # 移除扩展名
    name_without_ext = os.path.splitext(filename)[0] if '.' in filename else filename
    extension = os.path.splitext(filename)[1] if '.' in filename else ''

    # 按下划线分割
    parts = name_without_ext.split('_')

    if len(parts) == 3:
        # 旧格式: ANY_204_1
        prefix = parts[0]
        case_num = parts[1]
        angle_num = parts[2]
        projection_type = None
        logger.debug(f"解析旧格式: {filename} -> 前缀={prefix}, 病例={case_num}, 角度={angle_num}")
    elif len(parts) >= 4:
        # 新格式: ANY_204_1_0 或 ANY_204_1_0_extra
        prefix = parts[0]
        case_num = parts[1]
        angle_num = parts[2]
        projection_type = parts[3]
        logger.debug(
            f"解析新格式: {filename} -> 前缀={prefix}, 病例={case_num}, 角度={angle_num}, 投影类型={projection_type}")
    else:
        # 无法解析的格式
        logger.warning(f"无法解析文件名格式: {filename} (分割后得到{len(parts)}部分)")
        return None, None, None, None, extension

    return prefix, case_num, angle_num, projection_type, extension


def generate_new_filename(original_filename, middle_num_offset=500):
    """
    根据原始文件名生成新的文件名
    新格式支持: ANY_204_1_0 -> ANY_704_1_0 (病例编号加500)
                ANY_204_1 -> ANY_704_1 (旧格式兼容)

    参数:
        original_filename: 原始文件名
        middle_num_offset: 病例编号的偏移量，默认500
    """
    # 解析文件名
    prefix, case_num, angle_num, projection_type, extension = parse_filename(original_filename)

    if prefix is None or case_num is None:
        # 解析失败，使用后备方案
        logger.warning(f"文件名格式不匹配 {original_filename}，在原文件名后添加'_rotated'")
        name_without_ext = os.path.splitext(original_filename)[0]
        ext = os.path.splitext(original_filename)[1] if '.' in original_filename else ''
        return f"{name_without_ext}_rotated{ext}"

    # 病例编号加偏移量
    try:
        new_case_num = int(case_num) + middle_num_offset
    except ValueError:
        logger.error(f"病例编号不是数字: {case_num}")
        return original_filename

    # 生成新文件名
    if projection_type is not None:
        # 新格式: ANY_204_1_0 -> ANY_704_1_0
        new_filename = f"{prefix}_{new_case_num}_{angle_num}_{projection_type}{extension}"
    else:
        # 旧格式: ANY_204_1 -> ANY_704_1
        new_filename = f"{prefix}_{new_case_num}_{angle_num}{extension}"

    logger.debug(f"文件名转换: {original_filename} -> {new_filename}")
    return new_filename


def get_matching_files(dicom_dir, mask_dir):
    """
    获取匹配的DICOM和mask文件对

    返回:
        matched_pairs: 列表，每个元素为(dicom_filename, mask_filename)
    """
    # 获取所有DICOM文件（无后缀）
    dicom_files = {}
    for f in os.listdir(dicom_dir):
        file_path = os.path.join(dicom_dir, f)
        if os.path.isfile(file_path):
            # 获取不带扩展名的文件名
            name = f if '.' not in f else os.path.splitext(f)[0]
            dicom_files[name] = f

    # 获取所有mask文件
    mask_files = {}
    for f in os.listdir(mask_dir):
        file_path = os.path.join(mask_dir, f)
        if os.path.isfile(file_path) and f.lower().endswith(('.tif', '.tiff')):
            # 获取不带扩展名的文件名
            name = os.path.splitext(f)[0]
            mask_files[name] = f

    # 找出匹配的文件对
    matched_names = set(dicom_files.keys()) & set(mask_files.keys())

    matched_pairs = []
    for name in sorted(matched_names):
        matched_pairs.append((dicom_files[name], mask_files[name]))

    logger.info(f"文件匹配结果:")
    logger.info(f"  DICOM文件总数: {len(dicom_files)}")
    logger.info(f"  Mask文件总数: {len(mask_files)}")
    logger.info(f"  匹配文件对: {len(matched_pairs)}")

    # 显示不匹配的文件
    dsa_only = set(dicom_files.keys()) - set(mask_files.keys())
    mask_only = set(mask_files.keys()) - set(dicom_files.keys())

    if dsa_only:
        logger.warning(f"仅有DICOM无mask的文件 ({len(dsa_only)}个):")
        for name in sorted(dsa_only)[:5]:
            logger.warning(f"  - {name}")
        if len(dsa_only) > 5:
            logger.warning(f"  ... 还有 {len(dsa_only) - 5} 个")

    if mask_only:
        logger.warning(f"仅有mask无DICOM的文件 ({len(mask_only)}个):")
        for name in sorted(mask_only)[:5]:
            logger.warning(f"  - {name}")
        if len(mask_only) > 5:
            logger.warning(f"  ... 还有 {len(mask_only) - 5} 个")

    return matched_pairs


def save_rotated_dicom(rotated_image, original_dicom_path, output_path):
    """保存旋转后的DICOM文件"""
    try:
        # 读取原始DICOM文件
        original_dicom = pydicom.dcmread(original_dicom_path)

        # 更新图像数据
        original_dicom.PixelData = rotated_image.tobytes()
        original_dicom.Rows, original_dicom.Columns = rotated_image.shape

        # 更新帧数信息
        if hasattr(original_dicom, 'NumberOfFrames'):
            original_dicom.NumberOfFrames = 1

        # 添加旋转信息到图像注释
        if 'ImageComments' in original_dicom:
            original_comment = original_dicom.ImageComments
        else:
            original_comment = ""
        original_dicom.ImageComments = f"Rotated - {original_comment}"

        # 保存为无后缀文件
        pydicom.dcmwrite(output_path, original_dicom)
        return True
    except Exception as e:
        logger.error(f"保存DICOM文件失败 {output_path}: {e}")
        return False


def process_images():
    """主处理函数"""

    # ========== 配置参数 ==========

    # 输入路径
    dicom_dir = r"D:\med_data\multi\preprocess\min\5\dicom"
    mask_dir = r"D:\med_data\multi\preprocess\min\5\mask"

    # 输出路径
    output_base = r"D:\med_data\multi\preprocess\min\5"
    output_dicom_dir = os.path.join(output_base, "dicom_rot_5")
    output_png_dir = os.path.join(output_base, "dicom_rot_5", "png")
    output_mask_dir = os.path.join(output_base, "mask_rot_5")

    # 旋转参数
    rotation_angle = -5  # 顺时针旋转角度
    case_num_offset = 500  # 病例编号增加量

    # ========== 初始化 ==========

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建输出目录: {dir_path}")

    # 获取匹配的文件对
    logger.info("正在分析文件匹配情况...")
    matched_pairs = get_matching_files(dicom_dir, mask_dir)

    if not matched_pairs:
        logger.error("未找到匹配的文件对，程序终止")
        return

    logger.info(f"开始处理 {len(matched_pairs)} 对文件")
    logger.info(f"旋转角度: {rotation_angle}° (顺时针)")
    logger.info(f"病例编号偏移: +{case_num_offset}")
    logger.info(f"文件命名示例:")

    # 显示命名示例
    if matched_pairs:
        example_dicom, example_mask = matched_pairs[0]
        old_name = example_dicom
        new_name = generate_new_filename(old_name, case_num_offset)
        logger.info(f"  {old_name} -> {new_name}")
        if '.' not in example_mask:
            old_mask_name = example_mask
        else:
            old_mask_name = os.path.splitext(example_mask)[0]
        new_mask_name = generate_new_filename(old_mask_name, case_num_offset)
        logger.info(f"  {example_mask} -> {new_mask_name}.tif")

    # ========== 处理文件 ==========

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for dicom_file, mask_file in tqdm(matched_pairs, desc="处理图像"):
        try:
            dicom_path = os.path.join(dicom_dir, dicom_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # 读取DICOM文件
            dicom_image, dicom_data = read_dicom_file(dicom_path)
            if dicom_image is None:
                logger.error(f"跳过 {dicom_file}: 无法读取DICOM文件")
                error_count += 1
                continue

            # 读取mask文件
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                logger.error(f"跳过 {dicom_file}: 无法读取mask文件 {mask_path}")
                error_count += 1
                continue

            logger.debug(f"处理: {dicom_file}")
            logger.debug(
                f"  DICOM: 形状={dicom_image.shape}, 类型={dicom_image.dtype}, 范围=[{dicom_image.min()}, {dicom_image.max()}]")
            logger.debug(
                f"  Mask: 形状={mask_image.shape}, 类型={mask_image.dtype}, 范围=[{mask_image.min()}, {mask_image.max()}]")

            # 旋转图像
            dicom_fill_value = float(dicom_image.max())
            mask_fill_value = 0.0

            rotated_dicom = rotate_image_center_fixed_size(dicom_image, rotation_angle, dicom_fill_value)
            rotated_mask = rotate_image_center_fixed_size(mask_image, rotation_angle, mask_fill_value)

            # 生成新文件名
            # DICOM文件名（无后缀）
            new_dicom_name = generate_new_filename(dicom_file, case_num_offset)

            # Mask文件名（保持扩展名）
            mask_name_without_ext = os.path.splitext(mask_file)[0]
            mask_ext = os.path.splitext(mask_file)[1]
            new_mask_base = generate_new_filename(mask_name_without_ext, case_num_offset)
            new_mask_name = f"{new_mask_base}{mask_ext}"

            # PNG文件名
            dicom_name_without_ext = dicom_file if '.' not in dicom_file else os.path.splitext(dicom_file)[0]
            new_png_base = generate_new_filename(dicom_name_without_ext, case_num_offset)
            new_png_name = f"{new_png_base}.png"

            # 保存旋转后的DICOM文件
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)
            if save_rotated_dicom(rotated_dicom, dicom_path, output_dicom_path):
                logger.debug(f"  已保存DICOM: {new_dicom_name}")
            else:
                logger.error(f"  保存DICOM失败: {new_dicom_name}")
                error_count += 1
                continue

            # 保存PNG格式（灰度图像，用于可视化）
            output_png_path = os.path.join(output_png_dir, new_png_name)
            normalized_dicom = normalize_image(rotated_dicom.copy())
            Image.fromarray(normalized_dicom).save(output_png_path, 'PNG')
            logger.debug(f"  已保存PNG: {new_png_name}")

            # 保存旋转后的mask文件
            output_mask_path = os.path.join(output_mask_dir, new_mask_name)
            cv2.imwrite(output_mask_path, rotated_mask)
            logger.debug(f"  已保存Mask: {new_mask_name}")

            processed_count += 1

            # 每处理10个文件输出一次进度
            if processed_count % 10 == 0:
                logger.info(f"已处理 {processed_count}/{len(matched_pairs)} 对文件")

        except Exception as e:
            logger.error(f"处理文件 {dicom_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    # ========== 输出统计 ==========

    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"总文件对: {len(matched_pairs)}")
    logger.info(f"成功处理: {processed_count}")
    logger.info(f"错误/跳过: {error_count}")
    logger.info(f"旋转角度: {rotation_angle}°")
    logger.info(f"病例编号偏移: +{case_num_offset}")
    logger.info(f"输出目录:")
    logger.info(f"  DICOM: {output_dicom_dir}")
    logger.info(f"  PNG: {output_png_dir}")
    logger.info(f"  Mask: {output_mask_dir}")

    # 验证输出文件
    output_dicom_count = len(
        [f for f in os.listdir(output_dicom_dir) if os.path.isfile(os.path.join(output_dicom_dir, f))])
    output_png_count = len([f for f in os.listdir(output_png_dir) if f.endswith('.png')])
    output_mask_count = len([f for f in os.listdir(output_mask_dir) if f.endswith(('.tif', '.tiff'))])

    logger.info(f"输出文件统计:")
    logger.info(f"  DICOM文件: {output_dicom_count}")
    logger.info(f"  PNG文件: {output_png_count}")
    logger.info(f"  Mask文件: {output_mask_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    process_images()