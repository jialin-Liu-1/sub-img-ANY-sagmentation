import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import re
import logging
from tqdm import tqdm
import shutil

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


def flip_image_horizontal(image):
    """
    左右翻转图像

    参数:
        image: 输入图像数组

    返回:
        flipped_image: 左右翻转后的图像
    """
    # 使用numpy进行左右翻转（沿列轴翻转）
    flipped_image = np.fliplr(image)
    return flipped_image


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


def generate_new_filename(original_filename, case_num_offset=5000):
    """
    根据原始文件名生成新的文件名（病例编号加偏移量）
    新格式支持: ANY_204_1_0 -> ANY_5204_1_0 (病例编号加5000)
                ANY_204_1 -> ANY_5204_1 (旧格式兼容)

    参数:
        original_filename: 原始文件名
        case_num_offset: 病例编号的偏移量，默认5000
    """
    # 解析文件名
    prefix, case_num, angle_num, projection_type, extension = parse_filename(original_filename)

    if prefix is None or case_num is None:
        # 解析失败，使用后备方案
        logger.warning(f"文件名格式不匹配 {original_filename}，在原文件名后添加'_flipped'")
        name_without_ext = os.path.splitext(original_filename)[0]
        ext = os.path.splitext(original_filename)[1] if '.' in original_filename else ''
        return f"{name_without_ext}_flipped{ext}"

    # 病例编号加偏移量
    try:
        new_case_num = int(case_num) + case_num_offset
    except ValueError:
        logger.error(f"病例编号不是数字: {case_num}")
        return original_filename

    # 生成新文件名
    if projection_type is not None:
        # 新格式: ANY_204_1_0 -> ANY_5204_1_0
        new_filename = f"{prefix}_{new_case_num}_{angle_num}_{projection_type}{extension}"
    else:
        # 旧格式: ANY_204_1 -> ANY_5204_1
        new_filename = f"{prefix}_{new_case_num}_{angle_num}{extension}"

    logger.debug(f"文件名转换: {original_filename} -> {new_filename}")
    return new_filename


def get_matching_files(dicom_dir, mask_dir):
    """
    获取匹配的DICOM和mask文件对

    参数:
        dicom_dir: DICOM文件目录
        mask_dir: Mask文件目录

    返回:
        matched_pairs: 列表，每个元素为(dicom_filename, mask_filename)
    """
    # 获取所有DICOM文件（无后缀或其他格式）
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
        if os.path.isfile(file_path):
            # 获取不带扩展名的文件名（支持.tif, .png等多种格式）
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


def save_flipped_dicom(flipped_image, original_dicom_path, output_path):
    """
    保存左右翻转后的DICOM文件

    参数:
        flipped_image: 翻转后的图像数组
        original_dicom_path: 原始DICOM文件路径
        output_path: 输出文件路径
    """
    try:
        # 读取原始DICOM文件
        original_dicom = pydicom.dcmread(original_dicom_path)

        # 更新图像数据
        original_dicom.PixelData = flipped_image.tobytes()
        original_dicom.Rows, original_dicom.Columns = flipped_image.shape

        # 更新帧数信息
        if hasattr(original_dicom, 'NumberOfFrames'):
            original_dicom.NumberOfFrames = 1

        # 添加翻转信息到图像注释
        if 'ImageComments' in original_dicom:
            original_comment = original_dicom.ImageComments
        else:
            original_comment = ""
        original_dicom.ImageComments = f"Flipped LR - {original_comment}"

        # 更新序列描述
        if hasattr(original_dicom, 'SeriesDescription'):
            original_series_desc = original_dicom.SeriesDescription
            original_dicom.SeriesDescription = f"{original_series_desc} Flipped"

        # 保存为无后缀文件
        pydicom.dcmwrite(output_path, original_dicom)
        return True
    except Exception as e:
        logger.error(f"保存DICOM文件失败 {output_path}: {e}")
        return False


def save_flipped_mask(flipped_mask, original_mask_path, output_path):
    """
    保存左右翻转后的mask文件
    保持原始文件格式

    参数:
        flipped_mask: 翻转后的mask数组
        original_mask_path: 原始mask文件路径
        output_path: 输出文件路径
    """
    try:
        # 根据输出文件扩展名选择保存方式
        ext = os.path.splitext(output_path)[1].lower()

        if ext in ['.tif', '.tiff']:
            # 保存为TIFF格式
            Image.fromarray(flipped_mask).save(output_path)
        elif ext == '.png':
            # 保存为PNG格式
            cv2.imwrite(output_path, flipped_mask)
        else:
            # 默认使用OpenCV保存
            cv2.imwrite(output_path, flipped_mask)

        return True
    except Exception as e:
        logger.error(f"保存Mask文件失败 {output_path}: {e}")
        return False


def process_flip_images():
    """主处理函数：左右翻转DSA和mask图像"""

    # ========== 配置参数 ==========

    # 输入路径
    dicom_dir = r"D:\med_data\multi\preprocess\min\7\dicom"
    mask_dir = r"D:\med_data\multi\preprocess\min\7\mask"

    # 输出路径
    output_dicom_dir = r"D:\med_data\multi\preprocess\min\7\all_DSA_flip"
    output_mask_dir = r"D:\med_data\multi\preprocess\min\7\all_mask_flip"

    # 病例编号偏移量
    case_num_offset = 5000

    # ========== 初始化 ==========

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建输出目录: {dir_path}")

    # 获取匹配的文件对
    logger.info("正在分析文件匹配情况...")
    matched_pairs = get_matching_files(dicom_dir, mask_dir)

    if not matched_pairs:
        logger.error("未找到匹配的文件对，程序终止")
        return

    logger.info(f"开始处理 {len(matched_pairs)} 对文件")
    logger.info(f"操作: 左右翻转")
    logger.info(f"病例编号偏移: +{case_num_offset}")

    # 显示命名示例
    logger.info("文件命名示例:")
    if matched_pairs:
        example_dicom, example_mask = matched_pairs[0]
        old_name = example_dicom if '.' not in example_dicom else os.path.splitext(example_dicom)[0]
        new_name = generate_new_filename(old_name, case_num_offset)
        logger.info(f"  DICOM: {example_dicom} -> {new_name}")

        old_mask_name = os.path.splitext(example_mask)[0]
        mask_ext = os.path.splitext(example_mask)[1]
        new_mask_name = f"{generate_new_filename(old_mask_name, case_num_offset)}{mask_ext}"
        logger.info(f"  Mask: {example_mask} -> {new_mask_name}")

    # ========== 处理文件 ==========

    processed_count = 0
    error_count = 0
    skipped_count = 0
    dicom_saved_count = 0
    mask_saved_count = 0

    for dicom_file, mask_file in tqdm(matched_pairs, desc="左右翻转处理"):
        try:
            dicom_path = os.path.join(dicom_dir, dicom_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # 生成新文件名
            # DICOM文件名（保持无后缀格式）
            dicom_name_without_ext = dicom_file if '.' not in dicom_file else os.path.splitext(dicom_file)[0]
            new_dicom_name = generate_new_filename(dicom_name_without_ext, case_num_offset)

            # Mask文件名（保持原始扩展名）
            mask_name_without_ext = os.path.splitext(mask_file)[0]
            mask_ext = os.path.splitext(mask_file)[1]
            new_mask_base = generate_new_filename(mask_name_without_ext, case_num_offset)
            new_mask_name = f"{new_mask_base}{mask_ext}"

            # 输出文件路径
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)
            output_mask_path = os.path.join(output_mask_dir, new_mask_name)

            # 检查输出文件是否已存在
            if os.path.exists(output_dicom_path) and os.path.exists(output_mask_path):
                logger.debug(f"输出文件已存在，跳过: {new_dicom_name}")
                skipped_count += 1
                continue

            # 读取DICOM文件
            if not os.path.exists(output_dicom_path):
                dicom_image, dicom_data = read_dicom_file(dicom_path)
                if dicom_image is None:
                    logger.error(f"跳过 {dicom_file}: 无法读取DICOM文件")
                    error_count += 1
                    continue

                logger.debug(f"处理DICOM: {dicom_file}")
                logger.debug(f"  原始形状: {dicom_image.shape}, 数据类型: {dicom_image.dtype}")
                logger.debug(f"  数值范围: [{dicom_image.min()}, {dicom_image.max()}]")

                # 左右翻转DICOM图像
                flipped_dicom = flip_image_horizontal(dicom_image)

                logger.debug(f"  翻转后形状: {flipped_dicom.shape}, 数据类型: {flipped_dicom.dtype}")

                # 验证翻转是否正确（检查图像尺寸未变）
                if flipped_dicom.shape != dicom_image.shape:
                    logger.error(f"翻转后图像尺寸改变: {dicom_image.shape} -> {flipped_dicom.shape}")
                    error_count += 1
                    continue

                # 保存翻转后的DICOM文件
                if save_flipped_dicom(flipped_dicom, dicom_path, output_dicom_path):
                    logger.debug(f"  已保存DICOM: {new_dicom_name}")
                    dicom_saved_count += 1
                else:
                    logger.error(f"  保存DICOM失败: {new_dicom_name}")
                    error_count += 1
                    continue

            # 读取并翻转mask文件
            if not os.path.exists(output_mask_path):
                mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_image is None:
                    # 尝试用PIL读取
                    try:
                        mask_image = np.array(Image.open(mask_path))
                    except:
                        logger.error(f"跳过 {dicom_file}: 无法读取mask文件 {mask_path}")
                        error_count += 1
                        continue

                logger.debug(f"处理Mask: {mask_file}")
                logger.debug(f"  原始形状: {mask_image.shape}, 数据类型: {mask_image.dtype}")
                logger.debug(f"  数值范围: [{mask_image.min()}, {mask_image.max()}]")

                # 左右翻转mask图像
                flipped_mask = flip_image_horizontal(mask_image)

                logger.debug(f"  翻转后形状: {flipped_mask.shape}")

                # 验证翻转是否正确
                if flipped_mask.shape != mask_image.shape:
                    logger.error(f"翻转后mask尺寸改变: {mask_image.shape} -> {flipped_mask.shape}")
                    error_count += 1
                    continue

                # 保存翻转后的mask文件
                if save_flipped_mask(flipped_mask, mask_path, output_mask_path):
                    logger.debug(f"  已保存Mask: {new_mask_name}")
                    mask_saved_count += 1
                else:
                    logger.error(f"  保存Mask失败: {new_mask_name}")
                    error_count += 1
                    continue

            processed_count += 1

            # 每处理20个文件输出一次进度
            if processed_count % 20 == 0:
                logger.info(f"已处理 {processed_count}/{len(matched_pairs)} 对文件")

        except Exception as e:
            logger.error(f"处理文件对 ({dicom_file}, {mask_file}) 时出错: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    # ========== 输出统计 ==========

    logger.info("=" * 60)
    logger.info("左右翻转处理完成!")
    logger.info(f"处理统计:")
    logger.info(f"  总文件对: {len(matched_pairs)}")
    logger.info(f"  成功处理: {processed_count}")
    logger.info(f"  跳过(已存在): {skipped_count}")
    logger.info(f"  错误/失败: {error_count}")
    logger.info(f"输出统计:")
    logger.info(f"  DICOM保存: {dicom_saved_count}")
    logger.info(f"  Mask保存: {mask_saved_count}")
    logger.info(f"操作参数:")
    logger.info(f"  翻转方式: 左右翻转 (水平翻转)")
    logger.info(f"  病例编号偏移: +{case_num_offset}")
    logger.info(f"输出目录:")
    logger.info(f"  DICOM: {output_dicom_dir}")
    logger.info(f"  Mask: {output_mask_dir}")

    # 验证输出文件
    if os.path.exists(output_dicom_dir):
        output_dicom_count = len([f for f in os.listdir(output_dicom_dir)
                                  if os.path.isfile(os.path.join(output_dicom_dir, f))])
        logger.info(f"  输出DICOM文件数: {output_dicom_count}")

    if os.path.exists(output_mask_dir):
        output_mask_count = len([f for f in os.listdir(output_mask_dir)
                                 if os.path.isfile(os.path.join(output_mask_dir, f))])
        logger.info(f"  输出Mask文件数: {output_mask_count}")

    # 显示命名转换示例
    logger.info("命名转换示例:")
    logger.info(f"  ANY_204_1_0 -> ANY_5204_1_0")
    logger.info(f"  ANY_204_1 -> ANY_5204_1")
    logger.info(f"  ANY_204_1_0.tif -> ANY_5204_1_0.tif")
    logger.info("=" * 60)


def test_flip_function():
    """测试翻转功能的函数"""
    logger.info("=" * 60)
    logger.info("测试翻转功能")
    logger.info("=" * 60)

    # 创建测试图像
    test_image = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]], dtype=np.uint16)

    logger.info(f"原始图像:")
    logger.info(f"{test_image}")

    # 执行翻转
    flipped_image = flip_image_horizontal(test_image)

    logger.info(f"翻转后图像:")
    logger.info(f"{flipped_image}")

    # 验证翻转结果
    expected = np.array([[4, 3, 2, 1],
                         [8, 7, 6, 5],
                         [12, 11, 10, 9]], dtype=np.uint16)

    if np.array_equal(flipped_image, expected):
        logger.info("✓ 翻转功能测试通过")
    else:
        logger.error("✗ 翻转功能测试失败")

    # 测试文件名解析
    test_filenames = [
        "ANY_204_1_0",
        "ANY_204_1",
        "ANY_204_1_0.tif",
        "ANY_5204_1_0"
    ]

    logger.info("文件名解析测试:")
    for filename in test_filenames:
        prefix, case_num, angle_num, proj_type, ext = parse_filename(filename)
        logger.info(f"  {filename} -> 前缀={prefix}, 病例={case_num}, 角度={angle_num}, 投影={proj_type}, 扩展名={ext}")

    logger.info("=" * 60)


if __name__ == "__main__":
    # 可选：运行测试
    # test_flip_function()

    # 运行主处理
    process_flip_images()