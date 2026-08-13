import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import glob
import shutil
from tqdm import tqdm
import logging
import re

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def adjust_contrast(dicom_data, contrast_percent):
    """
    调整DICOM图像的对比度

    Parameters:
    dicom_data: DICOM图像的像素数据
    contrast_percent: 对比度调整百分比（正数增加对比度，负数减少对比度）

    Returns:
    调整后的像素数据
    """
    # 将百分比转换为系数
    factor = 1 + contrast_percent / 100.0

    # 计算均值和标准差
    mean = np.mean(dicom_data)
    std = np.std(dicom_data)

    # 调整对比度
    adjusted_data = mean + (dicom_data - mean) * factor

    # 确保数据类型正确，裁剪到原始数据范围
    original_min = np.min(dicom_data)
    original_max = np.max(dicom_data)
    adjusted_data = np.clip(adjusted_data, original_min, original_max)

    # 转换为原始数据类型
    adjusted_data = adjusted_data.astype(dicom_data.dtype)

    return adjusted_data


def get_full_filename_without_ext(file_path):
    """
    获取完整文件名（不含扩展名）
    支持无后缀DICOM文件和有扩展名的文件

    Parameters:
    file_path: 文件路径或文件名

    Returns:
    无扩展名的完整文件名
    """
    filename = os.path.basename(file_path)

    # 检查是否有扩展名
    if '.' in filename:
        # 有扩展名，移除扩展名
        return os.path.splitext(filename)[0]
    else:
        # 无扩展名，直接返回
        return filename


def generate_new_filename_with_increment(old_filename, case_number_increment):
    """
    根据旧文件名生成新文件名，病例编号增加指定数值
    支持多种格式：
    旧格式: ANY_002_0 -> ANY_60002_0 (病例编号+60000)
    新格式: ANY_002_0_0 -> ANY_60002_0_0 (病例编号+60000)

    Parameters:
    old_filename: 旧文件名（无扩展名）
    case_number_increment: 病例编号增加的数值

    Returns:
    新文件名（无扩展名）
    """
    # 按下划线分割
    parts = old_filename.split('_')

    if len(parts) < 2:
        logger.warning(f"无法解析文件名格式: {old_filename}")
        return f"{old_filename}_contrast"

    try:
        # 病例编号是第二个部分（索引1）
        prefix = parts[0]  # 如 "ANY"
        case_num = int(parts[1])  # 如 002 -> 2
        remaining = '_'.join(parts[2:])  # 剩余部分，如 "0" 或 "0_0"

        # 病例编号增加
        new_case_num = case_num + case_number_increment

        # 保持原始的数字格式（补零）
        original_case_str = parts[1]
        if original_case_str.isdigit():
            # 保持原始位数，补零
            new_case_str = str(new_case_num).zfill(len(original_case_str))
        else:
            new_case_str = str(new_case_num)

        # 生成新文件名
        if remaining:
            new_filename = f"{prefix}_{new_case_str}_{remaining}"
        else:
            new_filename = f"{prefix}_{new_case_str}"

        logger.debug(f"文件名转换: {old_filename} -> {new_filename}")
        return new_filename

    except ValueError as e:
        logger.error(f"病例编号不是数字: {parts[1]} in {old_filename}")
        return f"{old_filename}_contrast"
    except Exception as e:
        logger.error(f"生成新文件名时出错: {old_filename}, {e}")
        return f"{old_filename}_contrast"


def load_classification_table(excel_path):
    """
    加载分类表格，使用第一列的完整文件名作为键

    Parameters:
    excel_path: Excel文件路径

    Returns:
    classification_dict: 字典，键为完整文件名（无扩展名），值为动脉瘤类型
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)

        logger.info(f"Excel文件加载成功，共 {len(df)} 行数据")
        logger.info(f"列名: {list(df.columns)}")

        # 显示前几行数据用于验证
        logger.info("前5行数据:")
        for i in range(min(5, len(df))):
            logger.info(f"  第{i + 1}行: {df.iloc[i, 0]} -> 类型 {df.iloc[i, 1]}")

        # 构建分类字典：使用第一列的完整文件名作为键
        classification_dict = {}
        for i in range(len(df)):
            # 获取完整文件名（无扩展名）
            full_filename = str(df.iloc[i, 0]).strip()
            # 移除可能的扩展名
            if '.' in full_filename:
                full_filename = os.path.splitext(full_filename)[0]

            aneurysm_type = str(df.iloc[i, 1]).strip()
            classification_dict[full_filename] = aneurysm_type

        logger.info(f"成功加载 {len(classification_dict)} 个分类记录")

        # 统计各类型数量
        type_counts = {}
        for atype in classification_dict.values():
            type_counts[atype] = type_counts.get(atype, 0) + 1

        logger.info("动脉瘤类型统计:")
        for atype in sorted(type_counts.keys()):
            logger.info(f"  类型 {atype}: {type_counts[atype]} 个文件")

        return classification_dict

    except Exception as e:
        logger.error(f"加载Excel文件时出错: {str(e)}")
        raise


def get_all_dicom_files(input_dir):
    """
    获取所有DICOM文件，建立文件名到文件路径的映射

    Parameters:
    input_dir: DICOM文件目录

    Returns:
    file_mapping: 字典，键为完整文件名（无扩展名），值为文件路径
    """
    file_mapping = {}

    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return file_mapping

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            # 获取无扩展名的完整文件名
            full_name = get_full_filename_without_ext(file)
            file_mapping[full_name] = file_path

    logger.info(f"在 {input_dir} 中找到 {len(file_mapping)} 个DICOM文件")
    return file_mapping


def process_dicom_images(
        input_dir=r"D:\med_data\multi\preprocess\all_min_DSA",
        excel_path=r"D:\med_data\multi\preprocess\min_seg.xlsx",
        output_dir=r"D:\med_data\multi\preprocess\con_min_DSA",
        png_output_dir=None,
        contrast_percent=15,
        case_number_increment=10000,
        batch_size=100
):
    """
    处理DICOM图像的主函数 - 使用完整文件名匹配

    Parameters:
    input_dir: 输入DICOM图像的目录
    excel_path: 分类表格Excel文件路径
    output_dir: 输出DICOM图像的目录
    png_output_dir: 输出PNG图像的目录（可选，None则不生成PNG）
    contrast_percent: 对比度调整百分比（可正可负）
    case_number_increment: 病例号增加的数字
    batch_size: 批处理大小
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    if png_output_dir:
        os.makedirs(png_output_dir, exist_ok=True)
        logger.info(f"创建PNG输出目录: {png_output_dir}")

    logger.info(f"创建DICOM输出目录: {output_dir}")

    # 读取Excel表格
    logger.info("正在读取分类表格...")
    classification_dict = load_classification_table(excel_path)

    if not classification_dict:
        logger.error("分类表格为空，程序终止")
        return None

    # 获取所有DICOM文件
    logger.info("正在扫描DICOM文件...")
    dicom_file_mapping = get_all_dicom_files(input_dir)

    if not dicom_file_mapping:
        logger.error("未找到任何DICOM文件，程序终止")
        return None

    # 找出匹配的文件
    matched_files = set(classification_dict.keys()) & set(dicom_file_mapping.keys())
    unmatched_in_excel = set(classification_dict.keys()) - set(dicom_file_mapping.keys())
    unmatched_in_folder = set(dicom_file_mapping.keys()) - set(classification_dict.keys())

    logger.info(f"文件匹配结果:")
    logger.info(f"  分类表条目: {len(classification_dict)}")
    logger.info(f"  DICOM文件: {len(dicom_file_mapping)}")
    logger.info(f"  匹配文件: {len(matched_files)}")
    logger.info(f"  仅在分类表中: {len(unmatched_in_excel)}")
    logger.info(f"  仅在文件夹中: {len(unmatched_in_folder)}")

    if unmatched_in_excel:
        logger.warning(f"分类表中有但文件夹中不存在的文件 (前10个):")
        for name in list(unmatched_in_excel)[:10]:
            logger.warning(f"  - {name}")

    if unmatched_in_folder:
        logger.warning(f"文件夹中有但分类表中不存在的文件 (前10个):")
        for name in list(unmatched_in_folder)[:10]:
            logger.warning(f"  - {name}")

    if not matched_files:
        logger.error("没有匹配的文件，程序终止")
        return None

    # 处理匹配的文件
    new_records = []
    processed_count = 0
    error_count = 0
    skipped_count = 0

    logger.info(f"开始处理 {len(matched_files)} 个匹配文件...")
    logger.info(f"对比度调整: {contrast_percent}%")
    logger.info(f"病例编号增量: +{case_number_increment}")

    # 将匹配的文件列表排序
    matched_files_list = sorted(matched_files)

    # 分批处理
    for i in tqdm(range(0, len(matched_files_list), batch_size), desc="处理批次"):
        batch_files = matched_files_list[i:i + batch_size]

        for old_filename in batch_files:
            try:
                file_path = dicom_file_mapping[old_filename]

                # 生成新文件名（病例编号增加）
                new_filename = generate_new_filename_with_increment(
                    old_filename, case_number_increment
                )

                new_file_path = os.path.join(output_dir, new_filename)

                # 检查输出文件是否已存在
                if os.path.exists(new_file_path):
                    logger.debug(f"输出文件已存在，跳过: {new_filename}")
                    skipped_count += 1

                    # 仍记录到新表格中
                    disease_code = classification_dict[old_filename]
                    new_records.append([new_filename, disease_code])
                    continue

                # 读取DICOM文件
                dicom_data = pydicom.dcmread(file_path, force=True)

                # 获取像素数据
                pixel_array = dicom_data.pixel_array

                logger.debug(f"处理: {old_filename}")
                logger.debug(f"  原始形状: {pixel_array.shape}, 类型: {pixel_array.dtype}")
                logger.debug(f"  数值范围: [{pixel_array.min()}, {pixel_array.max()}]")

                # 调整对比度
                adjusted_pixel_array = adjust_contrast(pixel_array, contrast_percent)

                logger.debug(f"  调整后范围: [{adjusted_pixel_array.min()}, {adjusted_pixel_array.max()}]")

                # 更新DICOM数据
                dicom_data.PixelData = adjusted_pixel_array.tobytes()
                dicom_data.Rows, dicom_data.Columns = adjusted_pixel_array.shape[:2]

                # 添加处理信息到DICOM标签
                if hasattr(dicom_data, 'ImageComments'):
                    original_comment = dicom_data.ImageComments if dicom_data.ImageComments else ""
                else:
                    original_comment = ""
                dicom_data.ImageComments = f"Contrast {contrast_percent}% - {original_comment}"

                # 保存调整后的DICOM文件
                pydicom.dcmwrite(new_file_path, dicom_data)

                # 可选：生成并保存PNG文件
                if png_output_dir:
                    png_filename = f"{new_filename}.png"
                    png_path = os.path.join(png_output_dir, png_filename)

                    # 将DICOM数据归一化到0-255范围并保存为PNG
                    normalized_array = normalize_to_uint8(adjusted_pixel_array)

                    # 保存为PNG
                    if len(normalized_array.shape) == 2:
                        img = Image.fromarray(normalized_array, 'L')
                    elif len(normalized_array.shape) == 3 and normalized_array.shape[2] == 3:
                        img = Image.fromarray(normalized_array, 'RGB')
                    else:
                        img = Image.fromarray(normalized_array.squeeze())

                    img.save(png_path)

                # 记录新表格数据
                disease_code = classification_dict[old_filename]
                new_records.append([new_filename, disease_code])

                processed_count += 1

            except Exception as e:
                logger.error(f"处理文件 {old_filename} 时出错: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue

    # 创建新表格
    logger.info("正在生成新分类表格...")
    new_df = pd.DataFrame(new_records, columns=['DSA文件名', '动脉瘤类型'])

    # 保存新表格
    new_excel_path = r"D:\med_data\multi\preprocess\min_seg_contrast_15.xlsx"

    try:
        # 创建输出目录（如果不存在）
        new_excel_dir = os.path.dirname(new_excel_path)
        if new_excel_dir and not os.path.exists(new_excel_dir):
            os.makedirs(new_excel_dir)

        with pd.ExcelWriter(new_excel_path, engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name='对比度调整', index=False)

            # 调整列宽
            worksheet = writer.sheets['对比度调整']
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 15

        logger.info(f"新分类表格已保存: {new_excel_path}")

        # 同时保存CSV格式
        new_csv_path = new_excel_path.replace('.xlsx', '.csv')
        new_df.to_csv(new_csv_path, index=False)
        logger.info(f"CSV备份已保存: {new_csv_path}")

    except Exception as e:
        logger.error(f"保存新表格时出错: {e}")

    # 输出处理统计
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"处理统计:")
    logger.info(f"  匹配文件总数: {len(matched_files)}")
    logger.info(f"  成功处理: {processed_count}")
    logger.info(f"  跳过(已存在): {skipped_count}")
    logger.info(f"  错误: {error_count}")
    logger.info(f"参数:")
    logger.info(f"  对比度调整: {contrast_percent}%")
    logger.info(f"  病例编号增量: +{case_number_increment}")
    logger.info(f"输出:")
    logger.info(f"  DICOM: {output_dir}")
    if png_output_dir:
        logger.info(f"  PNG: {png_output_dir}")
    logger.info(f"  新表格: {new_excel_path}")

    # 显示命名示例
    if matched_files_list:
        example_old = matched_files_list[0]
        example_new = generate_new_filename_with_increment(example_old, case_number_increment)
        logger.info(f"命名示例:")
        logger.info(f"  {example_old} -> {example_new}")

    logger.info("=" * 60)

    return {
        'total_processed': processed_count,
        'total_skipped': skipped_count,
        'total_errors': error_count,
        'output_dir': output_dir,
        'excel_output': new_excel_path
    }


def normalize_to_uint8(array):
    """
    将数组归一化到0-255的uint8范围

    Parameters:
    array: 输入数组

    Returns:
    归一化后的uint8数组
    """
    array_min = array.min()
    array_max = array.max()

    if array_max > array_min:
        normalized = ((array - array_min) / (array_max - array_min) * 255)
    else:
        normalized = np.zeros_like(array)

    return normalized.astype(np.uint8)


def verify_files(input_dir, output_dir, excel_path):
    """
    验证输入输出文件的一致性
    """
    logger.info("=" * 60)
    logger.info("验证文件处理结果:")

    # 读取新表格
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        logger.info(f"新表格文件数: {len(df)}")

    # 统计输出文件
    if os.path.exists(output_dir):
        output_files = [f for f in os.listdir(output_dir)
                        if os.path.isfile(os.path.join(output_dir, f))]
        logger.info(f"输出DICOM文件数: {len(output_files)}")

    logger.info("=" * 60)


def main():
    """
    主函数，配置各种参数
    """

    # ========== 配置参数 ==========

    params = {
        # 输入目录 - 待处理的DSA图像
        'input_dir': r"D:\med_data\multi\preprocess\all_min_DSA",

        # Excel文件路径 - 分类表格
        'excel_path': r"D:\med_data\multi\preprocess\min_seg.xlsx",

        # 输出目录 - 对比度调整后的DICOM
        'output_dir': r"D:\med_data\multi\preprocess\con_min_DSA_15",

        # PNG输出目录（可选，设置为None则不生成PNG）
        'png_output_dir': r"D:\med_data\multi\preprocess\con_min_DSA_png_15",

        # 对比度调整百分比（正数增加对比度，负数减少对比度）
        'contrast_percent': -15,

        # 病例号增加的数字
        'case_number_increment': 20000,

        # 批处理大小
        'batch_size': 50
    }

    # 新表格保存路径
    new_excel_path = r"D:\med_data\multi\preprocess\min_seg_contrast_15.xlsx"

    # ========== 开始处理 ==========

    logger.info("=" * 60)
    logger.info("DICOM图像对比度调整程序")
    logger.info("=" * 60)
    logger.info(f"参数设置:")
    logger.info(f"  输入目录: {params['input_dir']}")
    logger.info(f"  分类表格: {params['excel_path']}")
    logger.info(f"  输出目录: {params['output_dir']}")
    if params['png_output_dir']:
        logger.info(f"  PNG输出: {params['png_output_dir']}")
    logger.info(f"  对比度调整: {params['contrast_percent']}%")
    logger.info(f"  病例编号增量: +{params['case_number_increment']}")
    logger.info(f"  批处理大小: {params['batch_size']}")
    logger.info(f"  新表格路径: {new_excel_path}")
    logger.info("=" * 60)

    try:
        # 执行处理
        results = process_dicom_images(**params)

        if results:
            # 验证结果
            verify_files(params['input_dir'], params['output_dir'], new_excel_path)

            logger.info("所有处理完成!")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()