import os
import shutil
import pandas as pd
import logging
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_classification_table(excel_path):
    """
    加载分类表格，提取文件名前缀和动脉瘤类型

    参数:
        excel_path: Excel文件路径

    返回:
        classification_dict: 字典，键为文件名前缀（如"ANY_439"），值为动脉瘤类型（如"1"）
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

        # 构建分类字典
        classification_dict = {}
        for i in range(len(df)):
            dsa_name = str(df.iloc[i, 0]).strip()  # 第一列：DSA文件名前缀
            aneurysm_type = str(int(df.iloc[i, 1])).strip()  # 第二列：动脉瘤类型，转为整数再转字符串

            classification_dict[dsa_name] = aneurysm_type

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


def extract_name_prefix(filename):
    """
    从完整文件名中提取前缀（前两位+病例编号）
    例如: "ANY_439_0" -> "ANY_439"
          "ANY_001_0_0" -> "ANY_001"
          "ANY_123_0.tif" -> "ANY_123"

    参数:
        filename: 完整文件名

    返回:
        prefix: 文件名前缀
    """
    # 移除扩展名
    name_without_ext = os.path.splitext(filename)[0] if '.' in filename else filename

    # 按下划线分割，取前两部分
    parts = name_without_ext.split('_')

    if len(parts) >= 2:
        # 返回前两部分作为前缀，如 "ANY_439"
        prefix = f"{parts[0]}_{parts[1]}"
        return prefix
    else:
        # 如果分割后少于2部分，返回原始名称
        logger.warning(f"无法提取文件名前缀: {filename}")
        return name_without_ext


def build_file_mapping(folder_path, file_extension=None):
    """
    构建文件夹中文件的映射，按文件名前缀分组

    参数:
        folder_path: 文件夹路径
        file_extension: 文件扩展名过滤（如".tif"），None表示不过滤

    返回:
        file_mapping: 字典，键为文件名前缀，值为文件列表
    """
    file_mapping = {}

    if not os.path.exists(folder_path):
        logger.warning(f"文件夹不存在: {folder_path}")
        return file_mapping

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            # 检查扩展名过滤
            if file_extension and not file.endswith(file_extension):
                continue

            # 提取文件名前缀
            prefix = extract_name_prefix(file)

            if prefix not in file_mapping:
                file_mapping[prefix] = []

            file_mapping[prefix].append((file, file_path))

    return file_mapping


def classify_and_copy_files(classification_dict, dsa_folder, mask_folder, output_base_folder):
    """
    根据分类信息复制DSA和mask文件到对应类型的文件夹

    参数:
        classification_dict: 分类字典，键为文件名前缀，值为动脉瘤类型
        dsa_folder: DSA文件源文件夹
        mask_folder: Mask文件源文件夹
        output_base_folder: 输出基础文件夹
    """

    # 构建DSA和mask文件的映射
    logger.info("正在分析DSA文件...")
    dsa_mapping = build_file_mapping(dsa_folder)
    logger.info(f"找到 {sum(len(files) for files in dsa_mapping.values())} 个DSA文件，"
                f"涉及 {len(dsa_mapping)} 个不同前缀")

    logger.info("正在分析Mask文件...")
    mask_mapping = build_file_mapping(mask_folder, file_extension='.tif')
    logger.info(f"找到 {sum(len(files) for files in mask_mapping.values())} 个Mask文件，"
                f"涉及 {len(mask_mapping)} 个不同前缀")

    # 统计信息
    copy_stats = {
        'total_matched': 0,
        'dsa_copied': 0,
        'mask_copied': 0,
        'dsa_missing': 0,
        'mask_missing': 0,
        'type_stats': {}
    }

    # 按动脉瘤类型分组处理
    logger.info("开始按动脉瘤类型分类复制文件...")

    for prefix, aneurysm_type in tqdm(classification_dict.items(), desc="分类处理"):
        # 创建输出文件夹
        type_dicom_folder = os.path.join(output_base_folder, str(aneurysm_type), "dicom")
        type_mask_folder = os.path.join(output_base_folder, str(aneurysm_type), "mask")

        for folder in [type_dicom_folder, type_mask_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # 初始化类型统计
        if aneurysm_type not in copy_stats['type_stats']:
            copy_stats['type_stats'][aneurysm_type] = {
                'count': 0,
                'dsa_copied': 0,
                'mask_copied': 0,
                'dsa_missing': 0,
                'mask_missing': 0
            }

        copy_stats['type_stats'][aneurysm_type]['count'] += 1

        # 复制DSA文件
        if prefix in dsa_mapping:
            dsa_files = dsa_mapping[prefix]
            for dsa_filename, dsa_path in dsa_files:
                dest_path = os.path.join(type_dicom_folder, dsa_filename)

                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(dsa_path, dest_path)
                        copy_stats['dsa_copied'] += 1
                        copy_stats['type_stats'][aneurysm_type]['dsa_copied'] += 1
                        logger.debug(f"复制DSA: {dsa_filename} -> 类型{aneurysm_type}")
                    except Exception as e:
                        logger.error(f"复制DSA文件失败 {dsa_filename}: {str(e)}")
                else:
                    logger.debug(f"DSA文件已存在: {dsa_filename}")
                    copy_stats['dsa_copied'] += 1  # 已存在也算成功
        else:
            logger.warning(f"未找到前缀 {prefix} 的DSA文件")
            copy_stats['dsa_missing'] += 1
            copy_stats['type_stats'][aneurysm_type]['dsa_missing'] += 1

        # 复制Mask文件
        if prefix in mask_mapping:
            mask_files = mask_mapping[prefix]
            for mask_filename, mask_path in mask_files:
                dest_path = os.path.join(type_mask_folder, mask_filename)

                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(mask_path, dest_path)
                        copy_stats['mask_copied'] += 1
                        copy_stats['type_stats'][aneurysm_type]['mask_copied'] += 1
                        logger.debug(f"复制Mask: {mask_filename} -> 类型{aneurysm_type}")
                    except Exception as e:
                        logger.error(f"复制Mask文件失败 {mask_filename}: {str(e)}")
                else:
                    logger.debug(f"Mask文件已存在: {mask_filename}")
                    copy_stats['mask_copied'] += 1  # 已存在也算成功
        else:
            logger.warning(f"未找到前缀 {prefix} 的Mask文件")
            copy_stats['mask_missing'] += 1
            copy_stats['type_stats'][aneurysm_type]['mask_missing'] += 1

        copy_stats['total_matched'] += 1

    # 输出统计结果
    logger.info("=" * 60)
    logger.info("文件分类复制完成!")
    logger.info(f"总处理条目: {copy_stats['total_matched']}")
    logger.info(f"DSA文件复制: {copy_stats['dsa_copied']} 个")
    logger.info(f"Mask文件复制: {copy_stats['mask_copied']} 个")
    logger.info(f"DSA文件缺失: {copy_stats['dsa_missing']} 个")
    logger.info(f"Mask文件缺失: {copy_stats['mask_missing']} 个")

    logger.info("各类型详细统计:")
    for atype in sorted(copy_stats['type_stats'].keys()):
        stats = copy_stats['type_stats'][atype]
        logger.info(f"  类型 {atype}:")
        logger.info(f"    总条目: {stats['count']}")
        logger.info(f"    DSA复制: {stats['dsa_copied']}")
        logger.info(f"    Mask复制: {stats['mask_copied']}")
        if stats['dsa_missing'] > 0:
            logger.info(f"    DSA缺失: {stats['dsa_missing']}")
        if stats['mask_missing'] > 0:
            logger.info(f"    Mask缺失: {stats['mask_missing']}")

    logger.info("=" * 60)


def verify_file_matching(classification_dict, dsa_folder, mask_folder):
    """
    验证分类表和实际文件的匹配情况
    """
    logger.info("=" * 60)
    logger.info("验证文件匹配情况")
    logger.info("=" * 60)

    dsa_mapping = build_file_mapping(dsa_folder)
    mask_mapping = build_file_mapping(mask_folder, file_extension='.tif')

    matched_both = 0
    matched_dsa_only = 0
    matched_mask_only = 0
    matched_none = 0

    missing_prefixes = []

    for prefix in classification_dict.keys():
        has_dsa = prefix in dsa_mapping
        has_mask = prefix in mask_mapping

        if has_dsa and has_mask:
            matched_both += 1
        elif has_dsa and not has_mask:
            matched_dsa_only += 1
            missing_prefixes.append((prefix, "Mask"))
        elif not has_dsa and has_mask:
            matched_mask_only += 1
            missing_prefixes.append((prefix, "DSA"))
        else:
            matched_none += 1
            missing_prefixes.append((prefix, "Both"))

    logger.info(f"分类表总条目: {len(classification_dict)}")
    logger.info(f"完全匹配(DSA+Mask): {matched_both}")
    logger.info(f"仅DSA匹配: {matched_dsa_only}")
    logger.info(f"仅Mask匹配: {matched_mask_only}")
    logger.info(f"无匹配: {matched_none}")

    if missing_prefixes:
        logger.info("缺失文件的条目（前20个）:")
        for prefix, missing_type in missing_prefixes[:20]:
            logger.info(f"  {prefix}: 缺失{missing_type}")
        if len(missing_prefixes) > 20:
            logger.info(f"  ... 还有 {len(missing_prefixes) - 20} 个条目缺失文件")


def preview_folder_structure(output_base_folder):
    """
    预览输出文件夹结构
    """
    logger.info("=" * 60)
    logger.info("输出文件夹结构预览:")
    logger.info("=" * 60)

    if os.path.exists(output_base_folder):
        for item in os.listdir(output_base_folder):
            item_path = os.path.join(output_base_folder, item)
            if os.path.isdir(item_path):
                # 统计子文件夹中的文件
                dicom_folder = os.path.join(item_path, "dicom")
                mask_folder = os.path.join(item_path, "mask")

                dicom_count = len(os.listdir(dicom_folder)) if os.path.exists(dicom_folder) else 0
                mask_count = len(os.listdir(mask_folder)) if os.path.exists(mask_folder) else 0

                logger.info(f"类型 {item}:")
                logger.info(f"  DICOM文件: {dicom_count} 个")
                logger.info(f"  Mask文件: {mask_count} 个")


def check_excel_duplicates(excel_path):
    """
    检查Excel文件中是否有重复的文件名前缀
    """
    try:
        df = pd.read_excel(excel_path)

        # 提取第一列数据
        first_column = df.iloc[:, 0].astype(str).str.strip()

        # 检查重复
        duplicates = first_column[first_column.duplicated()]

        if len(duplicates) > 0:
            logger.warning(f"发现 {len(duplicates)} 个重复的文件名前缀:")
            for dup in duplicates[:10]:
                # 找到重复项的所有行
                dup_rows = df[first_column == dup]
                logger.warning(f"  {dup}: 出现在 {len(dup_rows)} 行中，类型分别为 {list(dup_rows.iloc[:, 1])}")

            if len(duplicates) > 10:
                logger.warning(f"  ... 还有 {len(duplicates) - 10} 个重复项")
        else:
            logger.info("未发现重复的文件名前缀")

        return len(duplicates) == 0

    except Exception as e:
        logger.error(f"检查重复时出错: {str(e)}")
        return False


if __name__ == "__main__":
    # ========== 配置参数 ==========

    # Excel分类表路径
    excel_path = r"D:\med_data\ai\classify_500.xlsx"

    # 源文件夹路径
    dsa_source_folder = r"D:\med_data\multi\preprocess\MEAN\DICOM_back"  # DSA文件源文件夹
    mask_source_folder = r"D:\med_data\multi\preprocess\MEAN\mask_resized"  # Mask文件源文件夹

    # 输出基础文件夹
    output_base_folder = r"D:\med_data\multi\preprocess\MEAN"  # 输出基础文件夹

    # 选项
    verify_files = True  # 是否验证文件匹配
    check_duplicates = True  # 是否检查Excel重复项
    preview_structure = True  # 是否预览输出结构

    # ========== 开始处理 ==========

    logger.info("=" * 60)
    logger.info("DSA文件分类程序（基于Excel表格）")
    logger.info("=" * 60)
    logger.info(f"Excel文件: {excel_path}")
    logger.info(f"DSA源文件夹: {dsa_source_folder}")
    logger.info(f"Mask源文件夹: {mask_source_folder}")
    logger.info(f"输出基础文件夹: {output_base_folder}")
    logger.info("处理规则:")
    logger.info("  - 根据Excel表格中的文件名前缀进行匹配")
    logger.info("  - 按动脉瘤类型分类到不同文件夹")
    logger.info("  - 仅复制文件，不修改文件内容或文件名")

    try:
        # 检查Excel重复项
        if check_duplicates:
            check_excel_duplicates(excel_path)

        # 加载分类表
        classification_dict = load_classification_table(excel_path)

        if not classification_dict:
            logger.error("分类表为空，程序终止")
            exit()

        # 验证文件匹配情况
        if verify_files:
            verify_file_matching(classification_dict, dsa_source_folder, mask_source_folder)

        # 执行分类复制
        classify_and_copy_files(
            classification_dict=classification_dict,
            dsa_folder=dsa_source_folder,
            mask_folder=mask_source_folder,
            output_base_folder=output_base_folder
        )

        # 预览输出结构
        if preview_structure:
            preview_folder_structure(output_base_folder)

        logger.info("所有处理完成！")

        # 输出最终文件结构示例
        logger.info("=" * 60)
        logger.info("输出文件结构示例:")
        logger.info(f"{output_base_folder}")
        logger.info(f"├── 1")
        logger.info(f"│   ├── dicom    (类型1的DSA文件)")
        logger.info(f"│   └── mask     (类型1的Mask文件)")
        logger.info(f"├── 2")
        logger.info(f"│   ├── dicom    (类型2的DSA文件)")
        logger.info(f"│   └── mask     (类型2的Mask文件)")
        logger.info(f"└── ...")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()