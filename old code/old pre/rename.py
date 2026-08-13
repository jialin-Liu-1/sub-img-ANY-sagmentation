import os
import shutil
from pathlib import Path
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_dicom_filenames(input_dir, output_dir):
    """
    处理DICOM文件名，按照规则筛选和重命名

    规则：
    1. 去掉文件名最后一位数字为2的图像
    2. 对于有view1或view2的文件名，只保留view1的图像
    3. 去掉文件名中的view1字符
    """

    logger.info("=" * 60)
    logger.info("开始处理DICOM文件名...")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有DICOM文件（无后缀文件）
    dicom_files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and '.' not in file:  # 无后缀文件
            dicom_files.append(file)

    logger.info(f"找到 {len(dicom_files)} 个DICOM文件")

    if len(dicom_files) == 0:
        logger.warning("未找到DICOM文件")
        return {}

    # 文件处理统计
    kept_files = 0
    removed_last_digit_2 = 0
    kept_view1_only = 0

    # 用于配对的字典：原始文件名 -> 新文件名
    file_mapping = {}

    # 第一步：按规则筛选文件
    for filename in dicom_files:
        original_path = os.path.join(input_dir, filename)

        # 规则1：检查文件名最后一位数字是否为2
        # 分割文件名
        name_parts = filename.split('_')

        # 检查最后一部分是否为数字
        last_part = name_parts[-1]

        if last_part.isdigit() and int(last_part) == 2:
            logger.debug(f"跳过文件 {filename}：最后一位数字为2")
            removed_last_digit_2 += 1
            continue

        # 规则2：检查是否包含view1或view2
        has_view1 = any('view1' in part.lower() for part in name_parts)
        has_view2 = any('view2' in part.lower() for part in name_parts)

        # 如果有view2，跳过该文件
        if has_view2:
            logger.debug(f"跳过文件 {filename}：包含view2")
            continue

        # 如果有view1，需要去掉view1字符
        if has_view1:
            # 构建新文件名：去掉包含view1的部分
            new_parts = [part for part in name_parts if 'view1' not in part.lower()]
            new_filename = '_'.join(new_parts)

            # 确保新文件名与原始文件不同
            if new_filename != filename:
                logger.info(f"重命名: {filename} -> {new_filename}")
                kept_view1_only += 1
            else:
                new_filename = filename
        else:
            new_filename = filename

        # 添加到映射字典
        file_mapping[filename] = new_filename

        # 复制文件到输出目录
        output_path = os.path.join(output_dir, new_filename)

        # 如果目标文件已存在，添加后缀避免覆盖
        if os.path.exists(output_path):
            base_name = new_filename
            counter = 1
            while os.path.exists(output_path):
                new_filename = f"{base_name}_{counter}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1
            file_mapping[filename] = new_filename
            logger.warning(f"文件 {new_filename} 已存在，重命名为 {new_filename}")

        shutil.copy2(original_path, output_path)
        kept_files += 1

    # 输出统计信息
    logger.info("=" * 60)
    logger.info("DICOM文件处理完成!")
    logger.info(f"原始文件数: {len(dicom_files)}")
    logger.info(f"保留文件数: {kept_files}")
    logger.info(f"移除最后一位为2的文件: {removed_last_digit_2}")
    logger.info(f"保留view1并重命名的文件: {kept_view1_only}")
    logger.info(f"输出到: {output_dir}")
    logger.info("=" * 60)

    return file_mapping


def process_mask_filenames(input_dir, output_dir, dicom_mapping=None):
    """
    处理mask图像文件名，按照规则重命名以匹配DICOM文件

    规则：
    1. 对于有view1或view2的文件名，去掉view2
    2. 去掉文件名中的view1字符
    3. 对于有三个数字的文件名（如ANY_095_1_1.tif），去掉倒数第二个数字和下划线
    """

    logger.info("=" * 60)
    logger.info("开始处理mask文件名...")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有TIF文件
    tif_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            tif_files.append(file)

    logger.info(f"找到 {len(tif_files)} 个TIF文件")

    if len(tif_files) == 0:
        logger.warning("未找到TIF文件")
        return {}

    # 文件处理统计
    processed_files = 0
    removed_view2 = 0
    simplified_three_digits = 0
    view1_removed = 0

    # 用于配对的字典：原始文件名 -> 新文件名
    mask_mapping = {}

    # 获取DICOM输出目录中的所有文件名（无后缀）
    dicom_output_files = []
    if os.path.exists(output_dir.replace('mask', 'data')):  # 假设DICOM输出目录名为data
        dicom_output_dir = output_dir.replace('mask', 'data')
        if os.path.exists(dicom_output_dir):
            for file in os.listdir(dicom_output_dir):
                if os.path.isfile(os.path.join(dicom_output_dir, file)) and '.' not in file:
                    dicom_output_files.append(file)

    logger.info(f"参考DICOM文件数: {len(dicom_output_files)}")

    for filename in tif_files:
        original_path = os.path.join(input_dir, filename)

        # 去掉文件扩展名
        name_without_ext = os.path.splitext(filename)[0]

        # 分割文件名
        name_parts = name_without_ext.split('_')

        # 规则1和2：处理view1和view2
        has_view1 = any('view1' in part.lower() for part in name_parts)
        has_view2 = any('view2' in part.lower() for part in name_parts)

        new_parts = []
        if has_view2:
            # 跳过包含view2的文件
            logger.debug(f"跳过mask文件 {filename}：包含view2")
            removed_view2 += 1
            continue
        elif has_view1:
            # 去掉包含view1的部分
            new_parts = [part for part in name_parts if 'view1' not in part.lower()]
            view1_removed += 1
        else:
            new_parts = name_parts.copy()

        # 规则3：处理有三个数字的情况（如ANY_095_1_1）
        # 检查是否有4个或更多部分，且最后两个部分都是数字
        if len(new_parts) >= 4 and new_parts[-1].isdigit() and new_parts[-2].isdigit():
            # 去掉倒数第二个数字
            new_parts = new_parts[:-2] + [new_parts[-1]]
            simplified_three_digits += 1
            logger.debug(f"简化文件名: {'_'.join(name_parts)} -> {'_'.join(new_parts)}")

        # 构建新文件名
        new_filename = '_'.join(new_parts)

        # 添加.tif扩展名
        new_filename_with_ext = new_filename + '.tif'

        # 检查新文件名是否在DICOM文件列表中（去掉扩展名比较）
        if dicom_output_files and new_filename not in dicom_output_files:
            # 尝试匹配DICOM文件
            # 查找最接近的匹配
            best_match = None
            for dicom_file in dicom_output_files:
                if new_filename in dicom_file or dicom_file in new_filename:
                    best_match = dicom_file
                    break

            if best_match:
                logger.info(f"mask文件 {filename} 匹配到DICOM文件 {best_match}")
                new_filename = best_match
                new_filename_with_ext = new_filename + '.tif'
            else:
                logger.warning(f"mask文件 {filename} 未找到对应的DICOM文件，使用原名: {new_filename_with_ext}")

        # 添加到映射字典
        mask_mapping[filename] = new_filename_with_ext

        # 复制文件到输出目录
        output_path = os.path.join(output_dir, new_filename_with_ext)

        # 如果目标文件已存在，添加后缀避免覆盖
        if os.path.exists(output_path):
            base_name = os.path.splitext(new_filename_with_ext)[0]
            ext = '.tif'
            counter = 1
            while os.path.exists(output_path):
                new_temp_name = f"{base_name}_{counter}{ext}"
                output_path = os.path.join(output_dir, new_temp_name)
                counter += 1
            mask_mapping[filename] = new_temp_name
            logger.warning(f"文件 {new_filename_with_ext} 已存在，重命名为 {new_temp_name}")

        shutil.copy2(original_path, output_path)
        processed_files += 1

    # 输出统计信息
    logger.info("=" * 60)
    logger.info("mask文件处理完成!")
    logger.info(f"原始文件数: {len(tif_files)}")
    logger.info(f"处理文件数: {processed_files}")
    logger.info(f"移除view2文件: {removed_view2}")
    logger.info(f"去掉view1字符: {view1_removed}")
    logger.info(f"简化三个数字文件名: {simplified_three_digits}")
    logger.info(f"输出到: {output_dir}")

    # 检查配对情况
    if dicom_output_files:
        mask_files_no_ext = [os.path.splitext(f)[0] for f in mask_mapping.values()]
        matched_count = sum(1 for dicom_file in dicom_output_files if dicom_file in mask_files_no_ext)

        logger.info(f"成功配对文件数: {matched_count}/{len(dicom_output_files)}")

        # 列出未配对的DICOM文件
        unmatched_dicom = [f for f in dicom_output_files if f not in mask_files_no_ext]
        if unmatched_dicom:
            logger.warning(f"未找到mask配对的DICOM文件 ({len(unmatched_dicom)}个):")
            for f in unmatched_dicom[:10]:  # 最多显示10个
                logger.warning(f"  - {f}")
            if len(unmatched_dicom) > 10:
                logger.warning(f"  ... 还有 {len(unmatched_dicom) - 10} 个")

    logger.info("=" * 60)

    return mask_mapping


def verify_file_pairs(dicom_dir, mask_dir):
    """
    验证DICOM和mask文件是否配对成功
    """
    logger.info("=" * 60)
    logger.info("开始验证文件配对...")

    # 获取DICOM文件（无后缀）
    dicom_files = []
    for file in os.listdir(dicom_dir):
        file_path = os.path.join(dicom_dir, file)
        if os.path.isfile(file_path) and '.' not in file:
            dicom_files.append(file)

    # 获取mask文件（去掉.tif扩展名）
    mask_files = []
    for file in os.listdir(mask_dir):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            mask_files.append(os.path.splitext(file)[0])

    logger.info(f"DICOM文件数: {len(dicom_files)}")
    logger.info(f"mask文件数: {len(mask_files)}")

    # 找出配对的文件
    paired_files = set(dicom_files) & set(mask_files)
    dicom_only = set(dicom_files) - set(mask_files)
    mask_only = set(mask_files) - set(dicom_files)

    logger.info(f"成功配对: {len(paired_files)} 对")

    if dicom_only:
        logger.warning(f"只有DICOM没有mask的文件 ({len(dicom_only)}个):")
        for f in sorted(list(dicom_only))[:10]:
            logger.warning(f"  - {f}")
        if len(dicom_only) > 10:
            logger.warning(f"  ... 还有 {len(dicom_only) - 10} 个")

    if mask_only:
        logger.warning(f"只有mask没有DICOM的文件 ({len(mask_only)}个):")
        for f in sorted(list(mask_only))[:10]:
            logger.warning(f"  - {f}")
        if len(mask_only) > 10:
            logger.warning(f"  ... 还有 {len(mask_only) - 10} 个")

    # 输出配对示例
    if paired_files:
        logger.info("配对示例 (前5对):")
        for i, file in enumerate(sorted(list(paired_files))[:5]):
            logger.info(f"  {i + 1}. {file} <-> {file}.tif")

    logger.info("=" * 60)

    return {
        'paired': list(paired_files),
        'dicom_only': list(dicom_only),
        'mask_only': list(mask_only)
    }


def main():
    """
    主函数：处理DICOM和mask文件名配对
    """
    # 设置路径
    dicom_input_dir = r"D:\med_data\ai\0"
    mask_input_dir = r"D:\med_data\ANY\mask"
    dicom_output_dir = r"D:\med_data\ai\data"
    mask_output_dir = r"D:\med_data\ai\mask"

    logger.info("开始文件名配对处理...")

    # 1. 先处理DICOM文件
    dicom_mapping = process_dicom_filenames(dicom_input_dir, dicom_output_dir)

    # 2. 再处理mask文件
    mask_mapping = process_mask_filenames(mask_input_dir, mask_output_dir, dicom_mapping)

    # 3. 验证文件配对
    pairing_result = verify_file_pairs(dicom_output_dir, mask_output_dir)

    # 4. 生成配对报告
    logger.info("=" * 60)
    logger.info("文件配对处理完成!")
    logger.info(f"DICOM文件输出到: {dicom_output_dir}")
    logger.info(f"mask文件输出到: {mask_output_dir}")
    logger.info(f"成功配对: {len(pairing_result['paired'])} 对文件")

    # 保存配对结果到文件
    report_file = os.path.join(os.path.dirname(dicom_output_dir), "file_pairing_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("文件配对报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"DICOM输入目录: {dicom_input_dir}\n")
        f.write(f"mask输入目录: {mask_input_dir}\n")
        f.write(f"DICOM输出目录: {dicom_output_dir}\n")
        f.write(f"mask输出目录: {mask_output_dir}\n\n")

        f.write(f"成功配对文件数: {len(pairing_result['paired'])}\n")
        f.write(f"只有DICOM没有mask: {len(pairing_result['dicom_only'])}\n")
        f.write(f"只有mask没有DICOM: {len(pairing_result['mask_only'])}\n\n")

        f.write("成功配对文件列表:\n")
        f.write("-" * 30 + "\n")
        for i, file in enumerate(sorted(pairing_result['paired']), 1):
            f.write(f"{i:3d}. {file}\n")

        if pairing_result['dicom_only']:
            f.write(f"\n只有DICOM没有mask的文件:\n")
            f.write("-" * 30 + "\n")
            for file in sorted(pairing_result['dicom_only']):
                f.write(f"  {file}\n")

        if pairing_result['mask_only']:
            f.write(f"\n只有mask没有DICOM的文件:\n")
            f.write("-" * 30 + "\n")
            for file in sorted(pairing_result['mask_only']):
                f.write(f"  {file}\n")

    logger.info(f"详细报告已保存到: {report_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()