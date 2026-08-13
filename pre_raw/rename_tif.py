import os
import shutil
import logging
from tqdm import tqdm
import re

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_filename(filename):
    """
    解析文件名，提取各部分
    支持格式: ANY_003_1_0.tif

    返回:
        prefix: 前缀 (如 "ANY")
        case_num: 病例编号 (如 "003")
        remaining: 剩余部分 (如 "1_0")
        extension: 文件扩展名 (如 ".tif")
    """
    # 分离扩展名
    name_without_ext = os.path.splitext(filename)[0] if '.' in filename else filename
    extension = os.path.splitext(filename)[1] if '.' in filename else ''

    # 按下划线分割
    parts = name_without_ext.split('_')

    if len(parts) < 2:
        logger.warning(f"无法解析文件名格式: {filename}")
        return None, None, None, extension

    prefix = parts[0]  # "ANY"
    case_num = parts[1]  # "003"
    remaining = '_'.join(parts[2:]) if len(parts) > 2 else ''  # "1_0" 或其他

    return prefix, case_num, remaining, extension


def generate_new_filename(original_filename, case_num_increment=10000):
    """
    根据原始文件名生成新文件名，病例编号增加指定数值

    参数:
        original_filename: 原始文件名
        case_num_increment: 病例编号增加的数值

    返回:
        新文件名
    """
    prefix, case_num_str, remaining, extension = parse_filename(original_filename)

    if prefix is None or case_num_str is None:
        logger.warning(f"无法生成新文件名: {original_filename}")
        return original_filename

    try:
        # 将病例编号转换为整数并增加
        case_num_int = int(case_num_str)
        new_case_num_int = case_num_int + case_num_increment

        # 保持原始的数字格式（补零）
        new_case_num_str = str(new_case_num_int).zfill(len(case_num_str))

        # 生成新文件名
        if remaining:
            new_name = f"{prefix}_{new_case_num_str}_{remaining}{extension}"
        else:
            new_name = f"{prefix}_{new_case_num_str}{extension}"

        return new_name

    except ValueError:
        logger.error(f"病例编号不是数字: {case_num_str} in {original_filename}")
        return original_filename


def copy_and_rename_mask_files(
        input_dir=r"D:\med_data\multi\preprocess\all_min_mask",
        output_dir=r"D:\med_data\multi\preprocess\all_min_mask1",
        case_num_increment=10000
):
    """
    复制并重命名mask文件

    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        case_num_increment: 病例编号增加的数值
    """

    # 检查输入文件夹是否存在
    if not os.path.exists(input_dir):
        logger.error(f"输入文件夹不存在: {input_dir}")
        return

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出文件夹: {output_dir}")

    # 获取所有mask文件（.tif格式）
    mask_files = []
    for f in os.listdir(input_dir):
        file_path = os.path.join(input_dir, f)
        if os.path.isfile(file_path) and f.lower().endswith(('.tif', '.tiff')):
            mask_files.append(f)

    logger.info(f"找到 {len(mask_files)} 个mask文件")

    if not mask_files:
        logger.warning("未找到任何mask文件")
        return

    # 处理统计
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 显示命名示例
    if mask_files:
        example_old = mask_files[0]
        example_new = generate_new_filename(example_old, case_num_increment)
        logger.info(f"命名示例: {example_old} -> {example_new}")

    logger.info(f"开始处理 {len(mask_files)} 个mask文件...")
    logger.info(f"病例编号增量: +{case_num_increment}")

    for old_filename in tqdm(mask_files, desc="重命名mask文件"):
        try:
            # 生成新文件名
            new_filename = generate_new_filename(old_filename, case_num_increment)

            # 构建完整路径
            old_path = os.path.join(input_dir, old_filename)
            new_path = os.path.join(output_dir, new_filename)

            # 检查新文件名是否与旧文件名相同
            if old_filename == new_filename:
                logger.warning(f"文件名未改变: {old_filename}")
                skipped_count += 1
                continue

            # 检查目标文件是否已存在
            if os.path.exists(new_path):
                logger.debug(f"目标文件已存在，跳过: {new_filename}")
                skipped_count += 1
                continue

            # 复制文件到新位置并重命名
            shutil.copy2(old_path, new_path)

            logger.debug(f"重命名: {old_filename} -> {new_filename}")
            processed_count += 1

        except Exception as e:
            logger.error(f"处理文件 {old_filename} 时出错: {e}")
            error_count += 1
            continue

    # 输出统计
    logger.info("=" * 60)
    logger.info("Mask文件重命名完成!")
    logger.info(f"处理统计:")
    logger.info(f"  总文件数: {len(mask_files)}")
    logger.info(f"  成功处理: {processed_count}")
    logger.info(f"  跳过(已存在/未改变): {skipped_count}")
    logger.info(f"  错误: {error_count}")
    logger.info(f"参数:")
    logger.info(f"  病例编号增量: +{case_num_increment}")
    logger.info(f"输出:")
    logger.info(f"  输出文件夹: {output_dir}")

    # 验证输出
    if os.path.exists(output_dir):
        output_files = [f for f in os.listdir(output_dir)
                        if os.path.isfile(os.path.join(output_dir, f))]
        logger.info(f"  输出文件数: {len(output_files)}")

    logger.info("=" * 60)


def verify_rename_results(input_dir, output_dir, case_num_increment=10000):
    """
    验证重命名结果的正确性
    """
    logger.info("=" * 60)
    logger.info("验证重命名结果:")

    if not os.path.exists(input_dir):
        logger.error(f"输入文件夹不存在: {input_dir}")
        return

    if not os.path.exists(output_dir):
        logger.error(f"输出文件夹不存在: {output_dir}")
        return

    # 获取输入文件列表
    input_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.tif', '.tiff'))]

    # 获取输出文件列表
    output_files = [f for f in os.listdir(output_dir)
                    if os.path.isfile(os.path.join(output_dir, f))]

    logger.info(f"输入文件数: {len(input_files)}")
    logger.info(f"输出文件数: {len(output_files)}")

    # 验证每个输入文件是否有对应的输出文件
    matched_count = 0
    unmatched_files = []

    for input_file in input_files:
        expected_output = generate_new_filename(input_file, case_num_increment)
        if expected_output in output_files:
            matched_count += 1
        else:
            unmatched_files.append((input_file, expected_output))

    logger.info(f"匹配文件数: {matched_count}/{len(input_files)}")

    if unmatched_files:
        logger.warning(f"未找到对应输出文件的文件 ({len(unmatched_files)}个):")
        for input_file, expected_output in unmatched_files[:10]:
            logger.warning(f"  {input_file} -> 期望: {expected_output}")
        if len(unmatched_files) > 10:
            logger.warning(f"  ... 还有 {len(unmatched_files) - 10} 个")

    # 显示几个重命名示例
    if input_files:
        logger.info("重命名示例（前5个）:")
        for i, input_file in enumerate(input_files[:5]):
            output_file = generate_new_filename(input_file, case_num_increment)
            exists = "✓" if output_file in output_files else "✗"
            logger.info(f"  {exists} {input_file} -> {output_file}")

    logger.info("=" * 60)


def show_filename_examples(input_dir, case_num_increment=10000):
    """
    显示文件名转换示例
    """
    logger.info("=" * 60)
    logger.info("文件名转换示例:")

    # 各种可能的文件名格式
    examples = [
        "ANY_003_1_0.tif",
        "ANY_003_1_0.tiff",
        "ANY_003_1.tif",
        "ANY_003.tif",
        "ANY_10003_1_0.tif",
        "ANY_10003_1_0"
    ]

    for example in examples:
        new_name = generate_new_filename(example, case_num_increment)
        logger.info(f"  {example} -> {new_name}")

    # 如果有实际文件，也显示一些实际文件的转换
    if os.path.exists(input_dir):
        actual_files = [f for f in os.listdir(input_dir)
                        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.tif', '.tiff'))]
        if actual_files:
            logger.info("实际文件转换示例（前5个）:")
            for actual_file in actual_files[:5]:
                new_name = generate_new_filename(actual_file, case_num_increment)
                logger.info(f"  {actual_file} -> {new_name}")

    logger.info("=" * 60)


def main():
    """主函数"""

    # ========== 配置参数 ==========

    # 输入文件夹
    input_dir = r"D:\med_data\multi\preprocess\all_min_mask"

    # 输出文件夹
    output_dir = r"D:\med_data\multi\preprocess\all_min_mask2"

    # 病例编号增量
    case_num_increment = 20000

    # 其他选项
    show_examples = True  # 是否显示命名示例
    verify_results = True  # 是否验证结果

    # ========== 开始处理 ==========

    logger.info("=" * 60)
    logger.info("Mask文件重命名程序")
    logger.info("=" * 60)
    logger.info(f"输入文件夹: {input_dir}")
    logger.info(f"输出文件夹: {output_dir}")
    logger.info(f"病例编号增量: +{case_num_increment}")
    logger.info(f"文件格式: .tif / .tiff")
    logger.info(f"命名规则: 病例编号 + {case_num_increment}")
    logger.info(f"示例: ANY_003_1_0.tif -> ANY_10003_1_0.tif")

    # 显示命名示例
    if show_examples:
        show_filename_examples(input_dir, case_num_increment)

    try:
        # 执行重命名
        copy_and_rename_mask_files(
            input_dir=input_dir,
            output_dir=output_dir,
            case_num_increment=case_num_increment
        )

        # 验证结果
        if verify_results:
            verify_rename_results(input_dir, output_dir, case_num_increment)

        logger.info("所有处理完成!")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()