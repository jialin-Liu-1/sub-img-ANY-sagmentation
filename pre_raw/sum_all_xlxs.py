import os
import pandas as pd
import shutil
import logging
from tqdm import tqdm
from collections import defaultdict

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_file_info(base_path, aneurysm_type):
    """
    收集指定文件夹中的DSA文件信息

    参数:
        base_path: 主文件夹路径（如 r"D:\med_data\multi\preprocess\min\1"）
        aneurysm_type: 动脉瘤类型（如 "1", "2", "4"等）

    返回:
        file_info_list: 文件信息列表，每个元素为(dsa_filename, aneurysm_type, dsa_path, mask_path)
    """
    dsa_dir = os.path.join(base_path, "all_dicom")
    mask_dir = os.path.join(base_path, "all_tif")

    if not os.path.exists(dsa_dir):
        logger.warning(f"DSA文件夹不存在: {dsa_dir}")
        return []

    if not os.path.exists(mask_dir):
        logger.warning(f"Mask文件夹不存在: {mask_dir}")
        return []

    # 获取DSA文件列表（无后缀文件）
    dsa_files = {}
    for f in os.listdir(dsa_dir):
        file_path = os.path.join(dsa_dir, f)
        if os.path.isfile(file_path):
            # 获取文件名（无后缀）
            name = f if '.' not in f else os.path.splitext(f)[0]
            dsa_files[name] = {
                'filename': f,
                'path': file_path
            }

    # 获取mask文件列表
    mask_files = {}
    for f in os.listdir(mask_dir):
        file_path = os.path.join(mask_dir, f)
        if os.path.isfile(file_path):
            # 获取文件名（无后缀）
            name = os.path.splitext(f)[0]
            mask_files[name] = {
                'filename': f,
                'path': file_path
            }

    # 找出匹配的文件对
    matched_names = set(dsa_files.keys()) & set(mask_files.keys())

    logger.info(f"文件夹 {base_path}:")
    logger.info(f"  动脉瘤类型: {aneurysm_type}")
    logger.info(f"  DSA文件总数: {len(dsa_files)}")
    logger.info(f"  Mask文件总数: {len(mask_files)}")
    logger.info(f"  匹配文件对: {len(matched_names)}")

    # 构建文件信息列表
    file_info_list = []
    for name in sorted(matched_names):
        file_info_list.append({
            'dsa_name': name,  # 无后缀的DSA文件名
            'aneurysm_type': str(aneurysm_type),
            'dsa_path': dsa_files[name]['path'],
            'dsa_filename': dsa_files[name]['filename'],  # 原始文件名（可能无后缀）
            'mask_path': mask_files[name]['path'],
            'mask_filename': mask_files[name]['filename']  # 原始文件名（可能带扩展名）
        })

    # 显示不匹配的文件
    dsa_only = set(dsa_files.keys()) - set(mask_files.keys())
    mask_only = set(mask_files.keys()) - set(dsa_files.keys())

    if dsa_only:
        logger.warning(f"  仅有DSA无mask的文件: {len(dsa_only)} 个")
        for name in sorted(dsa_only)[:5]:
            logger.warning(f"    - {name}")

    if mask_only:
        logger.warning(f"  仅有mask无DSA的文件: {len(mask_only)} 个")
        for name in sorted(mask_only)[:5]:
            logger.warning(f"    - {name}")

    return file_info_list


def generate_classification_excel(all_file_info, output_excel_path):
    """
    生成动脉瘤分类表格

    参数:
        all_file_info: 所有文件信息列表
        output_excel_path: 输出Excel文件路径
    """
    if not all_file_info:
        logger.error("没有文件信息，无法生成表格")
        return False

    # 创建DataFrame
    df_data = []
    for info in all_file_info:
        df_data.append({
            'DSA文件名': info['dsa_name'],  # 无后缀的完整文件名
            '动脉瘤类型': info['aneurysm_type']
        })

    df = pd.DataFrame(df_data)

    # 按动脉瘤类型和文件名排序
    df = df.sort_values(by=['动脉瘤类型', 'DSA文件名'])

    # 保存到Excel
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_excel_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 使用ExcelWriter以获得更多控制
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='动脉瘤分类', index=False)

            # 获取工作表以调整列宽
            worksheet = writer.sheets['动脉瘤分类']
            worksheet.column_dimensions['A'].width = 30  # DSA文件名列宽
            worksheet.column_dimensions['B'].width = 15  # 动脉瘤类型列宽

        logger.info(f"分类表格已保存: {output_excel_path}")

        # 输出统计信息
        logger.info("=" * 60)
        logger.info("分类表格统计:")
        logger.info(f"  总文件数: {len(df)}")
        logger.info(f"  动脉瘤类型分布:")
        for atype in sorted(df['动脉瘤类型'].unique()):
            count = len(df[df['动脉瘤类型'] == atype])
            logger.info(f"    类型 {atype}: {count} 个文件")
        logger.info(f"  列名: {list(df.columns)}")
        logger.info("=" * 60)

        # 显示前10行作为预览
        logger.info("表格前10行预览:")
        for i in range(min(10, len(df))):
            logger.info(f"  {df.iloc[i, 0]} -> 类型 {df.iloc[i, 1]}")

        return True

    except Exception as e:
        logger.error(f"保存Excel文件失败: {e}")
        return False


def merge_files_to_common_folders(all_file_info, output_dsa_dir, output_mask_dir):
    """
    将所有文件复制合并到统一的文件夹中

    参数:
        all_file_info: 所有文件信息列表
        output_dsa_dir: 输出DSA文件夹路径
        output_mask_dir: 输出Mask文件夹路径
    """
    # 创建输出文件夹
    for dir_path in [output_dsa_dir, output_mask_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建输出文件夹: {dir_path}")

    # 统计信息
    dsa_copied = 0
    mask_copied = 0
    dsa_skipped = 0
    mask_skipped = 0
    errors = 0

    logger.info(f"开始复制文件...")
    logger.info(f"DSA输出: {output_dsa_dir}")
    logger.info(f"Mask输出: {output_mask_dir}")

    for info in tqdm(all_file_info, desc="复制文件"):
        try:
            # 复制DSA文件
            dsa_src = info['dsa_path']
            dsa_dst = os.path.join(output_dsa_dir, info['dsa_filename'])

            if not os.path.exists(dsa_dst):
                shutil.copy2(dsa_src, dsa_dst)
                dsa_copied += 1
                logger.debug(f"复制DSA: {info['dsa_filename']}")
            else:
                dsa_skipped += 1
                logger.debug(f"DSA已存在: {info['dsa_filename']}")

            # 复制Mask文件
            mask_src = info['mask_path']
            mask_dst = os.path.join(output_mask_dir, info['mask_filename'])

            if not os.path.exists(mask_dst):
                shutil.copy2(mask_src, mask_dst)
                mask_copied += 1
                logger.debug(f"复制Mask: {info['mask_filename']}")
            else:
                mask_skipped += 1
                logger.debug(f"Mask已存在: {info['mask_filename']}")

        except Exception as e:
            logger.error(f"复制文件时出错 {info['dsa_name']}: {e}")
            errors += 1

    # 输出统计
    logger.info("=" * 60)
    logger.info("文件复制完成!")
    logger.info(f"DSA文件:")
    logger.info(f"  新复制: {dsa_copied}")
    logger.info(f"  已存在: {dsa_skipped}")
    logger.info(f"Mask文件:")
    logger.info(f"  新复制: {mask_copied}")
    logger.info(f"  已存在: {mask_skipped}")
    logger.info(f"错误: {errors}")
    logger.info("=" * 60)


def verify_output(output_excel_path, output_dsa_dir, output_mask_dir):
    """
    验证输出结果

    参数:
        output_excel_path: Excel文件路径
        output_dsa_dir: DSA输出文件夹
        output_mask_dir: Mask输出文件夹
    """
    logger.info("=" * 60)
    logger.info("验证输出结果:")

    # 验证Excel文件
    if os.path.exists(output_excel_path):
        try:
            df = pd.read_excel(output_excel_path)
            logger.info(f"Excel文件验证: ✓")
            logger.info(f"  行数: {len(df)}")
            logger.info(f"  列: {list(df.columns)}")
            logger.info(f"  动脉瘤类型: {sorted(df['动脉瘤类型'].unique())}")
        except Exception as e:
            logger.error(f"Excel文件验证失败: {e}")
    else:
        logger.error(f"Excel文件不存在: {output_excel_path}")

    # 验证DSA文件夹
    if os.path.exists(output_dsa_dir):
        dsa_files = [f for f in os.listdir(output_dsa_dir)
                     if os.path.isfile(os.path.join(output_dsa_dir, f))]
        logger.info(f"DSA文件夹验证: ✓")
        logger.info(f"  文件数量: {len(dsa_files)}")

        # 按动脉瘤类型分类统计（从Excel读取）
        if os.path.exists(output_excel_path):
            try:
                df = pd.read_excel(output_excel_path)
                for atype in sorted(df['动脉瘤类型'].unique()):
                    type_files = df[df['动脉瘤类型'] == atype]['DSA文件名'].tolist()
                    existing = sum(1 for f in type_files if f in dsa_files or f in
                                   [os.path.splitext(x)[0] for x in dsa_files])
                    logger.info(f"  类型 {atype}: {existing}/{len(type_files)} 文件存在")
            except:
                pass
    else:
        logger.error(f"DSA文件夹不存在: {output_dsa_dir}")

    # 验证Mask文件夹
    if os.path.exists(output_mask_dir):
        mask_files = [f for f in os.listdir(output_mask_dir)
                      if os.path.isfile(os.path.join(output_mask_dir, f))]
        logger.info(f"Mask文件夹验证: ✓")
        logger.info(f"  文件数量: {len(mask_files)}")
    else:
        logger.error(f"Mask文件夹不存在: {output_mask_dir}")

    logger.info("=" * 60)


def main():
    """主函数"""

    # ========== 配置参数 ==========

    # 基础路径
    base_path = r"D:\med_data\multi\preprocess\min"

    # 要处理的动脉瘤类型文件夹
    aneurysm_folders = ["1", "2", "4", "5", "6", "7"]

    # 输出路径
    output_excel_path = r"D:\med_data\multi\preprocess\min_seg.xlsx"
    output_dsa_dir = r"D:\med_data\multi\preprocess\all_min_DSA"
    output_mask_dir = r"D:\med_data\multi\preprocess\all_min_mask"

    # 其他选项
    verify_results = True  # 是否验证结果

    # ========== 开始处理 ==========

    logger.info("=" * 60)
    logger.info("DSA文件信息收集和合并程序")
    logger.info("=" * 60)
    logger.info(f"处理文件夹:")
    for folder in aneurysm_folders:
        folder_path = os.path.join(base_path, folder)
        logger.info(f"  类型 {folder}: {folder_path}")
    logger.info(f"输出:")
    logger.info(f"  分类表格: {output_excel_path}")
    logger.info(f"  合并DSA: {output_dsa_dir}")
    logger.info(f"  合并Mask: {output_mask_dir}")

    # 收集所有文件信息
    all_file_info = []
    total_matched = 0
    total_dsa_only = 0
    total_mask_only = 0

    logger.info("正在收集文件信息...")

    for folder in aneurysm_folders:
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            logger.warning(f"文件夹不存在，跳过: {folder_path}")
            continue

        # 收集该文件夹的文件信息
        file_info_list = collect_file_info(folder_path, folder)

        if file_info_list:
            all_file_info.extend(file_info_list)
            total_matched += len(file_info_list)

        logger.info(f"  类型 {folder}: 收集到 {len(file_info_list)} 对匹配文件")

    logger.info(f"总共收集到 {len(all_file_info)} 对匹配文件")

    if not all_file_info:
        logger.error("没有找到任何匹配的文件对，程序终止")
        return

    # 检查是否有重复的文件名
    dsa_names = [info['dsa_name'] for info in all_file_info]
    if len(dsa_names) != len(set(dsa_names)):
        logger.warning("发现重复的DSA文件名!")
        from collections import Counter
        duplicates = [name for name, count in Counter(dsa_names).items() if count > 1]
        logger.warning(f"重复的文件名 ({len(duplicates)} 个):")
        for dup in duplicates[:10]:
            logger.warning(f"  {dup}: 出现 {Counter(dsa_names)[dup]} 次")
        if len(duplicates) > 10:
            logger.warning(f"  ... 还有 {len(duplicates) - 10} 个重复文件名")

    # 生成分类表格
    logger.info("正在生成分类表格...")
    if generate_classification_excel(all_file_info, output_excel_path):
        logger.info("✓ 分类表格生成成功")
    else:
        logger.error("✗ 分类表格生成失败")

    # 合并文件到统一文件夹
    logger.info("正在合并文件到统一文件夹...")
    merge_files_to_common_folders(all_file_info, output_dsa_dir, output_mask_dir)

    # 验证结果
    if verify_results:
        verify_output(output_excel_path, output_dsa_dir, output_mask_dir)

    # 最终总结
    logger.info("=" * 60)
    logger.info("程序执行完成!")
    logger.info(f"处理的动脉瘤类型: {aneurysm_folders}")
    logger.info(f"收集的文件对总数: {len(all_file_info)}")
    logger.info(f"分类表格: {output_excel_path}")
    logger.info(f"DSA合并文件夹: {output_dsa_dir}")
    logger.info(f"Mask合并文件夹: {output_mask_dir}")

    # 显示各类型文件统计
    type_stats = defaultdict(int)
    for info in all_file_info:
        type_stats[info['aneurysm_type']] += 1

    logger.info("各类型文件数量:")
    for atype in sorted(type_stats.keys()):
        logger.info(f"  类型 {atype}: {type_stats[atype]} 个")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()