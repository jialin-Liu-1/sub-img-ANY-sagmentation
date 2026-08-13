import os
import re
import shutil
from pathlib import Path


def rename_tif_files():
    # 源文件夹路径
    source_dir = r"D:\med_data\ai\translate\train_all_trans(2)"
    # 目标文件夹路径
    target_dir = r"D:\med_data\ai\translate\contrast_mask"

    # 创建目标文件夹（如果不存在）
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # 计数器
    processed_count = 0
    error_count = 0

    # 编译正则表达式模式，用于匹配文件名
    # 模式：ANY_数字_数字.tif
    pattern = re.compile(r'^(ANY_)(\d+)_(\d+\.tif)$', re.IGNORECASE)

    print(f"开始处理文件夹: {source_dir}")
    print(f"目标文件夹: {target_dir}")
    print("-" * 50)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.tif'):
            continue

        file_path = os.path.join(source_dir, filename)

        # 确保是文件而不是文件夹
        if not os.path.isfile(file_path):
            continue

        # 使用正则表达式匹配文件名
        match = pattern.match(filename)

        if match:
            try:
                prefix = match.group(1)  # "ANY_"
                case_num = match.group(2)  # 病例号
                suffix = match.group(3)  # "图像号.tif"

                # 将病例号转为整数并加上30000
                new_case_num = int(case_num) + 30000

                # 构建新文件名
                new_filename = f"{prefix}{new_case_num:05d}_{suffix}"
                new_file_path = os.path.join(target_dir, new_filename)

                # 复制并重命名文件
                shutil.copy2(file_path, new_file_path)

                print(f"✓ {filename} -> {new_filename}")
                processed_count += 1

            except Exception as e:
                print(f"✗ 处理文件 {filename} 时出错: {str(e)}")
                error_count += 1
        else:
            print(f"? 跳过不匹配的文件: {filename}")

    print("-" * 50)
    print(f"处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"文件已保存到: {target_dir}")


if __name__ == "__main__":
    rename_tif_files()