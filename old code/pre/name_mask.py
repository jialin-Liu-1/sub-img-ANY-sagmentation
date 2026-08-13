import os
import shutil
import re


def batch_copy_and_rename():
    # 源目录和目标目录
    source_dir = r"D:\med_data\ai\translate\train_all_trans(2)"
    target_dir = r"D:\med_data\ai\translate\contrast20mask"

    # 如果目标目录不存在则创建
    os.makedirs(target_dir, exist_ok=True)

    # 匹配文件名模式：ANY_数字_数字.tif
    pattern = re.compile(r'^(ANY)_(\d+)_(\d+)\.tif$', re.IGNORECASE)

    processed_count = 0
    error_count = 0

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.tif'):
            continue

        match = pattern.match(filename)
        if not match:
            print(f"跳过不符合格式的文件: {filename}")
            error_count += 1
            continue

        prefix = match.group(1)  # ANY
        case_num = int(match.group(2))  # 病例号
        img_num = match.group(3)  # 图像号

        # 病例号加6000
        new_case_num = case_num + 60000

        # 构造新文件名
        new_filename = f"{prefix}_{new_case_num}_{img_num}.tif"

        # 源文件路径和目标文件路径
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, new_filename)

        # 检查目标文件是否已存在
        if os.path.exists(dst_path):
            print(f"⚠ 跳过（目标已存在）: {new_filename}")
            error_count += 1
            continue

        try:
            # 复制并重命名文件（使用copy2保留元数据）
            shutil.copy2(src_path, dst_path)
            print(f"✓ 已复制并重命名: {filename} -> {new_filename}")
            processed_count += 1
        except Exception as e:
            print(f"✗ 处理文件 {filename} 时出错: {e}")
            error_count += 1

    print(f"\n处理完成！")
    print(f"成功复制: {processed_count} 个文件")
    print(f"跳过/出错: {error_count} 个文件")
    print(f"原始文件保留在: {source_dir}")


if __name__ == "__main__":
    batch_copy_and_rename()