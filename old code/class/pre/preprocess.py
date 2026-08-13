import os
import shutil
from pathlib import Path


def process_files():
    # 定义路径
    source_dir_2_1 = Path("D:/med_data/ai/0")
    source_dir_mask = Path("D:/med_data/ANY/mask")
    target_dir_data = Path("D:/med_data/ai/data")
    target_dir_mask1 = Path("D:/med_data/ai/mask")

    # 创建目标文件夹（如果不存在）
    target_dir_data.mkdir(parents=True, exist_ok=True)
    target_dir_mask1.mkdir(parents=True, exist_ok=True)

    # 第一步：处理 2_1 文件夹中的文件
    print("正在处理 2_1 文件夹中的文件...")
    for file_path in source_dir_2_1.iterdir():
        if file_path.is_file():
            filename = file_path.stem  # 获取文件名（不含扩展名）

            # 检查文件名最后一位是否为0或1
            if filename and filename[-1] in ('0', '1'):
                # 移动文件到 data 文件夹
                target_path = target_dir_data / file_path.name
                shutil.move(str(file_path), str(target_path))
                print(f"已移动: {file_path.name} -> {target_path}")

    # 第二步：处理 mask 文件夹中的文件
    print("\n正在处理 mask 文件夹中的文件...")
    for file_path in source_dir_mask.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.tif':
            filename = file_path.stem  # 获取文件名（不含扩展名）
            parts = filename.split('_')

            # 判断文件名格式并相应处理
            if len(parts) >= 3:
                if len(parts) == 4:  # 格式为 ANY_001_1_1 的情况
                    # 去掉倒数第二个数字，重新组合文件名
                    new_filename = f"{parts[0]}_{parts[1]}_{parts[3]}{file_path.suffix}"
                    new_file_path = target_dir_mask1 / new_filename

                    # 复制并重命名文件
                    shutil.copy2(str(file_path), str(new_file_path))
                    print(f"已重命名并复制: {file_path.name} -> {new_filename}")

                elif len(parts) == 3:  # 格式为 ANY_002_1 的情况
                    # 直接复制到目标文件夹
                    target_path = target_dir_mask1 / file_path.name
                    shutil.copy2(str(file_path), str(target_path))
                    print(f"已复制: {file_path.name} -> {target_path}")


if __name__ == "__main__":
    process_files()
    print("\n所有文件处理完成！")