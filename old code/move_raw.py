import os
import re
import shutil
from pathlib import Path

# 配置路径
SOURCE_ROOT = Path(r"Y:\Projects\MC-DSA")
TARGET_ROOT = Path(r"F:\1E-8")
DIFF = 45  # 数字后缀差值

# 正则匹配：匹配类似 contrast_vessel_image.dat_数字 的文件（可能有.raw后缀，也可能没有）
# 提取数字部分
pattern = re.compile(r"contrast_vessel_image\.dat_(\d+)(?:\.raw)?$")

def get_number_from_filename(filename: str):
    """从文件名中提取数字后缀，如果没有匹配则返回None"""
    match = pattern.match(filename)
    if match:
        return int(match.group(1))
    return None

def process_case(case_dir: Path):
    """处理单个病例文件夹"""
    contrast_dir = case_dir / "contrast"
    if not contrast_dir.exists() or not contrast_dir.is_dir():
        print(f"  跳过：{case_dir.name} - 没有contrast文件夹")
        return False

    # 获取contrast目录下所有文件
    try:
        files = list(contrast_dir.iterdir())
    except PermissionError:
        print(f"  跳过：{case_dir.name} - 无法读取contrast文件夹")
        return False

    # 构建数字后缀 -> 文件路径的映射
    num_to_file = {}
    for f in files:
        if f.is_file():
            num = get_number_from_filename(f.name)
            if num is not None:
                num_to_file[num] = f

    if not num_to_file:
        print(f"  跳过：{case_dir.name} - contrast文件夹中没有匹配的文件")
        return False

    # 找出所有符合差值要求的数字对
    numbers = sorted(num_to_file.keys())
    max_num = max(numbers)
    pairs = []
    for num in numbers:
        target_num = num + DIFF
        if target_num <= max_num and target_num in num_to_file:
            pairs.append((num, target_num))

    if not pairs:
        print(f"  跳过：{case_dir.name} - 没有找到差值{DIFF}的文件对")
        return False

    # 创建该病例的目标文件夹
    case_target_dir = TARGET_ROOT / case_dir.name
    case_target_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for num1, num2 in pairs:
        file1 = num_to_file[num1]
        file2 = num_to_file[num2]

        # 复制文件（保留原文件名）
        dest1 = case_target_dir / file1.name
        dest2 = case_target_dir / file2.name

        try:
            shutil.copy2(file1, dest1)  # copy2 会尽量保留元数据
            shutil.copy2(file2, dest2)
            copied_count += 2
            print(f"  复制成功：{file1.name} 和 {file2.name}")
        except Exception as e:
            print(f"  复制失败：{file1.name} 或 {file2.name} - {e}")

    print(f"  病例 {case_dir.name} 完成，共复制 {copied_count} 个文件")
    return True

def main():
    if not SOURCE_ROOT.exists():
        print(f"错误：源目录不存在 {SOURCE_ROOT}")
        return

    # 创建目标根目录
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # 遍历源目录下的所有子目录（病例文件夹）
    case_dirs = [d for d in SOURCE_ROOT.iterdir() if d.is_dir() and d.name.isdigit()]
    if not case_dirs:
        print(f"在 {SOURCE_ROOT} 中没有找到数字命名的病例文件夹")
        return

    print(f"找到 {len(case_dirs)} 个病例文件夹")
    processed = 0
    for case_dir in sorted(case_dirs, key=lambda x: int(x.name)):
        print(f"\n处理病例: {case_dir.name}")
        if process_case(case_dir):
            processed += 1

    print(f"\n全部完成！共处理 {processed} 个病例，目标文件夹：{TARGET_ROOT}")

if __name__ == "__main__":
    main()