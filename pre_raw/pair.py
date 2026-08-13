import os
from pathlib import Path


def check_file_pairs():
    # 定义路径
    data_dir = Path("D:/med_data/ANY/process/NEW_DSA_mix_resized")
    mask1_dir = Path("D:/med_data/ANY/process/mask_500_resized")

    # 获取两个文件夹中的所有文件名（不含路径）
    data_files = {f.stem for f in data_dir.iterdir() if f.is_file()}
    mask1_files = {f.stem for f in mask1_dir.iterdir() if f.is_file()}

    # 找出配对的文件
    paired_files = data_files & mask1_files

    # 找出未配对的文件
    unpaired_data = data_files - mask1_files
    unpaired_mask1 = mask1_files - data_files

    # 输出结果
    print("=" * 50)
    print("文件配对检查结果")
    print("=" * 50)

    print(f"\n数据文件夹 ({data_dir}): {len(data_files)} 个文件")
    print(f"掩码文件夹 ({mask1_dir}): {len(mask1_files)} 个文件")
    print(f"成功配对: {len(paired_files)} 对文件")

    # 显示未配对的文件
    if unpaired_data:
        print(f"\n❌ 在 data 文件夹中未找到配对的文件 ({len(unpaired_data)} 个):")
        for file in sorted(unpaired_data):
            print(f"  - {file}")

    if unpaired_mask1:
        print(f"\n❌ 在 mask1 文件夹中未找到配对的文件 ({len(unpaired_mask1)} 个):")
        for file in sorted(unpaired_mask1):
            print(f"  - {file}")

    if not unpaired_data and not unpaired_mask1:
        print(f"\n✅ 完美！所有文件都已配对！")

    # 可选：显示部分配对文件作为示例
    if paired_files:
        print(f"\n📋 配对文件示例 (前5个):")
        for i, file in enumerate(sorted(paired_files)[:5]):
            print(f"  {i + 1}. {file}")
        if len(paired_files) > 5:
            print(f"  ... 还有 {len(paired_files) - 5} 个配对文件")


def check_file_pairs_with_extensions():
    """检查包含扩展名的完整文件名配对"""
    # 定义路径
    data_dir = Path("D:/med_data/ANY/process/NEW_DSA_mix_resized")
    mask1_dir = Path("D:/med_data/ANY/process/mask_500_resized")

    # 获取两个文件夹中的所有文件名（包含扩展名）
    data_files = {f.name for f in data_dir.iterdir() if f.is_file()}
    mask1_files = {f.name for f in mask1_dir.iterdir() if f.is_file()}

    # 找出配对的文件
    paired_files = data_files & mask1_files

    # 找出未配对的文件
    unpaired_data = data_files - mask1_files
    unpaired_mask1 = mask1_files - data_files

    # 输出结果
    print("=" * 50)
    print("完整文件名配对检查（包含扩展名）")
    print("=" * 50)

    print(f"\n数据文件夹: {len(data_files)} 个文件")
    print(f"掩码文件夹: {len(mask1_files)} 个文件")
    print(f"成功配对: {len(paired_files)} 对文件")

    if unpaired_data:
        print(f"\n❌ data 文件夹中未配对的文件 ({len(unpaired_data)} 个):")
        for file in sorted(unpaired_data):
            print(f"  - {file}")

    if unpaired_mask1:
        print(f"\n❌ mask1 文件夹中未配对的文件 ({len(unpaired_mask1)} 个):")
        for file in sorted(unpaired_mask1):
            print(f"  - {file}")


def get_unpaired_files_list():
    """获取未配对文件的详细列表"""
    data_dir = Path("D:/med_data/ANY/process/NEW_DSA_mix_resized")
    mask1_dir = Path("D:/med_data/ANY/process/mask_500_resized")

    data_files = {f.stem: f for f in data_dir.iterdir() if f.is_file()}
    mask1_files = {f.stem: f for f in mask1_dir.iterdir() if f.is_file()}

    unpaired_data = {stem: path for stem, path in data_files.items() if stem not in mask1_files}
    unpaired_mask1 = {stem: path for stem, path in mask1_files.items() if stem not in data_files}

    return unpaired_data, unpaired_mask1


if __name__ == "__main__":
    # 主要检查函数
    check_file_pairs()

    print("\n" + "=" * 50)
    print("详细信息")
    print("=" * 50)

    # 获取未配对文件的详细路径
    unpaired_data, unpaired_mask1 = get_unpaired_files_list()

    if unpaired_data:
        print(f"\n📁 data 文件夹中未配对文件的完整路径:")
        for stem, path in unpaired_data.items():
            print(f"  {path}")

    if unpaired_mask1:
        print(f"\n📁 mask1 文件夹中未配对文件的完整路径:")
        for stem, path in unpaired_mask1.items():
            print(f"  {path}")

    # 如果需要检查包含扩展名的配对情况，取消下面的注释
    # print("\n" + "=" * 50)
    # check_file_pairs_with_extensions()