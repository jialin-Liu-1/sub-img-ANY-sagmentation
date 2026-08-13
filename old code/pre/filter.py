import pandas as pd
import os
import shutil
from pathlib import Path


def process_medical_data():
    """
    处理医学图像数据和分类表格
    """

    # 定义文件路径
    base_path = r"D:\med_data\ai"
    classify_file = os.path.join(base_path, "classify.xlsx")
    train1_path = os.path.join(base_path, "train1")  # DICOM文件（无后缀）
    train2_path = os.path.join(base_path, "train2")  # TIF文件

    # 输出路径
    train11_path = os.path.join(base_path, "train11")  # 新DICOM文件夹
    train22_path = os.path.join(base_path, "train22")  # 新TIF文件夹
    classify1_file = os.path.join(base_path, "classify1.xlsx")  # 新分类表格
    classify2_file = os.path.join(base_path, "classify2.xlsx")  # 统计表格

    # 创建输出文件夹
    os.makedirs(train11_path, exist_ok=True)
    os.makedirs(train22_path, exist_ok=True)

    print("=" * 60)
    print("开始处理医学图像数据...")
    print("=" * 60)

    # 1. 读取分类表格 - 只读取前两列
    print("\n1. 读取分类表格...")
    try:
        # 使用 usecols 参数只读取前两列
        df = pd.read_excel(classify_file, header=None, usecols=[0, 1])
        # 设置列名
        df.columns = ['病历号', '分类']
        print(f"   原始数据：共 {len(df)} 条记录")
        print(f"   分类分布：\n{df['分类'].value_counts().sort_index()}")

        # 显示前几行确认
        print(f"   前5行数据：")
        print(df.head())

    except Exception as e:
        print(f"   读取表格失败: {e}")
        return

    # 2. 筛选有效病例（分类为1-7）
    print("\n2. 筛选有效病例（分类1-7）...")
    valid_df = df[df['分类'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
    valid_df = valid_df.reset_index(drop=True)
    print(f"   有效病例：共 {len(valid_df)} 条记录")
    print(f"   有效病例分类分布：\n{valid_df['分类'].value_counts().sort_index()}")

    # 3. 统计无效病例
    invalid_count = len(df) - len(valid_df)
    print(f"   无效病例（分类0）：{invalid_count} 条")

    # 4. 保存新的分类表格（仅有效病例，只保存两列）
    print("\n3. 保存新的分类表格...")
    try:
        valid_df.to_excel(classify1_file, index=False, header=False)
        print(f"   已保存至: {classify1_file}")

        # 验证保存的文件
        test_df = pd.read_excel(classify1_file, header=None)
        print(f"   验证 - 新表格行数: {len(test_df)}，列数: {len(test_df.columns)}")

    except Exception as e:
        print(f"   保存失败: {e}")

    # 5. 生成统计表格
    print("\n4. 生成统计表格...")
    stats = []
    total_valid = len(valid_df)

    for category in range(1, 8):
        count = len(valid_df[valid_df['分类'] == category])
        if count > 0:
            percentage = (count / total_valid) * 100
            stats.append({
                '分类': category,
                '数量': count,
                '占比(%)': round(percentage, 2)
            })

    stats_df = pd.DataFrame(stats)
    print(f"   统计结果：")
    print(stats_df.to_string(index=False))

    try:
        stats_df.to_excel(classify2_file, index=False)
        print(f"   已保存至: {classify2_file}")
    except Exception as e:
        print(f"   保存失败: {e}")

    # 6. 复制图像文件
    print("\n5. 复制图像文件...")

    # 获取train1和train2文件夹中的所有文件
    train1_files = os.listdir(train1_path)
    train2_files = os.listdir(train2_path)

    print(f"   train1文件夹中文件数: {len(train1_files)}")
    print(f"   train2文件夹中文件数: {len(train2_files)}")

    copied_dicom = 0
    copied_tif = 0
    missing_files = []
    missing_categories = {}  # 记录缺失文件的分类

    for _, row in valid_df.iterrows():
        patient_id = row['病历号']
        category = row['分类']

        # 每个病历号对应两个文件：_0 和 _1
        for suffix in ['_0', '_1']:
            # 处理DICOM文件（无后缀）
            dicom_filename = f"{patient_id}{suffix}"
            dicom_src = os.path.join(train1_path, dicom_filename)
            dicom_dst = os.path.join(train11_path, dicom_filename)

            # 处理TIF文件
            tif_filename = f"{patient_id}{suffix}.tif"
            tif_src = os.path.join(train2_path, tif_filename)
            tif_dst = os.path.join(train22_path, tif_filename)

            # 复制DICOM文件
            if os.path.exists(dicom_src):
                try:
                    shutil.copy2(dicom_src, dicom_dst)
                    copied_dicom += 1
                except Exception as e:
                    print(f"   复制失败 {dicom_filename}: {e}")
                    missing_files.append(dicom_filename)
                    missing_categories[dicom_filename] = category
            else:
                missing_files.append(dicom_filename)
                missing_categories[dicom_filename] = category

            # 复制TIF文件
            if os.path.exists(tif_src):
                try:
                    shutil.copy2(tif_src, tif_dst)
                    copied_tif += 1
                except Exception as e:
                    print(f"   复制失败 {tif_filename}: {e}")
                    missing_files.append(tif_filename)
                    missing_categories[tif_filename] = category
            else:
                missing_files.append(tif_filename)
                missing_categories[tif_filename] = category

    # 7. 输出复制结果
    print("\n6. 复制完成统计：")
    expected_files = len(valid_df) * 4
    actual_files = copied_dicom + copied_tif
    print(f"   复制的DICOM文件数: {copied_dicom}")
    print(f"   复制的TIF文件数: {copied_tif}")
    print(f"   期望复制文件总数: {expected_files}")
    print(f"   实际复制文件总数: {actual_files}")

    if missing_files:
        print(f"\n   缺失文件数: {len(missing_files)}")

        # 按分类统计缺失文件
        missing_by_category = {}
        for filename, category in missing_categories.items():
            if category not in missing_by_category:
                missing_by_category[category] = []
            missing_by_category[category].append(filename)

        print(f"   按分类缺失情况：")
        for category in sorted(missing_by_category.keys()):
            print(f"     分类 {category}: {len(missing_by_category[category])} 个文件缺失")

        # 保存缺失文件列表到文本文件
        missing_file_path = os.path.join(base_path, "missing_files.txt")
        with open(missing_file_path, 'w', encoding='utf-8') as f:
            f.write("缺失文件列表：\n")
            for filename in missing_files:
                f.write(f"{filename}\n")
        print(f"   缺失文件列表已保存至: {missing_file_path}")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)

    # 返回处理结果供进一步分析
    return {
        '原始记录数': len(df),
        '有效记录数': len(valid_df),
        '无效记录数': invalid_count,
        '复制的DICOM文件': copied_dicom,
        '复制的TIF文件': copied_tif,
        '缺失文件数': len(missing_files),
        '统计表格': stats_df
    }


def check_file_structure():
    """
    检查文件结构，帮助确认文件是否存在
    """
    base_path = r"D:\med_data\ai"

    print("检查文件夹结构...")

    # 检查关键路径是否存在
    paths_to_check = [
        base_path,
        os.path.join(base_path, "train1"),
        os.path.join(base_path, "train2")
    ]

    for path in paths_to_check:
        if os.path.exists(path):
            print(f"  ✓ {path} 存在")
            if path != base_path:
                files = os.listdir(path)[:5]  # 只显示前5个文件
                if files:
                    print(f"    示例文件: {files}")
        else:
            print(f"  ✗ {path} 不存在")

    # 检查分类表格
    classify_file = os.path.join(base_path, "classify.xlsx")
    if os.path.exists(classify_file):
        print(f"  ✓ 分类表格存在")
        # 尝试读取，只读前两列
        try:
            df = pd.read_excel(classify_file, header=None, usecols=[0, 1])
            print(f"    表格行数: {len(df)}")
            print(f"    前几行:\n{df.head()}")
        except Exception as e:
            print(f"    无法读取表格: {e}")
    else:
        print(f"  ✗ 分类表格不存在")


if __name__ == "__main__":
    # 首先检查文件结构
    check_file_structure()

    print("\n" + "=" * 60)
    response = input("是否继续处理？(y/n): ")
    if response.lower() == 'y':
        result = process_medical_data()

        # 显示关键结果
        if result:
            print("\n处理结果汇总：")
            for key, value in result.items():
                if key != '统计表格':
                    print(f"  {key}: {value}")
    else:
        print("处理已取消")