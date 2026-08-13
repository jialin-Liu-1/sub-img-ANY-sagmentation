import pandas as pd
import os
import shutil
from pathlib import Path


def extract_class5_images():
    """
    提取分类为5的病历号对应的图像文件
    文件命名：病历号 + _0 或 _1
    """

    # 定义路径
    base_path = r"D:\med_data\ai"
    classify_file = os.path.join(base_path, "classify_t.xlsx")

    # 源文件夹
    train1_path = r"D:\med_data\ai\test1"  # DICOM文件（无后缀）
    train2_path = r"D:\med_data\ai\test2"  # TIF文件

    # 目标文件夹
    dest1_path = r"D:\med_data\ai\preprocess\aug\test"  # DICOM文件目标
    dest2_path = r"D:\med_data\ai\preprocess\aug\test(1)"  # TIF文件目标

    print("=" * 70)
    print("提取分类为5的图像文件")
    print("=" * 70)

    # 1. 检查源文件夹是否存在
    print("\n1. 检查文件夹...")
    paths_to_check = {
        "分类表格": classify_file,
        "DICOM源文件夹": train1_path,
        "TIF源文件夹": train2_path
    }

    all_paths_exist = True
    for name, path in paths_to_check.items():
        if os.path.exists(path):
            print(f"   ✓ {name}: {path}")
        else:
            print(f"   ✗ {name}: {path} 不存在")
            all_paths_exist = False

    if not all_paths_exist:
        print("\n错误：必要的文件夹不存在，请检查路径")
        return

    # 2. 创建目标文件夹
    print("\n2. 创建目标文件夹...")
    os.makedirs(dest1_path, exist_ok=True)
    os.makedirs(dest2_path, exist_ok=True)
    print(f"   ✓ DICOM目标文件夹: {dest1_path}")
    print(f"   ✓ TIF目标文件夹: {dest2_path}")

    # 3. 读取分类表格（只读前两列）
    print("\n3. 读取分类表格...")
    try:
        df = pd.read_excel(classify_file, header=None, usecols=[0, 1])
        df.columns = ['病历号', '分类']
        print(f"   原始数据：共 {len(df)} 条记录")

        # 显示分类分布
        print(f"   分类分布：")
        for cat in sorted(df['分类'].unique()):
            count = len(df[df['分类'] == cat])
            print(f"     分类 {cat}: {count} 条")

    except Exception as e:
        print(f"   读取表格失败: {e}")
        return

    # 4. 提取分类为5的病历号
    print("\n4. 提取分类为5的病历号...")
    class5_df = df[df['分类'] == 5].copy()
    class5_df = class5_df.reset_index(drop=True)

    print(f"   找到 {len(class5_df)} 个分类为5的病历号")

    if len(class5_df) == 0:
        print("   没有找到分类为5的病例，程序结束")
        return

    # 显示前10个病历号
    print(f"   前10个病历号：")
    for i, row in class5_df.head(10).iterrows():
        print(f"     {row['病历号']}")

    # 5. 复制文件
    print("\n5. 开始复制文件...")

    # 获取源文件夹中的所有文件（用于快速查找）
    train1_files = set(os.listdir(train1_path))
    train2_files = set(os.listdir(train2_path))

    print(f"   DICOM源文件夹中有 {len(train1_files)} 个文件")
    print(f"   TIF源文件夹中有 {len(train2_files)} 个文件")

    copied_dicom = 0
    copied_tif = 0
    missing_files = []
    missing_by_patient = {}  # 按病历号记录缺失文件

    # 后缀列表
    suffixes = ['_0', '_1']

    for idx, row in class5_df.iterrows():
        patient_id = row['病历号']

        # 显示进度
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"   处理进度: {idx + 1}/{len(class5_df)}")

        patient_missing = []

        for suffix in suffixes:
            # DICOM文件（无后缀）
            dicom_filename = f"{patient_id}{suffix}"
            dicom_src = os.path.join(train1_path, dicom_filename)
            dicom_dst = os.path.join(dest1_path, dicom_filename)

            # TIF文件
            tif_filename = f"{patient_id}{suffix}.tif"
            tif_src = os.path.join(train2_path, tif_filename)
            tif_dst = os.path.join(dest2_path, tif_filename)

            # 复制DICOM文件
            if os.path.exists(dicom_src):
                try:
                    shutil.copy2(dicom_src, dicom_dst)
                    copied_dicom += 1
                except Exception as e:
                    print(f"     复制失败 {dicom_filename}: {e}")
                    patient_missing.append(dicom_filename)
            else:
                patient_missing.append(dicom_filename)

            # 复制TIF文件
            if os.path.exists(tif_src):
                try:
                    shutil.copy2(tif_src, tif_dst)
                    copied_tif += 1
                except Exception as e:
                    print(f"     复制失败 {tif_filename}: {e}")
                    patient_missing.append(tif_filename)
            else:
                patient_missing.append(tif_filename)

        if patient_missing:
            missing_files.extend(patient_missing)
            missing_by_patient[patient_id] = patient_missing

    # 6. 输出结果统计
    print("\n6. 复制完成统计：")
    expected_dicom = len(class5_df) * 2  # 每个病历号有 _0 和 _1 两个DICOM
    expected_tif = len(class5_df) * 2  # 每个病历号有 _0 和 _1 两个TIF

    print(f"   DICOM文件:")
    print(f"     期望复制: {expected_dicom}")
    print(f"     实际复制: {copied_dicom}")
    print(f"     成功率: {(copied_dicom / expected_dicom * 100):.1f}%")

    print(f"   TIF文件:")
    print(f"     期望复制: {expected_tif}")
    print(f"     实际复制: {copied_tif}")
    print(f"     成功率: {(copied_tif / expected_tif * 100):.1f}%")

    print(f"   总文件:")
    print(f"     期望复制: {expected_dicom + expected_tif}")
    print(f"     实际复制: {copied_dicom + copied_tif}")

    # 7. 处理缺失文件
    if missing_files:
        print(f"\n7. 缺失文件统计：")
        print(f"   共缺失 {len(missing_files)} 个文件")

        # 按病历号统计缺失
        print(f"\n   按病历号缺失情况：")
        for patient_id, missing in missing_by_patient.items():
            if missing:
                print(f"     {patient_id}: 缺失 {len(missing)} 个文件")
                if len(missing) <= 4:  # 只显示具体缺失文件如果不多
                    for f in missing:
                        print(f"       - {f}")

        # 保存缺失文件列表
        missing_file_path = os.path.join(base_path, r"preprocess\aug\missing_files_class5.txt")
        try:
            with open(missing_file_path, 'w', encoding='utf-8') as f:
                f.write("分类为5的病例缺失文件列表：\n")
                f.write("=" * 50 + "\n\n")
                for patient_id, missing in missing_by_patient.items():
                    if missing:
                        f.write(f"{patient_id}:\n")
                        for m in missing:
                            f.write(f"  {m}\n")
                        f.write("\n")
            print(f"\n   缺失文件列表已保存至: {missing_file_path}")
        except Exception as e:
            print(f"   保存缺失文件列表失败: {e}")

    # 8. 生成汇总表格
    print("\n8. 生成汇总信息...")

    # 为每个病历号记录复制状态
    summary_data = []
    for _, row in class5_df.iterrows():
        patient_id = row['病历号']
        patient_missing = missing_by_patient.get(patient_id, [])

        # 检查哪些文件存在
        dicom_0_exists = os.path.exists(os.path.join(dest1_path, f"{patient_id}_0"))
        dicom_1_exists = os.path.exists(os.path.join(dest1_path, f"{patient_id}_1"))
        tif_0_exists = os.path.exists(os.path.join(dest2_path, f"{patient_id}_0.tif"))
        tif_1_exists = os.path.exists(os.path.join(dest2_path, f"{patient_id}_1.tif"))

        summary_data.append({
            '病历号': patient_id,
            'DICOM_0': '✓' if dicom_0_exists else '✗',
            'DICOM_1': '✓' if dicom_1_exists else '✗',
            'TIF_0': '✓' if tif_0_exists else '✗',
            'TIF_1': '✓' if tif_1_exists else '✗',
            '缺失文件数': len(patient_missing)
        })

    summary_df = pd.DataFrame(summary_data)

    # 保存汇总表格
    summary_file = os.path.join(base_path, r"preprocess\aug\class5_summary.xlsx")
    try:
        summary_df.to_excel(summary_file, index=False)
        print(f"   复制状态汇总表已保存至: {summary_file}")
    except Exception as e:
        print(f"   保存汇总表失败: {e}")

    # 9. 最终统计
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"\n最终统计：")
    print(f"   分类为5的病历号总数: {len(class5_df)}")
    print(f"   成功复制的DICOM文件: {copied_dicom}/{expected_dicom}")
    print(f"   成功复制的TIF文件: {copied_tif}/{expected_tif}")
    print(f"\n文件保存位置：")
    print(f"   DICOM文件: {dest1_path}")
    print(f"   TIF文件: {dest2_path}")
    print(f"   汇总表格: {summary_file}")

    if missing_files:
        print(f"\n注意：有 {len(missing_files)} 个文件缺失，请查看:")
        print(f"   {missing_file_path}")

    return {
        '病历号总数': len(class5_df),
        '复制的DICOM': copied_dicom,
        '复制的TIF': copied_tif,
        '缺失文件数': len(missing_files),
        '汇总表格': summary_df
    }


def preview_class5_patients():
    """
    预览分类为5的病历号，不实际复制文件
    """

    base_path = r"D:\med_data\ai"
    classify_file = os.path.join(base_path, "classify_t.xlsx")

    print("=" * 70)
    print("预览分类为5的病历号")
    print("=" * 70)

    # 读取表格
    try:
        df = pd.read_excel(classify_file, header=None, usecols=[0, 1])
        df.columns = ['病历号', '分类']
    except Exception as e:
        print(f"读取表格失败: {e}")
        return

    # 提取分类为5的病历号
    class5_df = df[df['分类'] == 5].copy()

    print(f"\n找到 {len(class5_df)} 个分类为5的病历号")

    if len(class5_df) > 0:
        print(f"\n前20个病历号：")
        for i, row in class5_df.head(20).iterrows():
            print(f"  {row['病历号']}")

        if len(class5_df) > 20:
            print(f"  ... 共 {len(class5_df)} 个")

        # 显示完整的分类分布
        print(f"\n所有分类的分布：")
        for cat in sorted(df['分类'].unique()):
            count = len(df[df['分类'] == cat])
            percentage = (count / len(df)) * 100
            print(f"  分类 {cat}: {count} 条 ({percentage:.1f}%)")

    return class5_df

if __name__ == "__main__":

    choice = input("请输入选择 (1/2，默认为2): ").strip()

    if choice == "1":
        preview_class5_patients()
    else:
        # 先预览确认
        print("\n" + "=" * 70)
        preview_class5_patients()
        print("\n" + "=" * 70)

        confirm = input("\n确认要提取这些文件并复制吗？(y/n): ").strip()
        if confirm.lower() == 'y':
            result = extract_class5_images()

            if result:
                print("\n提取结果摘要：")
                for key, value in result.items():
                    if key != '汇总表格':
                        print(f"  {key}: {value}")
        else:
            print("操作已取消")