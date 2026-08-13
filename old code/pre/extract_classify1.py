import pandas as pd
import os
import re
from collections import defaultdict


def create_label_table_deduplicate():
    """
    读取文件夹内的文件，去重后创建标签表格
    每个病历号只保留一个文件（去掉 _0 或 _1 后缀）
    第一列：病历号（去掉后缀）
    第二列：可手动设置的标签（默认为1）
    """

    # 定义路径
    dicom_path = r"D:\med_data\ai\preprocess\aug\5"
    output_file = r"D:\med_data\ai\preprocess\aug\labels5.xlsx"

    print("=" * 60)
    print("创建去重后的标签表格")
    print("=" * 60)
    print("功能：每个病历号只保留一个文件，去掉 _0/_1 后缀")

    # 检查文件夹是否存在
    if not os.path.exists(dicom_path):
        print(f"错误：文件夹不存在 - {dicom_path}")
        return

    # 1. 获取所有文件
    print(f"\n1. 读取文件夹: {dicom_path}")

    try:
        all_files = os.listdir(dicom_path)
        # 过滤掉文件夹，只保留文件
        files = [f for f in all_files if os.path.isfile(os.path.join(dicom_path, f))]

        print(f"   找到 {len(files)} 个原始文件")
        if len(files) > 0:
            print(f"   原始文件示例: {files[:5]}")
        else:
            print("   警告：文件夹中没有文件")
            return

    except Exception as e:
        print(f"   读取文件夹失败: {e}")
        return

    # 2. 提取病历号（去掉 _0 或 _1 后缀）
    print("\n2. 提取病历号（去重）...")

    patient_ids = set()
    file_mapping = {}  # 用于记录每个病历号对应的原始文件

    # 定义正则表达式模式：匹配 ANY_数字_数字 格式
    pattern = r'^(ANY_\d+)_[01]$'

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            patient_id = match.group(1)  # 提取 ANY_数字 部分
            if patient_id not in file_mapping:
                # 只保留第一个遇到的该病历号的文件
                file_mapping[patient_id] = filename
                patient_ids.add(patient_id)

    # 将去重后的病历号排序
    sorted_patients = sorted(list(patient_ids))

    print(f"   去重后共有 {len(sorted_patients)} 个唯一病历号")
    print(f"   去重后示例: {sorted_patients[:10]}")

    # 显示去重前后的对比
    print(f"\n   去重前后对比（前10个）：")
    print(f"   {'原始文件':<20} -> {'病历号':<15}")
    print(f"   {'-' * 35}")

    count = 0
    for patient_id in sorted_patients[:10]:
        original_file = file_mapping[patient_id]
        print(f"   {original_file:<20} -> {patient_id:<15}")
        count += 1
        if count >= 10:
            break

    # 3. 询问用户设置默认标签
    print("\n3. 设置标签值")
    while True:
        try:
            default_label = input("   请输入默认标签值（1-7，默认为1）: ").strip()
            if default_label == "":
                default_label = 1
                print(f"   使用默认标签: {default_label}")
                break
            else:
                default_label = int(default_label)
                if 1 <= default_label <= 7:
                    print(f"   使用标签: {default_label}")
                    break
                else:
                    print("   标签值应在1-7之间")
        except ValueError:
            print("   输入无效，请输入整数")

    # 4. 创建数据框
    print("\n4. 创建表格数据...")

    data = {
        '病历号': sorted_patients,
        '标签': [default_label] * len(sorted_patients)
    }

    df = pd.DataFrame(data)

    print(f"   表格大小: {len(df)} 行 × {len(df.columns)} 列")

    # 5. 保存Excel文件
    print(f"\n5. 保存表格到: {output_file}")

    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存Excel文件
        df.to_excel(output_file, index=False)
        print(f"   保存成功！")

        # 验证保存的文件
        test_df = pd.read_excel(output_file)
        print(f"   验证 - 已保存 {len(test_df)} 行数据")

        # 显示前几行
        print(f"\n   表格预览（前20行）:")
        print(test_df.head(20).to_string(index=False))

    except Exception as e:
        print(f"   保存失败: {e}")
        return

    # 6. 生成统计信息
    print("\n6. 生成统计信息...")

    # 创建映射文件，记录每个病历号对应的原始文件
    mapping_file = r"D:\med_data\ai\preprocess\aug\file_mapping.xlsx"
    mapping_data = []
    for patient_id in sorted_patients:
        mapping_data.append({
            '病历号': patient_id,
            '保留的原始文件': file_mapping[patient_id],
            '标签': default_label
        })

    mapping_df = pd.DataFrame(mapping_data)

    try:
        mapping_df.to_excel(mapping_file, index=False)
        print(f"   文件映射表已保存到: {mapping_file}")
    except Exception as e:
        print(f"   保存映射表失败: {e}")

    # 显示汇总信息
    print(f"\n   {'=' * 40}")
    print(f"   处理完成汇总：")
    print(f"   {'原始文件数':<20}: {len(files)}")
    print(f"   {'去重后病历号数':<20}: {len(sorted_patients)}")
    print(f"   {'去重比例':<20}: {(len(sorted_patients) / len(files) * 100):.1f}%")
    print(f"   {'默认标签值':<20}: {default_label}")
    print(f"   {'=' * 40}")

    print("\n" + "=" * 60)
    print("操作完成！")
    print("=" * 60)
    print(f"\n后续操作提示：")
    print(f"1. 主表格: {output_file}")
    print(f"   - 第一列：病历号（已去重，格式 ANY_数字）")
    print(f"   - 第二列：标签值（当前为{default_label}，可在Excel中修改）")
    print(f"2. 映射表: {mapping_file}")
    print(f"   - 记录了每个病历号对应的原始文件")
    print(f"3. 如需调整标签，直接用Excel打开主表格修改第二列即可")

    return df, mapping_df


def create_label_table_with_selection():
    """
    高级版本：允许选择保留 _0 还是 _1 的文件
    """

    dicom_path = r"D:\med_data\ai\preprocess\aug\2dicom"
    output_file = r"D:\med_data\ai\preprocess\aug\labels2.xlsx"

    print("=" * 60)
    print("创建标签表格（可选择保留后缀）")
    print("=" * 60)

    # 检查文件夹
    if not os.path.exists(dicom_path):
        print(f"错误：文件夹不存在 - {dicom_path}")
        return

    # 获取文件列表
    all_files = os.listdir(dicom_path)
    files = [f for f in all_files if os.path.isfile(os.path.join(dicom_path, f))]

    print(f"\n找到 {len(files)} 个原始文件")

    # 按病历号分组
    patient_groups = defaultdict(list)
    pattern = r'^(ANY_\d+)_([01])$'

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            patient_id = match.group(1)
            suffix = match.group(2)
            patient_groups[patient_id].append((suffix, filename))

    print(f"发现 {len(patient_groups)} 个唯一病历号")

    # 询问保留哪个后缀
    print("\n请选择要保留的文件后缀：")
    print("1. 保留 _0 文件")
    print("2. 保留 _1 文件")
    print("3. 手动为每个病历号选择")

    choice = input("请输入选择 (1/2/3，默认为1): ").strip()

    selected_files = []

    if choice == "2":
        # 保留 _1
        for patient_id, suffixes in patient_groups.items():
            suffixes_dict = dict(suffixes)
            if '1' in suffixes_dict:
                selected_files.append((patient_id, suffixes_dict['1']))
            elif '0' in suffixes_dict:
                selected_files.append((patient_id, suffixes_dict['0']))
        print(f"选择保留 _1 文件，共 {len(selected_files)} 个病历号")

    elif choice == "3":
        # 手动选择
        print("\n开始手动选择（输入 q 退出）...")
        for patient_id, suffixes in sorted(patient_groups.items()):
            suffixes_dict = dict(suffixes)
            options = []
            if '0' in suffixes_dict:
                options.append(('0', suffixes_dict['0']))
            if '1' in suffixes_dict:
                options.append(('1', suffixes_dict['1']))

            print(f"\n病历号: {patient_id}")
            for idx, (suffix, filename) in enumerate(options):
                print(f"  {idx + 1}. {filename}")

            while True:
                sel = input(f"请选择要保留的文件 (1-{len(options)}，或输入q跳过): ").strip()
                if sel.lower() == 'q':
                    break
                try:
                    sel_idx = int(sel) - 1
                    if 0 <= sel_idx < len(options):
                        selected_files.append((patient_id, options[sel_idx][1]))
                        break
                    else:
                        print(f"请输入1-{len(options)}之间的数字")
                except ValueError:
                    print("输入无效")
    else:
        # 默认保留 _0
        for patient_id, suffixes in patient_groups.items():
            suffixes_dict = dict(suffixes)
            if '0' in suffixes_dict:
                selected_files.append((patient_id, suffixes_dict['0']))
            elif '1' in suffixes_dict:
                selected_files.append((patient_id, suffixes_dict['1']))
        print(f"选择保留 _0 文件，共 {len(selected_files)} 个病历号")

    # 按病历号排序
    selected_files.sort(key=lambda x: x[0])

    # 设置标签
    print("\n设置标签值...")
    while True:
        try:
            default_label = input("请输入默认标签值（1-7，默认为1）: ").strip()
            if default_label == "":
                default_label = 1
                break
            else:
                default_label = int(default_label)
                if 1 <= default_label <= 7:
                    break
                else:
                    print("标签值应在1-7之间")
        except ValueError:
            print("输入无效，请输入整数")

    # 创建表格
    data = {
        '病历号': [p[0] for p in selected_files],
        '保留的文件名': [p[1] for p in selected_files],
        '标签': [default_label] * len(selected_files)
    }

    df = pd.DataFrame(data)

    # 保存
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_excel(output_file, index=False)
        print(f"\n表格已保存到: {output_file}")
        print(f"共保存 {len(df)} 行数据")

        # 预览
        print(f"\n表格预览（前20行）:")
        print(df.head(20).to_string(index=False))

    except Exception as e:
        print(f"保存失败: {e}")

    print("\n" + "=" * 60)
    print("操作完成！")
    print("=" * 60)


if __name__ == "__main__":
    print("请选择功能:")
    print("1. 基础去重版本（自动去重，保留第一个遇到的文件）")
    print("2. 高级选择版本（可选择保留 _0 或 _1）")

    choice = input("请输入选择 (1/2，默认为1): ").strip()

    if choice == "2":
        create_label_table_with_selection()
    else:
        create_label_table_deduplicate()