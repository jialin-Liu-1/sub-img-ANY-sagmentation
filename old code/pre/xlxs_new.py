import pandas as pd
import re
import os
from collections import defaultdict


def convert_to_anatomical_location(vessel_type):
    """
    基于解剖位置进行分类，0-7对应Bouthillier分段

    分类规则：
    0: 无法确定解剖位置/非ICA系统且位置不明
    1: 颈段区域 (C1区域) - 颈内动脉颈段及附近
    2: 岩段区域 (C2区域) - 颈内动脉岩段及附近
    3: 破裂孔段区域 (C3区域) - 破裂孔附近
    4: 海绵窦区域 (C4区域) - 海绵窦内及附近
    5: 床突区域 (C5区域) - 前床突周围
    6: 眼动脉区域 (C6区域) - 视神经管、眼动脉起点周围
    7: 交通段区域 (C7区域) - 后交通动脉、脉络膜前动脉区域
    """
    if pd.isna(vessel_type):
        return 0

    original = str(vessel_type)
    text = original.lower().strip()

    # 1. 首先检查是否有明确的分段描述
    # 检查C1-C7明确标记
    for i in range(1, 8):
        if f'c{i}' in text or f'c{i} ' in text:
            return i

    # 2. 海绵窦区域 (C4) - 最明确的特征
    if 'cavernous' in text:
        return 4

    # 3. 眼动脉区域 (C6) - 包括各种相关描述
    if any(keyword in text for keyword in [
        'ophthalmic', 'paraophthalmic', 'ophtalmic', 'opthalmic',
        'superior hypophyseal', 'hypophyseal', 'sha', 'hypopheseal',
        'superior hypophyseal', 'sup hypophyseal'
    ]):
        return 6

    # 4. 交通段区域 (C7) - 后交通动脉相关
    if any(keyword in text for keyword in [
        'supraclinoid', 'supraclanoid',
        'pcom', 'posterior communicating', 'p-comm', 'pcomm',
        'choroidal', 'choroid'
    ]):
        return 7

    # 5. 床突区域 (C5) - 床突相关
    if any(keyword in text for keyword in [
        'paraclinoid', 'clinoid', 'carotid cave', 'carotid-cave'
    ]):
        return 5

    # 6. 岩段区域 (C2)
    if any(keyword in text for keyword in ['petrous', 'petro']):
        return 2

    # 7. 颈段区域 (C1)
    if 'cervical' in text:
        return 1

    # 8. 破裂孔段区域 (C3) - 较少见
    if 'lacerum' in text:
        return 3

    # 9. 特殊处理：根据解剖相邻关系
    # 前交通动脉 (ACoA) 在视交叉上方，接近眼动脉区域
    if any(keyword in text for keyword in ['acom', 'anterior communicating', 'anterior cerebral']):
        return 6  # 视为眼动脉区域

    # 10. 特殊处理：根据位置描述推断
    # "ICA bifurcation" - ICA分叉处属于C7
    if 'bifurcation' in text and 'ica' in text:
        return 7

    # "optic" 相关 - 视神经附近属于眼动脉区域
    if 'optic' in text and ('nerve' in text or 'canal' in text):
        return 6

    # "carotid siphon" - 颈动脉虹吸部通常在海绵窦段
    if 'siphon' in text:
        return 4

    # 11. 处理包含"ICA"但位置模糊的描述
    if 'ica' in text:
        # 如果只有简单的"ICA"或"L ICA"，返回0（需要人工复核）
        simple_patterns = ['^ica$', '^l ica$', '^r ica$', '^left ica$', '^right ica$']
        if any(re.match(pattern, text.strip()) for pattern in simple_patterns):
            return 0

        # 检查是否有其他位置线索
        if any(keyword in text for keyword in ['segment', 'portion', 'region', 'area']):
            # 有位置描述但不够具体，返回0
            return 0

    # 12. 非ICA系统的特殊处理
    # MCA - 大脑中动脉，大多数不易对应ICA分段
    if any(keyword in text for keyword in ['mca', 'middle cerebral']):
        # M1起始段可以对应C7
        if 'm1' in text or ('origin' in text and 'ica' in text):
            return 7
        return 0

    # PCA - 大脑后动脉
    if any(keyword in text for keyword in ['pca', 'posterior cerebral']):
        return 0

    # 基底动脉
    if 'basilar' in text:
        return 0

    # 椎动脉
    if any(keyword in text for keyword in ['vertebral', 'v4']):
        return 0

    # 13. 其他情况返回0
    return 0


def get_anatomical_description(location_num):
    """获取解剖位置描述"""
    descriptions = {
        0: "无法确定解剖位置",
        1: "颈段区域 (C1区域)",
        2: "岩段区域 (C2区域)",
        3: "破裂孔段区域 (C3区域)",
        4: "海绵窦区域 (C4区域)",
        5: "床突区域 (C5区域)",
        6: "眼动脉区域 (C6区域)",
        7: "交通段区域 (C7区域)"
    }
    return descriptions.get(location_num, f"未知区域 ({location_num})")


def analyze_classification_details(original_text, location_num):
    """分析分类的详细依据"""
    text = original_text.lower()
    details = {
        '主要依据': '',
        '关键词': [],
        '置信度': '中'
    }

    if location_num == 0:
        details['主要依据'] = '缺乏明确的解剖位置信息'
        details['置信度'] = '低'

        # 记录可能的关键词
        if 'aneurysm' in text:
            details['关键词'].append('仅提到动脉瘤')
        if 'ica' in text and not any(keyword in text for keyword in
                                     ['cavernous', 'ophthalmic', 'supraclinoid',
                                      'paraclinoid', 'petrous', 'cervical', 'sha']):
            details['关键词'].append('单纯ICA描述')

    elif location_num == 1:
        details['主要依据'] = '位于颈段区域 (C1)'
        if 'cervical' in text:
            details['关键词'].append('cervical')
            details['置信度'] = '高'

    elif location_num == 2:
        details['主要依据'] = '位于岩段区域 (C2)'
        if 'petrous' in text:
            details['关键词'].append('petrous')
            details['置信度'] = '高'
        elif 'petro' in text:
            details['关键词'].append('petro')
            details['置信度'] = '中'

    elif location_num == 3:
        details['主要依据'] = '位于破裂孔段区域 (C3)'
        if 'lacerum' in text:
            details['关键词'].append('lacerum')
            details['置信度'] = '高'

    elif location_num == 4:
        details['主要依据'] = '位于海绵窦区域 (C4)'
        if 'cavernous' in text:
            details['关键词'].append('cavernous')
            details['置信度'] = '高'
        elif 'siphon' in text:
            details['关键词'].append('siphon')
            details['置信度'] = '中'

    elif location_num == 5:
        details['主要依据'] = '位于床突区域 (C5)'
        if 'paraclinoid' in text:
            details['关键词'].append('paraclinoid')
            details['置信度'] = '高'
        elif 'clinoid' in text:
            details['关键词'].append('clinoid')
            details['置信度'] = '高'
        elif 'carotid cave' in text:
            details['关键词'].append('carotid cave')
            details['置信度'] = '高'
        elif 'cave' in text and 'carotid' in text:
            details['关键词'].append('carotid cave')
            details['置信度'] = '中'

    elif location_num == 6:
        details['主要依据'] = '位于眼动脉区域 (C6)'
        if 'ophthalmic' in text:
            details['关键词'].append('ophthalmic')
            details['置信度'] = '高'
        elif 'paraophthalmic' in text:
            details['关键词'].append('paraophthalmic')
            details['置信度'] = '高'
        elif 'sha' in text or 'hypophyseal' in text:
            details['关键词'].append('SHA/hypophyseal')
            details['置信度'] = '高'
        elif 'acom' in text or 'anterior communicating' in text:
            details['关键词'].append('ACoA')
            details['置信度'] = '中'
            details['主要依据'] = '前交通动脉位于视交叉上方，接近眼动脉区域'
        elif 'optic' in text:
            details['关键词'].append('optic')
            details['置信度'] = '中'

    elif location_num == 7:
        details['主要依据'] = '位于交通段区域 (C7)'
        if 'supraclinoid' in text:
            details['关键词'].append('supraclinoid')
            details['置信度'] = '高'
        elif 'pcom' in text or 'posterior communicating' in text:
            details['关键词'].append('PCOM')
            details['置信度'] = '高'
        elif 'choroidal' in text:
            details['关键词'].append('choroidal')
            details['置信度'] = '高'
        elif 'bifurcation' in text and 'ica' in text:
            details['关键词'].append('ICA bifurcation')
            details['置信度'] = '高'
        elif 'm1' in text and 'origin' in text:
            details['关键词'].append('M1 origin')
            details['置信度'] = '中'
            details['主要依据'] = 'M1段从ICA分叉处发出，属于交通段区域'

    # 如果没有找到明确关键词，降低置信度
    if not details['关键词']:
        details['置信度'] = '低'
        details['主要依据'] += '（根据解剖关系推断）'

    details['关键词'] = ', '.join(details['关键词'])
    return details


def extract_case_info_with_anatomical_classification():
    """提取病历号和动脉瘤解剖位置分类到新表格"""
    # 文件路径
    file_path = r"D:\med_data\ai\pipeline.xlsx"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    try:
        # 读取Excel文件
        print("正在读取Excel文件...")
        df = pd.read_excel(file_path)

        # 获取病例号列（第一列）和动脉瘤类型列（第13列）
        if df.shape[1] >= 13:
            case_numbers = df.iloc[:, 0]  # 第一列
            aneurysm_types = df.iloc[:, 12]  # 第13列
        else:
            print("错误：表格列数不足13列")
            return

        # 准备存储结果的数据
        results = []
        conversion_details = []
        classification_stats = defaultdict(lambda: {'count': 0, 'examples': []})

        print(f"共找到 {len(case_numbers)} 个病例")
        print("正在提取和转换数据...")
        print("-" * 80)

        # 逐病例处理
        processed_count = 0
        for idx, (case_num, aneurysm_type) in enumerate(zip(case_numbers, aneurysm_types), 1):
            # 清理数据
            case_num_str = str(case_num).strip()

            # 检查动脉瘤类型是否为空
            if pd.isna(aneurysm_type):
                continue

            aneurysm_type_str = str(aneurysm_type).strip()

            # 如果动脉瘤类型为空字符串，跳过
            if not aneurysm_type_str:
                continue

            # 转换为解剖位置编号
            location_num = convert_to_anatomical_location(aneurysm_type_str)

            # 分析分类依据
            classification_details = analyze_classification_details(aneurysm_type_str, location_num)

            # 保存结果
            results.append({
                '病历号': case_num_str,
                '解剖位置编号': location_num,
                '解剖位置描述': get_anatomical_description(location_num)
            })

            # 保存详细转换信息
            conversion_details.append({
                '病历号': case_num_str,
                '原始动脉瘤类型': aneurysm_type_str,
                '解剖位置编号': location_num,
                '解剖位置描述': get_anatomical_description(location_num),
                '分类置信度': classification_details['置信度'],
                '主要依据': classification_details['主要依据'],
                '关键词': classification_details['关键词']
            })

            # 更新统计信息（只保存前3个示例）
            if classification_stats[location_num]['count'] < 3:
                classification_stats[location_num]['examples'].append(aneurysm_type_str)
            classification_stats[location_num]['count'] += 1
            processed_count += 1

            # 显示进度
            if idx % 50 == 0:
                print(f"已处理 {idx}/{len(case_numbers)} 行数据")

        print(f"完成！共处理 {processed_count} 个有效病例")

        # 创建新表格的数据框
        df_new = pd.DataFrame(results)

        # 创建详细转换信息的数据框
        df_details = pd.DataFrame(conversion_details)

        # 打印统计信息
        print_statistics(classification_stats, processed_count)

        # 保存结果
        save_results(df_new, df_details, classification_stats, processed_count)

        return df_new, df_details

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_statistics(stats, total_cases):
    """打印统计信息"""
    print("\n" + "=" * 80)
    print("解剖位置分类统计结果")
    print("=" * 80)
    print(f"{'分类编号':<10} {'解剖位置':<25} {'病例数':<10} {'百分比':<10} {'示例':<25}")
    print("-" * 80)

    # 按分类编号排序
    for location_num in sorted(stats.keys()):
        count = stats[location_num]['count']
        percentage = (count / total_cases) * 100 if total_cases > 0 else 0

        # 获取示例
        examples = stats[location_num]['examples']
        if examples:
            # 取第一个示例，最多显示20个字符
            example = examples[0]
            if len(example) > 22:
                example = example[:19] + "..."
        else:
            example = ""

        print(f"{location_num:<10} {get_anatomical_description(location_num):<25} "
              f"{count:<10} {percentage:.1f}%     {example:<25}")

    print("-" * 80)
    print(f"总计有效病例: {total_cases} 个")

    # 计算可归类比例（排除0类）
    classified_count = total_cases - stats.get(0, {}).get('count', 0)
    classified_percentage = (classified_count / total_cases) * 100 if total_cases > 0 else 0
    print(f"可归类到解剖位置的比例: {classified_percentage:.1f}% ({classified_count}/{total_cases})")

    # 显示各类占比饼图（文本形式）
    print("\n各类占比（文本饼图）:")
    for location_num in sorted(stats.keys()):
        count = stats[location_num]['count']
        if count > 0:
            bar_length = int((count / total_cases) * 50)
            bar = '█' * bar_length
            print(f"{location_num}: {get_anatomical_description(location_num)[:15]}... {bar} {count}例")


def save_results(df_new, df_details, stats, total_cases):
    """保存结果到文件"""
    try:
        output_file = "病例_解剖位置分类表.xlsx"

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. 保存主要结果表
            df_new.to_excel(writer, sheet_name='解剖位置分类', index=False)

            # 2. 保存详细转换表
            df_details.to_excel(writer, sheet_name='详细转换信息', index=False)

            # 3. 保存统计表
            stats_data = []
            for location_num in sorted(stats.keys()):
                count = stats[location_num]['count']
                percentage = (count / total_cases) * 100 if total_cases > 0 else 0

                # 获取示例
                examples = stats[location_num]['examples']
                example_str = "; ".join(examples) if examples else ""

                stats_data.append({
                    '解剖位置编号': location_num,
                    '解剖位置描述': get_anatomical_description(location_num),
                    '病例数': count,
                    '百分比': f"{percentage:.1f}%",
                    '示例病例': example_str
                })

            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)

            # 4. 保存分类规则说明
            rules_data = [
                ["分类编号", "解剖位置", "Bouthillier分段", "典型关键词/描述", "包含的其他血管"],
                [0, "无法确定解剖位置", "N/A", "单纯ICA, aneurysm, 位置不明", "所有无法定位的"],
                [1, "颈段区域", "C1 - Cervical", "cervical, neck", "颈段ICA"],
                [2, "岩段区域", "C2 - Petrous", "petrous, petro, carotid canal", "岩段ICA"],
                [3, "破裂孔段区域", "C3 - Lacerum", "lacerum", "破裂孔段ICA"],
                [4, "海绵窦区域", "C4 - Cavernous", "cavernous, siphon", "海绵窦段ICA"],
                [5, "床突区域", "C5 - Clinoidal", "paraclinoid, clinoid, carotid cave", "床突段ICA"],
                [6, "眼动脉区域", "C6 - Ophthalmic", "ophthalmic, paraophthalmic, SHA, ACoA, optic",
                 "眼动脉、垂体上动脉、前交通动脉"],
                [7, "交通段区域", "C7 - Communicating", "supraclinoid, PCOM, choroidal, bifurcation",
                 "后交通动脉、脉络膜前动脉、ICA分叉"]
            ]

            rules_df = pd.DataFrame(rules_data[1:], columns=rules_data[0])
            rules_df.to_excel(writer, sheet_name='分类规则', index=False)

        print(f"\n新表格已保存到: {output_file}")
        print("工作表包含:")
        print("1. '解剖位置分类' - 病历号和分类编号")
        print("2. '详细转换信息' - 原始类型、分类编号、置信度和依据")
        print("3. '统计信息' - 详细分类统计")
        print("4. '分类规则' - 解剖位置分类说明")

        # 显示新表格的前几行
        print("\n新表格前10行预览:")
        print(df_new.head(10).to_string())

    except Exception as e:
        print(f"保存结果时发生错误: {e}")
        import traceback
        traceback.print_exc()


def validate_classification():
    """验证分类结果的准确性"""
    test_cases = [
        # (原始描述, 期望分类, 说明)
        ("ICA: Cavernous", 4, "明确海绵窦区域"),
        ("ICA: Paraophthalmic", 6, "眼动脉旁区域"),
        ("ICA: Supraclinoid", 7, "床突上段"),
        ("Left ophthalmic", 6, "眼动脉区域"),
        ("right superior hypophyseal", 6, "垂体上动脉（眼段）"),
        ("L PCOM", 7, "后交通动脉"),
        ("ICA: Petrous", 2, "岩段"),
        ("carotid cave", 5, "颈动脉窝（床突段）"),
        ("ICA aneurysm", 0, "单纯ICA，位置不明"),
        ("ICA: Cavernous or ICA: Supraclinoid", 4, "取第一个明确位置"),
        ("anterior communicating artery", 6, "前交通动脉在视交叉上方"),
        ("Cervical carotid", 1, "颈段"),
        ("anterior choroidal", 7, "脉络膜前动脉"),
        ("MCA", 0, "大脑中动脉不易对应"),
        ("M1 origin from ICA", 7, "M1起始于ICA分叉"),
        ("basilar tip aneurysm", 0, "基底动脉尖"),
        ("Right cavernous ICA", 4, "海绵窦段ICA"),
        ("L paraclinoid giant aneurysm", 5, "床突旁"),
        ("R superior hypophyseal artery", 6, "垂体上动脉"),
        ("optic nerve aneurysm", 6, "视神经附近"),
        ("carotid siphon", 4, "颈动脉虹吸部"),
        ("ICA bifurcation", 7, "ICA分叉处"),
        ("paraophthalmic aneurysms", 6, "眼动脉旁"),
        ("L cavernous internal carotid artery", 4, "海绵窦段ICA"),
        ("Right A1-A2 aneurysm", 6, "前交通动脉区域"),
        ("supraclinoidal ICA", 7, "床突上段"),
        ("petrocavernous junction", 4, "岩海绵窦交界"),
        ("clinoidal aneurysm", 5, "床突"),
        ("SHA aneurysm", 6, "垂体上动脉"),
        ("PCOM artery aneurysm", 7, "后交通动脉"),
    ]

    print("解剖位置分类验证:")
    print("=" * 90)
    print(f"{'原始描述':<35} {'期望':<5} {'实际':<5} {'状态':<4} {'说明':<30}")
    print("-" * 90)

    correct = 0
    total = len(test_cases)

    for original, expected, desc in test_cases:
        actual = convert_to_anatomical_location(original)
        status = "✓" if actual == expected else "✗"
        if status == "✓":
            correct += 1

        print(f"{original:<35} {expected:<5} {actual:<5} {status:<4} {desc:<30}")

    print("=" * 90)
    accuracy = (correct / total) * 100
    print(f"测试准确率: {accuracy:.1f}% ({correct}/{total})")

    # 显示分类分布
    print("\n测试用例分类分布:")
    test_results = defaultdict(int)
    for original, expected, _ in test_cases:
        actual = convert_to_anatomical_location(original)
        test_results[actual] += 1

    for loc_num in sorted(test_results.keys()):
        print(f"  分类{loc_num}: {get_anatomical_description(loc_num)[:15]}... {test_results[loc_num]}例")

    return accuracy


def main():
    """主函数"""
    print("=" * 80)
    print("动脉瘤解剖位置分类提取程序")
    print("基于Bouthillier分段法的解剖区域分类 (0-7)")
    print("=" * 80)

    # 验证分类规则
    print("\n正在验证分类规则...")
    validate_classification()

    print("\n" + "=" * 80)
    print("开始处理病例数据...")
    print("-" * 80)

    # 执行提取和转换
    results = extract_case_info_with_anatomical_classification()

    if results:
        print("\n" + "=" * 80)
        print("程序执行完成！")
        print("=" * 80)
        print("\n总结:")
        print("1. 基于解剖区域而不是血管名称进行分类")
        print("2. 同一解剖区域的不同血管归为同一类")
        print("3. 提供分类置信度和依据")
        print("4. 统计可归类比例")
        print("5. 生成详细的分类规则文档")


if __name__ == "__main__":
    main()