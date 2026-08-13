import pandas as pd
from collections import Counter
import os


def extract_and_statistics_vessel_segments():
    # 文件路径
    file_path = r"D:\med_data\ai\pipeline.xlsx"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    try:
        # 读取Excel文件，M列是第13列（索引从0开始，所以是第12列）
        # 使用列名'M'或索引12
        df = pd.read_excel(file_path)

        # 获取第13列（M列）的数据
        # 方法1：使用列索引（从0开始）
        if df.shape[1] >= 13:  # 确保至少有13列
            vessel_segments = df.iloc[:, 12]  # 第13列的索引是12
        else:
            # 方法2：尝试使用列名'M'
            try:
                vessel_segments = df['M']
            except KeyError:
                # 方法3：尝试查找包含"血管段"或相关关键词的列
                column_found = False
                for col in df.columns:
                    if any(keyword in str(col).lower() for keyword in ['血管段', '血管', '类型', 'segment', 'vessel']):
                        vessel_segments = df[col]
                        column_found = True
                        print(f"使用列名: {col}")
                        break

                if not column_found:
                    print("错误：找不到第13列或相关血管段信息列")
                    return

        # 清理数据：去除空格，处理空值
        vessel_segments_clean = vessel_segments.dropna().astype(str).str.strip()

        # 去除空字符串
        vessel_segments_clean = vessel_segments_clean[vessel_segments_clean != '']

        if len(vessel_segments_clean) == 0:
            print("警告：没有找到有效的血管段类型数据")
            return

        # 统计不同类型
        type_counter = Counter(vessel_segments_clean)

        # 获取所有唯一类型
        unique_types = sorted(type_counter.keys())

        # 打印统计结果
        print("=" * 60)
        print("血管段类型统计结果")
        print("=" * 60)
        print(f"总病例数: {len(vessel_segments_clean)}")
        print(f"血管段类型总数: {len(unique_types)}")
        print("\n所有血管段类型（按字母顺序排序）:")
        print("-" * 40)

        for i, type_name in enumerate(unique_types, 1):
            count = type_counter[type_name]
            print(f"{i:3d}. {type_name:<30} 出现次数: {count:3d}")

        print("-" * 40)

        # 按出现频率排序打印
        print("\n按出现频率排序:")
        print("-" * 40)
        sorted_types = sorted(type_counter.items(), key=lambda x: x[1], reverse=True)

        for i, (type_name, count) in enumerate(sorted_types, 1):
            print(f"{i:3d}. {type_name:<30} 出现次数: {count:3d}")

        print("=" * 60)

        # 保存结果到文件
        save_results(unique_types, type_counter, sorted_types)

        return unique_types, type_counter

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        print(f"错误详情: {str(e)}")
        return None


def save_results(unique_types, type_counter, sorted_types):
    """保存统计结果到文件"""
    try:
        output_file = "vessel_segments_statistics.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("血管段类型统计结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"总病例数: {sum(type_counter.values())}\n")
            f.write(f"血管段类型总数: {len(unique_types)}\n")

            f.write("\n所有血管段类型（按字母顺序排序）:\n")
            f.write("-" * 40 + "\n")

            for i, type_name in enumerate(unique_types, 1):
                count = type_counter[type_name]
                f.write(f"{i:3d}. {type_name:<30} 出现次数: {count:3d}\n")

            f.write("\n按出现频率排序:\n")
            f.write("-" * 40 + "\n")

            for i, (type_name, count) in enumerate(sorted_types, 1):
                f.write(f"{i:3d}. {type_name:<30} 出现次数: {count:3d}\n")

        print(f"详细统计结果已保存到: {output_file}")

        # 同时保存为CSV文件以便进一步分析
        df_results = pd.DataFrame({
            '血管段类型': list(type_counter.keys()),
            '出现次数': list(type_counter.values())
        })
        df_results.to_csv('vessel_segments_statistics.csv', index=False, encoding='utf-8-sig')
        print(f"CSV格式统计结果已保存到: vessel_segments_statistics.csv")

    except Exception as e:
        print(f"保存结果时发生错误: {e}")


def show_data_preview(file_path):
    """显示数据预览"""
    try:
        df = pd.read_excel(file_path)
        print("数据预览:")
        print(f"数据形状: {df.shape} (行数, 列数)")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head())
        print("\n第13列（M列）前10个值:")
        if df.shape[1] >= 13:
            print(df.iloc[:10, 12])
    except Exception as e:
        print(f"数据预览失败: {e}")


if __name__ == "__main__":
    print("正在读取Excel文件...")

    # 如果需要查看数据结构和列名，可以取消下面的注释
    # show_data_preview(r"D:\med_data\ai\pipeline.xlsx")

    # 提取并统计血管段类型
    results = extract_and_statistics_vessel_segments()

    if results:
        unique_types, type_counter = results
        print("\n程序执行完成！")