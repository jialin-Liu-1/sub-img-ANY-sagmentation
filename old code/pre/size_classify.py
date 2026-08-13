import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def classify_size(radius_ratio, thresholds):
    """
    根据radius_ratio值和阈值进行分类

    参数:
    radius_ratio: 要分类的值
    thresholds: 阈值列表 [th1, th2, th3]

    返回:
    0: 极小 (< th1)
    1: 小 (th1 <= value < th2)
    2: 中等 (th2 <= value <= th3)
    3: 大 (> th3)
    """
    if radius_ratio < thresholds[0]:
        return 0  # 极小
    elif thresholds[0] <= radius_ratio < thresholds[1]:
        return 1  # 小
    elif thresholds[1] <= radius_ratio <= thresholds[2]:
        return 2  # 中等
    else:  # radius_ratio > thresholds[2]
        return 3  # 大


def get_max_threshold(size_class, class_thresholds):
    """
    根据尺寸分类获取对应的最大阈值

    参数:
    size_class: 尺寸分类 (0, 1, 2, 3)
    class_thresholds: 各类别的最大阈值字典

    返回:
    对应类别的最大阈值
    """
    return class_thresholds.get(size_class, None)


def analyze_distribution(df, thresholds, class_thresholds):
    """分析当前阈值下的分布情况"""
    df_temp = df.copy()
    df_temp['size_classes'] = df_temp['radius_ratio'].apply(lambda x: classify_size(x, thresholds))

    class_counts = df_temp['size_classes'].value_counts().sort_index()

    print(f"\n当前分类阈值设置: [{thresholds[0]}, {thresholds[1]}, {thresholds[2]}]")
    print("=" * 60)

    size_labels = {
        0: f'极小 (<{thresholds[0]}) [最大阈值: {class_thresholds[0]:.4f}]',
        1: f'小 ({thresholds[0]}-{thresholds[1]}) [最大阈值: {class_thresholds[1]:.4f}]',
        2: f'中等 ({thresholds[1]}-{thresholds[2]}) [最大阈值: {class_thresholds[2]:.4f}]',
        3: f'大 (>{thresholds[2]}) [最大阈值: {class_thresholds[3]:.4f}]'
    }

    for class_id in range(4):
        count = class_counts.get(class_id, 0)
        percentage = count / len(df) * 100
        print(f"类别 {class_id} ({size_labels[class_id]}): {count:4d} 例 ({percentage:5.1f}%)")

    print("=" * 60)

    # 计算分布的均衡性（使用变异系数）
    counts = [class_counts.get(i, 0) for i in range(4)]
    if sum(counts) > 0:
        cv = np.std(counts) / np.mean(counts)
        print(f"分布变异系数: {cv:.3f} (越小越均衡)")

        # 计算最大最小值比例
        max_count = max(counts)
        min_count = min(counts)
        if min_count > 0:
            ratio = max_count / min_count
            print(f"最大/最小类别比例: {ratio:.2f}")

    return class_counts


def suggest_thresholds(df, num_suggestions=5):
    """建议一些可能让分布更均衡的阈值组合"""
    print("\n" + "=" * 60)
    print("建议的阈值组合（基于百分位数）：")
    print("=" * 60)

    data = df['radius_ratio'].values

    # 基于百分位数的建议
    percentiles_options = [
        ([25, 50, 75], "25%, 50%, 75% 百分位数"),
        ([20, 40, 60], "20%, 40%, 60% 百分位数"),
        ([30, 60, 80], "30%, 60%, 80% 百分位数"),
        ([15, 35, 70], "15%, 35%, 70% 百分位数"),
        ([33, 66, 85], "33%, 66%, 85% 百分位数"),
    ]

    suggestions = []
    for percentiles, description in percentiles_options[:num_suggestions]:
        th = [np.percentile(data, p) for p in percentiles]
        suggestions.append((th, description))

        print(f"\n{description}:")
        print(f"阈值: [{th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f}]")

        # 显示这个阈值下的分布（使用临时最大阈值）
        temp_class_thresholds = {0: th[0], 1: th[1], 2: th[2], 3: np.max(data)}
        class_counts = analyze_distribution(df, th, temp_class_thresholds)


def plot_radius_distribution_histogram(df, thresholds=None, class_thresholds=None):
    """
    绘制动脉瘤半径尺寸分布的直方图和箱线图

    参数:
    df: 包含radius_ratio的数据框
    thresholds: 阈值列表，用于在图上标注分类边界
    class_thresholds: 各类别的最大阈值
    """
    fig = plt.figure(figsize=(16, 10))

    # 创建子图布局
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. 直方图（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(df['radius_ratio'], bins=30, edgecolor='black',
                                alpha=0.7, color='steelblue')

    # 添加阈值线
    if thresholds:
        colors_th = ['green', 'orange', 'red']
        for i, th in enumerate(thresholds):
            ax1.axvline(x=th, color=colors_th[i], linestyle='--', linewidth=2,
                        label=f'分类阈值{i + 1}: {th:.4f}')

    # 添加统计线
    mean_val = df['radius_ratio'].mean()
    median_val = df['radius_ratio'].median()
    ax1.axvline(x=mean_val, color='red', linestyle='-', linewidth=2, alpha=0.5, label=f'均值: {mean_val:.4f}')
    ax1.axvline(x=median_val, color='purple', linestyle='-', linewidth=2, alpha=0.5, label=f'中位数: {median_val:.4f}')

    ax1.set_xlabel('Radius Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('频数', fontsize=12, fontweight='bold')
    ax1.set_title('动脉瘤半径尺寸分布直方图', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 箱线图（右上）
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = df['radius_ratio'].values

    # 创建箱线图
    bp = ax2.boxplot([box_data], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)

    # 添加散点（显示所有数据点）
    x_jitter = np.random.normal(1, 0.04, size=len(box_data))
    ax2.scatter(x_jitter, box_data, alpha=0.3, color='blue', s=30, edgecolors='black', linewidth=0.5)

    # 标注统计信息
    q1 = np.percentile(box_data, 25)
    q3 = np.percentile(box_data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    stats_text = f'最小值: {np.min(box_data):.4f}\n'
    stats_text += f'Q1: {q1:.4f}\n'
    stats_text += f'中位数: {np.median(box_data):.4f}\n'
    stats_text += f'Q3: {q3:.4f}\n'
    stats_text += f'最大值: {np.max(box_data):.4f}\n'
    stats_text += f'IQR: {iqr:.4f}'

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_ylabel('Radius Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('动脉瘤半径尺寸箱线图', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels(['所有病例'])
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 累积分布图（左下）
    ax3 = fig.add_subplot(gs[1, 0])

    # 计算累积分布
    sorted_data = np.sort(df['radius_ratio'])
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    ax3.plot(sorted_data, cumulative, marker='.', linestyle='-', linewidth=2,
             markersize=3, color='darkblue')

    # 添加百分位数线
    percentiles = [25, 50, 75, 90]
    colors_perc = ['green', 'orange', 'red', 'purple']
    for p, color in zip(percentiles, colors_perc):
        val = np.percentile(sorted_data, p)
        ax3.axhline(y=p / 100, color=color, linestyle='--', alpha=0.5)
        ax3.axvline(x=val, color=color, linestyle='--', alpha=0.5)
        ax3.plot(val, p / 100, 'o', color=color, markersize=8)
        ax3.text(val, p / 100, f'  {p}%: {val:.4f}',
                 fontsize=9, verticalalignment='bottom')

    ax3.set_xlabel('Radius Ratio', fontsize=12, fontweight='bold')
    ax3.set_ylabel('累积概率', fontsize=12, fontweight='bold')
    ax3.set_title('动脉瘤半径尺寸累积分布图', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)

    # 4. 带分类标注的直方图（右下）
    ax4 = fig.add_subplot(gs[1, 1])

    if thresholds:
        # 根据阈值给数据着色
        bins_hist = 30
        n, bins, patches = ax4.hist(df['radius_ratio'], bins=bins_hist,
                                    edgecolor='black', alpha=0.3, color='gray')

        # 重新着色每个柱子基于其所属的类别
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for center, patch in zip(bin_centers, patches):
            if center < thresholds[0]:
                patch.set_facecolor('#ff9999')  # 极小
                patch.set_alpha(0.7)
            elif thresholds[0] <= center < thresholds[1]:
                patch.set_facecolor('#66b3ff')  # 小
                patch.set_alpha(0.7)
            elif thresholds[1] <= center <= thresholds[2]:
                patch.set_facecolor('#99ff99')  # 中等
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor('#ffcc99')  # 大
                patch.set_alpha(0.7)

        # 添加阈值线
        for i, th in enumerate(thresholds):
            ax4.axvline(x=th, color='black', linestyle='--', linewidth=2, alpha=0.7)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff9999', alpha=0.7, label=f'极小 (<{thresholds[0]})'),
            Patch(facecolor='#66b3ff', alpha=0.7, label=f'小 ({thresholds[0]}-{thresholds[1]})'),
            Patch(facecolor='#99ff99', alpha=0.7, label=f'中等 ({thresholds[1]}-{thresholds[2]})'),
            Patch(facecolor='#ffcc99', alpha=0.7, label=f'大 (>{thresholds[2]})')
        ]
        ax4.legend(handles=legend_elements, loc='upper right')

        title_text = f'按分类标注的尺寸分布\n阈值: [{thresholds[0]:.4f}, {thresholds[1]:.4f}, {thresholds[2]:.4f}]'
    else:
        ax4.hist(df['radius_ratio'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        title_text = '动脉瘤半径尺寸分布'

    ax4.set_xlabel('Radius Ratio', fontsize=12, fontweight='bold')
    ax4.set_ylabel('频数', fontsize=12, fontweight='bold')
    ax4.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('动脉瘤半径尺寸综合分析', fontsize=16, fontweight='bold', y=1.02)

    return fig


def print_detailed_statistics(df):
    """打印详细的统计信息"""
    print("\n" + "=" * 60)
    print("动脉瘤半径尺寸详细统计信息")
    print("=" * 60)

    data = df['radius_ratio'].values

    # 基本统计
    print(f"样本数量: {len(data)}")
    print(f"最小值: {np.min(data):.6f}")
    print(f"最大值: {np.max(data):.6f}")
    print(f"平均值: {np.mean(data):.6f}")
    print(f"中位数: {np.median(data):.6f}")
    print(f"标准差: {np.std(data):.6f}")
    print(f"方差: {np.var(data):.6f}")

    print("\n百分位数:")
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(data, p)
        print(f"  {p}%: {value:.6f}")

    print("\n分布形态:")
    try:
        from scipy import stats
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        print(f"偏度: {skewness:.4f} ", end="")
        if skewness > 0.5:
            print("(右偏分布)")
        elif skewness < -0.5:
            print("(左偏分布)")
        else:
            print("(近似对称分布)")

        print(f"峰度: {kurtosis:.4f} ", end="")
        if kurtosis > 0.5:
            print("(尖峰分布)")
        elif kurtosis < -0.5:
            print("(平峰分布)")
        else:
            print("(正态峰度)")
    except ImportError:
        print("提示: 安装 scipy 库可获取更详细的分布形态分析")

    # 四分位数和异常值检测
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    print(f"\n异常值检测 (基于IQR):")
    print(f"  下界: {lower_bound:.6f}")
    print(f"  上界: {upper_bound:.6f}")
    print(f"  异常值数量: {len(outliers)} ({len(outliers) / len(data) * 100:.1f}%)")
    if len(outliers) > 0:
        print(f"  异常值范围: [{np.min(outliers):.6f}, {np.max(outliers):.6f}]")

    print("=" * 60)


def display_threshold_settings(thresholds, class_thresholds):
    """显示当前的阈值设置"""
    print("\n" + "=" * 60)
    print("当前阈值设置")
    print("=" * 60)
    print(f"分类阈值 (用于分类边界):")
    print(f"  阈值1 (极小/小): {thresholds[0]:.4f}")
    print(f"  阈值2 (小/中等): {thresholds[1]:.4f}")
    print(f"  阈值3 (中等/大): {thresholds[2]:.4f}")
    print(f"\n各类别最大阈值 (用于表格第6列):")
    print(f"  类别0 (极小) 最大阈值: {class_thresholds[0]:.4f}")
    print(f"  类别1 (小) 最大阈值: {class_thresholds[1]:.4f}")
    print(f"  类别2 (中等) 最大阈值: {class_thresholds[2]:.4f}")
    print(f"  类别3 (大) 最大阈值: {class_thresholds[3]:.4f}")
    print("=" * 60)


def main():
    # 文件路径
    input_file = r"D:\med_data\ai\translate\all_mask\location_all.xlsx"
    output_file = r"D:\med_data\ai\translate\all_mask\location_size.xlsx"

    try:
        # 读取Excel文件
        print("正在读取文件...")
        df = pd.read_excel(input_file, usecols=[0, 1, 2, 3])
        df.columns = ['filename', 'x_ratio', 'y_ratio', 'radius_ratio']

        print(f"成功读取数据，共 {len(df)} 条记录")
        print("\nradius_ratio数据分布情况：")
        print("=" * 50)
        print(df['radius_ratio'].describe())
        print("=" * 50)

        # 打印详细统计信息
        print_detailed_statistics(df)

        # ========== 在这里设置您想要的阈值 ==========
        # 1. 分类阈值（用于确定类别边界）
        thresholds = [0.014, 0.022, 0.036]  # 新的分类阈值

        # 2. 各类别的最大阈值（用于表格第6列）
        class_thresholds = {
            0: 0.014,  # 极小的最大阈值
            1: 0.022,  # 小的最大阈值
            2: 0.036,  # 中等的最大阈值
            3: 0.14  # 大的最大阈值
        }

        # 其他可选的阈值设置示例（取消注释即可使用）：
        # thresholds = [0.02, 0.04, 0.1]  # 原始阈值
        # class_thresholds = {0: 0.02, 1: 0.04, 2: 0.1, 3: 0.2}

        # thresholds = [0.015, 0.035, 0.08]
        # class_thresholds = {0: 0.015, 1: 0.035, 2: 0.08, 3: 0.15}

        # 显示当前阈值设置
        display_threshold_settings(thresholds, class_thresholds)

        print("\n如需修改阈值，请在代码的 main() 函数中调整 thresholds 和 class_thresholds 变量的值")
        print("=" * 60)

        # 分析当前阈值下的分布
        print("\n当前阈值下的分布分析：")
        class_counts = analyze_distribution(df, thresholds, class_thresholds)

        # 提供阈值建议（可选功能）
        print("\n是否需要查看阈值建议？这将帮助您找到更均衡的分类阈值。")
        response = input("输入 'y' 查看建议，或直接按回车跳过: ").strip().lower()

        if response == 'y':
            suggest_thresholds(df)

            print("\n" + "=" * 60)
            print("您可以根据上述建议，修改代码中的 thresholds 变量值")
            print("然后重新运行程序以获得更均衡的分类")
            print("=" * 60)

            # 询问是否继续使用当前阈值
            continue_current = input(f"\n是否继续使用当前阈值 {thresholds} 进行处理？(y/n): ")
            if continue_current.lower() != 'y':
                print("程序终止。请修改阈值后重新运行。")
                return

        # 应用最终选择的阈值进行分类
        df['size_classes'] = df['radius_ratio'].apply(lambda x: classify_size(x, thresholds))

        # 添加最大阈值列
        df['max_threshold'] = df['size_classes'].apply(lambda x: class_thresholds[x])

        # 创建分类标签（用于显示）
        size_labels_display = {
            0: f'极小 (<{thresholds[0]}, 最大阈值:{class_thresholds[0]:.3f})',
            1: f'小 ({thresholds[0]}-{thresholds[1]}, 最大阈值:{class_thresholds[1]:.3f})',
            2: f'中等 ({thresholds[1]}-{thresholds[2]}, 最大阈值:{class_thresholds[2]:.3f})',
            3: f'大 (>{thresholds[2]}, 最大阈值:{class_thresholds[3]:.3f})'
        }

        # 保存到新Excel文件
        print(f"\n正在保存到文件: {output_file}")
        df.to_excel(output_file, index=False)
        print("文件保存成功！")

        # 显示前几行数据示例
        print("\n保存的数据示例（前10行）：")
        print(df.head(10).to_string())

        # 绘制初始的尺寸分布直方图（无阈值分类）
        print("\n正在生成初始尺寸分布图...")
        fig_initial = plot_radius_distribution_histogram(df)
        plt.show()

        # 生成带阈值标注的尺寸分布图
        print("\n正在生成带分类标注的尺寸分布图...")
        fig_classified = plot_radius_distribution_histogram(df, thresholds, class_thresholds)

        # 保存分布图
        hist_path = Path(output_file).parent / "radius_distribution_histogram.png"
        fig_classified.savefig(hist_path, dpi=300, bbox_inches='tight')
        print(f"尺寸分布图已保存到: {hist_path}")

        # 生成分类饼状图
        print("\n正在生成分类饼状图...")
        fig_pie, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 准备绘图数据
        class_counts_final = df['size_classes'].value_counts().sort_index()
        labels = [size_labels_display[i] for i in class_counts_final.index]
        sizes = class_counts_final.values
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        explode = (0.05, 0.05, 0.05, 0.05)

        # 第一个图：带百分比的饼图
        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')

        ax1.set_title(f'动脉瘤尺寸分类分布\n分类阈值: [{thresholds[0]}, {thresholds[1]}, {thresholds[2]}]',
                      fontsize=14, fontweight='bold', pad=20)

        # 第二个图：带数量的饼图
        labels_with_count = [f'{size_labels_display[i]}\n({class_counts_final[i]}例)'
                             for i in class_counts_final.index]

        wedges2, texts2, autotexts2 = ax2.pie(sizes, explode=explode,
                                              labels=labels_with_count,
                                              colors=colors, autopct='%1.1f%%',
                                              shadow=True, startangle=90)

        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        for text in texts2:
            text.set_fontsize(10)
            text.set_fontweight('bold')

        ax2.set_title('动脉瘤尺寸分类分布（含数量）', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('动脉瘤尺寸分类分析报告', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存饼图
        pie_path = Path(output_file).parent / "size_classification_pie.png"
        plt.savefig(pie_path, dpi=300, bbox_inches='tight')
        print(f"分类饼状图已保存到: {pie_path}")

        plt.show()

        # 打印最终统计信息
        print("\n最终统计信息：")
        print("=" * 60)
        print(f"使用的分类阈值: [{thresholds[0]}, {thresholds[1]}, {thresholds[2]}]")
        print(f"各类别最大阈值:")
        for class_id in range(4):
            print(f"  类别{class_id}: {class_thresholds[class_id]:.4f}")
        print(f"\n总病例数: {len(df)}")
        print(f"radius_ratio 范围: [{df['radius_ratio'].min():.4f}, {df['radius_ratio'].max():.4f}]")
        print("=" * 60)

        # 打印各类别统计
        print("\n各类别详细统计:")
        for class_id in sorted(class_counts_final.index):
            count = class_counts_final[class_id]
            percentage = count / len(df) * 100
            print(f"类别 {class_id} ({size_labels_display[class_id]}): {count} 例 ({percentage:.1f}%)")

        # 统计最大阈值列的信息
        print(f"\n最大阈值列统计:")
        print(f"  唯一值: {df['max_threshold'].unique()}")
        print(f"  值计数:")
        for th, count in df['max_threshold'].value_counts().sort_index().items():
            print(f"    {th:.4f}: {count} 例")

        print("\n所有图表和分析完成！")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"发生错误：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()