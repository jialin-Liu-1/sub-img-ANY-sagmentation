import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import matplotlib

# 设置matplotlib使用非交互式后端
matplotlib.use('Agg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入血管提取器
from multi.ves_block import EnhancedVasculatureExtractor, LayerScanVasculatureExtractor, EnhancedLayerScanVasculatureExtractor


class EnhancedVesselVisualizer:
    """增强版血管提取可视化处理器"""

    def __init__(self):
        # 输入输出路径
        self.input_dir = Path(r"D:\med_data\ai\ce")
        self.output_dir = Path(r"D:\med_data\ai\ce4")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 提取参数
        self.vessel_threshold = 0.5
        self.min_diameter_ratio = 0.4

        # 可视化参数
        self.fig_size = (15, 10)
        self.dpi = 150

        """# 创建血管提取器
        self.extractor = EnhancedVasculatureExtractor(
    vessel_threshold=0.5,
    min_vessel_diameter_ratio=0.3,
    min_vessel_length=100,
    search_radius=5,
    local_window_size=5,
    angle_tolerance=70,
    # 新增参数
    diameter_continuity_weight=0.3,  # 调高此值增强主血管选择
    backtracking_penalty=10.0,       # 调高此值防止折返
    center_correction_window=7,      # 增大窗口提高中心校正效果
    multi_scale_centrality=True      # 启用多尺度中心性
)
        """
        # 替换原有的提取器
        self.extractor = EnhancedLayerScanVasculatureExtractor(
            vessel_threshold=0.5,
            min_scan_length=10,
            gap_tolerance=3,
            center_smooth_window=5,
            continuity_threshold=0.6,
            backward_threshold=0.2,  # 20%图像高度
            width_stability_threshold=0.4
        )
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"血管阈值: {self.vessel_threshold}")
        print(f"最小直径比例: {self.min_diameter_ratio}")

    def find_dicom_files(self):
        """查找所有DICOM文件"""
        files = []
        for file_path in self.input_dir.glob("*"):
            if file_path.is_file() and not file_path.suffix:
                try:
                    pydicom.dcmread(file_path, force=True)
                    files.append(file_path)
                except:
                    continue
        return files

    def read_dicom_image(self, file_path):
        """读取并归一化DICOM图像"""
        dicom_data = pydicom.dcmread(file_path, force=True)
        image = dicom_data.pixel_array.astype(np.float32)

        # 归一化到0-1范围
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)

        return image

    def create_comprehensive_visualization(self, image, vessel_info, filename):
        """
        创建综合可视化图像，包含：
        1. 原始图像 + 二值化血管掩模
        2. 血管中心线叠加
        3. 直径变化曲线
        4. 中心线质量评估
        5. 提取参数信息
        """
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)

        # 创建2x3的子图布局
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. 原始图像 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image, cmap='gray')
        ax1.set_title('原始DSA图像', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. 二值化血管掩模 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        binary_mask = (image < self.vessel_threshold).astype(np.uint8)
        ax2.imshow(binary_mask, cmap='gray')
        ax2.set_title(f'二值化血管掩模\n(阈值={self.vessel_threshold})', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 统计信息
        vessel_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.size
        vessel_ratio = vessel_pixels / total_pixels * 100
        ax2.text(0.02, 0.98, f'血管像素: {vessel_pixels:,}\n占比: {vessel_ratio:.1f}%',
                 transform=ax2.transAxes, color='white', fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # 3. 中心线叠加 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image, cmap='gray', alpha=0.7)

        if vessel_info['centerline_points'] is not None:
            points = vessel_info['centerline_points']

            # 绘制中心线
            ax3.scatter(points[:, 1], points[:, 0],
                        s=20, c='red', alpha=0.8, label='中心线点')

            # 连接中心线点
            if len(points) > 1:
                ax3.plot(points[:, 1], points[:, 0],
                         'y-', linewidth=1.5, alpha=0.6, label='中心线')

            # 标记起点和终点
            if 'start_point' in vessel_info and vessel_info['start_point'] is not None:
                start = vessel_info['start_point']
                ax3.scatter(start[1], start[0],
                            s=200, c='green', marker='o',
                            edgecolors='white', linewidth=2, label='起点')

            if 'end_point' in vessel_info and vessel_info['end_point'] is not None:
                end = vessel_info['end_point']
                ax3.scatter(end[1], end[0],
                            s=200, c='blue', marker='x',
                            linewidth=3, label='终点')

            ax3.legend(loc='upper right', fontsize=9)

            # 中心线信息
            centerline_length = len(points)
            if len(points) > 1:
                # 计算中心线总长度
                total_length = 0
                for i in range(1, len(points)):
                    dist = np.linalg.norm(points[i] - points[i - 1])
                    total_length += dist

                ax3.text(0.02, 0.98, f'中心线点数: {centerline_length}\n总长度: {total_length:.1f} px',
                         transform=ax3.transAxes, color='white', fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax3.set_title('血管中心线提取', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 4. 直径变化曲线 (左下)
        ax4 = fig.add_subplot(gs[1, 0])
        if vessel_info['diameters'] is not None and len(vessel_info['diameters']) > 0:
            diameters = vessel_info['diameters']

            # 计算沿中心线的距离
            if 'centerline_points' in vessel_info and vessel_info['centerline_points'] is not None:
                points = vessel_info['centerline_points']
                if len(points) > 1:
                    distances = [0]
                    for i in range(1, len(points)):
                        dist = np.linalg.norm(points[i] - points[i - 1])
                        distances.append(distances[-1] + dist)

                    ax4.plot(distances, diameters, 'b-', linewidth=2, label='血管直径')

                    # 标记最大直径
                    max_diameter = np.max(diameters)
                    max_idx = np.argmax(diameters)
                    ax4.scatter(distances[max_idx], max_diameter,
                                s=100, c='red', marker='o', label=f'最大直径: {max_diameter:.1f}px')

                    # 添加阈值线
                    if 'max_diameter' in vessel_info:
                        threshold_diameter = vessel_info['max_diameter'] * self.min_diameter_ratio
                        ax4.axhline(y=threshold_diameter, color='r', linestyle='--',
                                    alpha=0.7, label=f'停止阈值: {threshold_diameter:.1f}px')

                    ax4.set_xlabel('沿中心线距离 (像素)', fontsize=11)
                    ax4.set_ylabel('血管直径 (像素)', fontsize=11)
                    ax4.set_title('血管直径变化曲线', fontsize=12, fontweight='bold')
                    ax4.legend(fontsize=9)
                    ax4.grid(True, alpha=0.3)

            # 直径统计
            if len(diameters) > 0:
                diameter_stats = f'平均直径: {np.mean(diameters):.1f} px\n'
                diameter_stats += f'最小直径: {np.min(diameters):.1f} px\n'
                diameter_stats += f'最大直径: {np.max(diameters):.1f} px\n'
                diameter_stats += f'直径标准差: {np.std(diameters):.1f} px'

                ax4.text(0.02, 0.98, diameter_stats,
                         transform=ax4.transAxes, fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax4.set_title('血管直径变化曲线', fontsize=12, fontweight='bold')

        # 5. 中心线质量评估 (中下)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')

        quality_text = "中心线质量评估:\n"
        quality_text += "=" * 30 + "\n"

        if vessel_info['centerline_points'] is not None:
            points = vessel_info['centerline_points']
            num_points = len(points)

            # 评估指标
            quality_score = 0
            max_score = 5

            # 1. 中心线长度
            if num_points >= 50:
                quality_text += "✓ 中心线长度: 优秀\n"
                quality_score += 1
            elif num_points >= 30:
                quality_text += "✓ 中心线长度: 良好\n"
                quality_score += 0.7
            elif num_points >= 10:
                quality_text += "⚠ 中心线长度: 一般\n"
                quality_score += 0.3
            else:
                quality_text += "✗ 中心线长度: 不足\n"

            # 2. 直径连续性
            if 'diameters' in vessel_info and vessel_info['diameters'] is not None:
                diameters = vessel_info['diameters']
                if len(diameters) > 1:
                    diameter_changes = np.abs(np.diff(diameters))
                    avg_change = np.mean(diameter_changes)

                    if avg_change < 2.0:
                        quality_text += "✓ 直径连续性: 优秀\n"
                        quality_score += 1
                    elif avg_change < 5.0:
                        quality_text += "✓ 直径连续性: 良好\n"
                        quality_score += 0.7
                    else:
                        quality_text += "⚠ 直径连续性: 不稳定\n"
                        quality_score += 0.3

            # 3. 停止原因分析
            if 'stopping_reason' in vessel_info:
                reason = vessel_info['stopping_reason']
                quality_text += f"停止原因: {reason}\n"

                if 'diameter_too_small' in reason:
                    quality_text += "⚠ 因直径过小而停止\n"
                elif 'reached_min_length' in reason:
                    quality_text += "✓ 达到最小长度要求\n"
                    quality_score += 1
                elif 'reached_top_boundary' in reason:
                    quality_text += "✓ 追踪到图像边界\n"
                    quality_score += 1

            # 4. 中心线平滑度
            if len(points) > 2:
                angles = []
                for i in range(1, len(points) - 1):
                    v1 = points[i] - points[i - 1]
                    v2 = points[i + 1] - points[i]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        v1_norm = v1 / np.linalg.norm(v1)
                        v2_norm = v2 / np.linalg.norm(v2)
                        dot = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                        angle = np.degrees(np.arccos(dot))
                        angles.append(angle)

                if angles:
                    avg_angle = np.mean(angles)
                    if avg_angle < 20:
                        quality_text += f"✓ 中心线平滑度: 优秀 ({avg_angle:.1f}°)\n"
                        quality_score += 1
                    elif avg_angle < 40:
                        quality_text += f"✓ 中心线平滑度: 良好 ({avg_angle:.1f}°)\n"
                        quality_score += 0.7
                    else:
                        quality_text += f"⚠ 中心线平滑度: 一般 ({avg_angle:.1f}°)\n"
                        quality_score += 0.3

            # 总体质量评分
            quality_percentage = (quality_score / max_score) * 100
            quality_text += "\n" + "=" * 30 + "\n"
            quality_text += f"总体质量评分: {quality_percentage:.0f}%\n"

            if quality_percentage >= 80:
                quality_text += "评级: 优秀 ★★★"
            elif quality_percentage >= 60:
                quality_text += "评级: 良好 ★★"
            elif quality_percentage >= 40:
                quality_text += "评级: 一般 ★"
            else:
                quality_text += "评级: 需要改进"

        ax5.text(0.05, 0.95, quality_text, fontsize=10,
                 verticalalignment='top', transform=ax5.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax5.set_title('中心线质量评估', fontsize=12, fontweight='bold')

        # 6. 提取参数和统计信息 (右下)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        info_text = "提取参数和统计信息:\n"
        info_text += "=" * 30 + "\n\n"

        # 参数信息
        info_text += "提取参数:\n"
        info_text += f"• 血管阈值: {self.vessel_threshold}\n"
        info_text += f"• 最小直径比例: {self.min_diameter_ratio}\n"

        # 统计信息
        if vessel_info['centerline_points'] is not None:
            points = vessel_info['centerline_points']
            info_text += "提取统计:\n"
            info_text += f"• 中心线点数: {len(points)}\n"

            if 'max_diameter' in vessel_info:
                info_text += f"• 最大血管直径: {vessel_info['max_diameter']:.1f} px\n"

            if 'diameters' in vessel_info and vessel_info['diameters'] is not None:
                diameters = vessel_info['diameters']
                if len(diameters) > 0:
                    info_text += f"• 平均血管直径: {np.mean(diameters):.1f} px\n"
                    info_text += f"• 直径变化范围: [{np.min(diameters):.1f}, {np.max(diameters):.1f}] px\n"

            if 'stopping_reason' in vessel_info:
                info_text += f"• 停止原因: {vessel_info['stopping_reason']}\n"

        # 图像信息
        info_text += "\n图像信息:\n"
        info_text += f"• 图像尺寸: {image.shape}\n"
        info_text += f"• 文件名: {filename}\n"

        ax6.text(0.05, 0.95, info_text, fontsize=10,
                 verticalalignment='top', transform=ax6.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax6.set_title('参数和统计信息', fontsize=12, fontweight='bold')

        # 添加总标题
        plt.suptitle(f'血管提取和中心线生成质量评估 - {filename}',
                     fontsize=14, fontweight='bold', y=0.98)

        # 保存图像
        output_path = self.output_dir / f"{filename}_visualization.png"
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def process_single_file(self, file_path):
        """处理单个文件"""
        filename = file_path.stem if file_path.suffix else file_path.name

        try:
            print(f"处理文件: {filename}")

            # 读取图像
            image = self.read_dicom_image(file_path)
            print(f"  图像尺寸: {image.shape}")

            # 提取血管信息
            print("  正在提取血管...")
            vessel_info = self.extractor.extract_main_vasculature(image)

            # 创建综合可视化
            print("  生成可视化图像...")
            output_path = self.create_comprehensive_visualization(image, vessel_info, filename)

            print(f"  ✓ 成功保存: {output_path.name}")

            # 保存原始数据用于进一步分析
            data_path = self.output_dir / f"{filename}_data.npz"
            np.savez_compressed(
                str(data_path),
                image=image,
                centerline_points=vessel_info.get('centerline_points'),
                diameters=vessel_info.get('diameters'),
                vessel_info=vessel_info
            )

            return True, vessel_info

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None

    def run(self):
        """运行处理器"""
        print("=" * 60)
        print("开始血管提取和中心线生成质量评估...")
        print("=" * 60)

        # 查找所有文件
        files = self.find_dicom_files()

        if not files:
            print("未找到DICOM文件")
            return

        print(f"找到 {len(files)} 个DICOM文件")
        print("=" * 60)

        # 处理统计
        success_count = 0
        quality_scores = []

        # 处理每个文件
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            success, vessel_info = self.process_single_file(file_path)

            if success:
                success_count += 1

                # 计算质量分数
                if vessel_info and vessel_info.get('centerline_points') is not None:
                    points = vessel_info['centerline_points']
                    num_points = len(points)

                    # 简单质量评分：基于中心线长度
                    if num_points >= 50:
                        quality_scores.append(90)
                    elif num_points >= 30:
                        quality_scores.append(70)
                    elif num_points >= 10:
                        quality_scores.append(50)
                    else:
                        quality_scores.append(30)

        # 生成总体报告
        self.generate_summary_report(success_count, len(files), quality_scores)

        print("=" * 60)
        print("处理完成!")
        print(f"总文件数: {len(files)}")
        print(f"成功处理: {success_count}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)

    def generate_summary_report(self, success_count, total_files, quality_scores):
        """生成总体质量报告"""
        if not quality_scores:
            return

        report_path = self.output_dir / "quality_summary_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("血管提取质量评估报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"处理时间: {os.path.basename(self.output_dir)}\n")
            f.write(f"总文件数: {total_files}\n")
            f.write(f"成功处理: {success_count}\n")
            f.write(f"成功率: {success_count / total_files * 100:.1f}%\n\n")

            # 质量统计
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                std_quality = np.std(quality_scores)
                max_quality = np.max(quality_scores)
                min_quality = np.min(quality_scores)

                f.write("质量评分统计:\n")
                f.write("-" * 30 + "\n")
                f.write(f"平均质量: {avg_quality:.1f}%\n")
                f.write(f"质量标准差: {std_quality:.1f}%\n")
                f.write(f"最高质量: {max_quality:.1f}%\n")
                f.write(f"最低质量: {min_quality:.1f}%\n\n")

                # 质量分布
                f.write("质量分布:\n")
                f.write("-" * 30 + "\n")

                bins = [(90, 100, "优秀"), (70, 89, "良好"),
                        (50, 69, "一般"), (0, 49, "需要改进")]

                for min_val, max_val, label in bins:
                    count = sum(1 for score in quality_scores if min_val <= score <= max_val)
                    percentage = count / len(quality_scores) * 100
                    f.write(f"{label} ({min_val}-{max_val}%): {count} 个 ({percentage:.1f}%)\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("提取参数:\n")
            f.write("-" * 30 + "\n")
            f.write(f"血管阈值: {self.vessel_threshold}\n")
            f.write(f"最小直径比例: {self.min_diameter_ratio}\n")

        print(f"质量报告已保存到: {report_path}")


def main():
    """主函数"""
    visualizer = EnhancedVesselVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()