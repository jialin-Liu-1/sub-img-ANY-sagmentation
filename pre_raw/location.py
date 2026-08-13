import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import warnings
import logging

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaskLocationExtractor:
    """从mask图像中提取动脉瘤位置信息（使用最小外接圆）"""

    def __init__(self, mask_dir, output_excel, visualization_dir):
        self.mask_dir = Path(mask_dir)
        self.output_excel = Path(output_excel)
        self.visualization_dir = Path(visualization_dir)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.location_data = []

    def find_mask_files(self):
        """查找所有TIF格式的mask文件"""
        mask_files = sorted(list(self.mask_dir.glob("*.tif")))
        logger.info(f"找到 {len(mask_files)} 个TIF mask文件")
        return mask_files

    def load_mask(self, mask_path):
        """加载mask图像并二值化"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"无法读取: {mask_path.name}")
            return None

        # 二值化
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return (mask_binary > 127).astype(np.uint8)

    def find_main_aneurysm_region(self, mask):
        """找到最大的动脉瘤区域"""
        labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)
        if num_labels == 0:
            return None

        regions = regionprops(labeled_mask)
        if not regions:
            return None

        # 选择最大区域
        largest_region = max(regions, key=lambda r: r.area)
        return largest_region if largest_region.area >= 10 else None

    def calculate_minimum_enclosing_circle(self, region):
        """计算最小外接圆"""
        coords = region.coords  # [y, x]格式

        if len(coords) < 3:
            # 点数太少，使用边界框
            min_row, min_col, max_row, max_col = region.bbox
            center_x = (min_col + max_col) / 2
            center_y = (min_row + max_row) / 2
            radius = max(max_row - min_row, max_col - min_col) / 2
            return center_y, center_x, radius

        # 使用OpenCV的minEnclosingCircle
        points = coords[:, [1, 0]].astype(np.float32)  # 转换为[x, y]格式
        (center_x, center_y), radius = cv2.minEnclosingCircle(points)
        radius *= 1.1  # 增加10%余量确保完全覆盖

        return center_y, center_x, radius

    def create_circle_mask(self, shape, center_y, center_x, radius):
        """创建圆形mask"""
        h, w = shape
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        return (dist <= radius).astype(np.uint8)

    def check_coverage_and_adjust(self, original_mask, center_y, center_x, radius):
        """检查覆盖率并调整半径"""
        h, w = original_mask.shape
        initial_circle = self.create_circle_mask((h, w), center_y, center_x, radius)

        aneurysm_area = np.sum(original_mask > 0)
        overlap_area = np.sum(np.logical_and(original_mask > 0, initial_circle > 0))
        coverage = overlap_area / aneurysm_area if aneurysm_area > 0 else 0

        if coverage > 0.95:
            return center_y, center_x, radius, coverage

        # 逐步增加半径直到覆盖率达到99%
        adjusted_radius = radius
        max_radius = min(h, w) / 2

        for _ in range(20):
            if adjusted_radius >= max_radius:
                break

            adjusted_radius *= 1.05
            new_circle = self.create_circle_mask((h, w), center_y, center_x, adjusted_radius)
            new_overlap = np.sum(np.logical_and(original_mask > 0, new_circle > 0))
            new_coverage = new_overlap / aneurysm_area if aneurysm_area > 0 else 0

            if new_coverage > 0.99:
                return center_y, center_x, adjusted_radius, new_coverage

        return center_y, center_x, adjusted_radius, coverage

    def visualize_single_overlap(self, original_mask, circle_mask, filename, center_x, center_y, radius):
        """生成单个重叠可视化图像（圆心和动脉瘤中心重叠）"""
        try:
            # 创建重叠图像
            overlap = np.zeros((*original_mask.shape, 3), dtype=np.uint8)

            # 动脉瘤区域：绿色
            aneurysm_mask = original_mask > 0
            overlap[aneurysm_mask, 1] = 255

            # 圆形区域：红色（半透明）
            circle_area = circle_mask > 0
            overlap[circle_area, 0] = 128

            # 重叠区域：黄色
            overlap_area = np.logical_and(aneurysm_mask, circle_area)
            overlap[overlap_area, 0] = 255
            overlap[overlap_area, 1] = 255

            # 绘制圆心
            center_pt = (int(center_x), int(center_y))
            cv2.circle(overlap, center_pt, 5, (0, 0, 255), -1)  # 红色圆心
            cv2.circle(overlap, center_pt, int(radius), (255, 255, 255), 2)  # 白色圆边界

            # 保存图像
            output_name = filename.replace('.tif', '_overlap.png')
            output_path = self.visualization_dir / output_name
            cv2.imwrite(str(output_path), cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR))

            return True
        except Exception as e:
            logger.error(f"保存可视化失败: {e}")
            return False

    def process_single_mask(self, mask_path):
        """处理单个mask文件"""
        filename = mask_path.name
        logger.info(f"处理: {filename}")

        # 加载mask
        mask = self.load_mask(mask_path)
        if mask is None:
            return None

        h, w = mask.shape

        # 找到主要动脉瘤区域
        region = self.find_main_aneurysm_region(mask)
        if region is None:
            logger.warning(f"  {filename}: 未找到有效动脉瘤区域")
            return None

        # 计算最小外接圆
        center_y, center_x, radius = self.calculate_minimum_enclosing_circle(region)

        # 调整半径确保完全覆盖
        center_y, center_x, final_radius, coverage = self.check_coverage_and_adjust(
            mask, center_y, center_x, radius
        )

        # 计算归一化参数
        image_diagonal = np.sqrt(h ** 2 + w ** 2)
        height_ratio = max(0.0, min(1.0, center_y / h))
        width_ratio = max(0.0, min(1.0, center_x / w))
        radius_ratio = max(0.0, min(1.0, final_radius / image_diagonal))

        # 创建圆形mask并生成可视化 - 传入final_radius参数
        circle_mask = self.create_circle_mask((h, w), center_y, center_x, final_radius)
        self.visualize_single_overlap(mask, circle_mask, filename, center_x, center_y, final_radius)

        # 打印坐标信息
        logger.info(f"  中心坐标: ({center_x:.1f}, {center_y:.1f})")
        logger.info(f"  像素半径: {final_radius:.1f}")
        logger.info(f"  X轴比率(0=左,1=右): {width_ratio:.4f}")
        logger.info(f"  Y轴比率(0=上,1=下): {height_ratio:.4f}")
        logger.info(f"  半径比率: {radius_ratio:.4f}")
        logger.info(f"  覆盖率: {coverage:.1%}")

        # 返回结果
        return {
            'filename': filename.replace('.tif', ''),
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'radius_ratio': radius_ratio,
            'center_x': center_x,
            'center_y': center_y,
            'pixel_radius': final_radius,
            'image_width': w,
            'image_height': h,
            'coverage': coverage
        }

    def process_all_masks(self):
        """处理所有mask文件"""
        logger.info("开始提取动脉瘤位置信息...")

        mask_files = self.find_mask_files()
        if not mask_files:
            logger.error("未找到mask文件")
            return False

        successful_count = 0
        for mask_path in mask_files:
            result = self.process_single_mask(mask_path)
            if result:
                self.location_data.append(result)
                successful_count += 1

        logger.info(f"处理完成: 成功 {successful_count}/{len(mask_files)}")
        return successful_count > 0

    def save_to_excel(self):
        """保存结果到Excel"""
        if not self.location_data:
            logger.error("没有数据可保存")
            return False

        df = pd.DataFrame(self.location_data)
        output_df = df[['filename', 'width_ratio', 'height_ratio', 'radius_ratio',
                        'center_x', 'center_y', 'pixel_radius',
                        'image_width', 'image_height', 'coverage']].copy()
        output_df.columns = ['filename', 'x_ratio(0=left,1=right)', 'y_ratio(0=top,1=bottom)',
                             'radius_ratio', 'center_x(pixels)', 'center_y(pixels)',
                             'radius(pixels)', 'image_width', 'image_height', 'coverage']

        output_df.to_excel(self.output_excel, index=False)
        logger.info(f"位置信息已保存: {self.output_excel}")
        logger.info(f"总记录数: {len(output_df)}")

        return True


def main():
    """主函数 - 所有路径在此配置"""

    # ========== 配置参数 ==========

    # Mask文件目录
    mask_dir = r"D:\med_data\multi\min_mask"

    # 输出Excel表格路径
    output_excel = r"D:\med_data\multi\min_location.xlsx"

    # 可视化输出目录（只保存重叠PNG图像）
    visualization_dir = r"D:\med_data\multi\min_location"

    # ========== 开始处理 ==========

    logger.info("=" * 60)
    logger.info("动脉瘤位置信息提取")
    logger.info("=" * 60)
    logger.info(f"Mask目录: {mask_dir}")
    logger.info(f"输出表格: {output_excel}")
    logger.info(f"可视化目录: {visualization_dir}")

    # 创建提取器并处理
    extractor = MaskLocationExtractor(mask_dir, output_excel, visualization_dir)

    try:
        if extractor.process_all_masks():
            extractor.save_to_excel()

            # 输出统计摘要
            df = pd.DataFrame(extractor.location_data)
            if len(df) > 0:
                logger.info("=" * 60)
                logger.info(f"统计摘要 (共{len(df)}个样本):")
                logger.info(f"  平均X轴比率: {df['width_ratio'].mean():.4f} ± {df['width_ratio'].std():.4f}")
                logger.info(f"  平均Y轴比率: {df['height_ratio'].mean():.4f} ± {df['height_ratio'].std():.4f}")
                logger.info(f"  平均半径比率: {df['radius_ratio'].mean():.4f} ± {df['radius_ratio'].std():.4f}")
                logger.info(f"  平均覆盖率: {df['coverage'].mean():.1%}")
                logger.info(f"  平均中心坐标: ({df['center_x'].mean():.1f}, {df['center_y'].mean():.1f})")
                logger.info("=" * 60)

        logger.info("处理完成!")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()