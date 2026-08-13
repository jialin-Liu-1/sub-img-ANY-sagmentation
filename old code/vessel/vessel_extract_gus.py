import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import logging
from tqdm import tqdm
from scipy import ndimage
from skimage import morphology
from scipy.ndimage import gaussian_filter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DSAAnalyzer:
    """DSA图像分析器 - 处理多帧DICOM文件，只标记像素值下降的区域，并进行形态学后处理"""

    def __init__(self, excel_path, image_base_dir, output_dir):
        self.excel_path = excel_path
        self.image_base_dir = image_base_dir
        self.output_dir = output_dir
        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'skipped_missing': 0,
            'sequences_found': 0,
            'sequences_processed': 0
        }

    def read_excel_files(self):
        """读取Excel文件中的文件名列表"""
        try:
            df = pd.read_excel(self.excel_path)
            file_names = df.iloc[:, 0].dropna().tolist()
            logger.info(f"从Excel读取到 {len(file_names)} 个文件名")
            return file_names
        except Exception as e:
            logger.error(f"读取Excel文件失败: {e}")
            return []

    def find_sequence_files(self, file_name):
        """
        查找指定文件名的两个DSA序列文件
        返回: dict {'0': 文件路径, '1': 文件路径}
        """
        sequences = {'0': None, '1': None}

        if not os.path.exists(self.image_base_dir):
            logger.warning(f"图像目录不存在: {self.image_base_dir}")
            return sequences

        # 构建两个可能的文件名
        file_0 = f"{file_name}_0"
        file_1 = f"{file_name}_1"

        file_0_path = os.path.join(self.image_base_dir, file_0)
        file_1_path = os.path.join(self.image_base_dir, file_1)

        # 检查文件是否存在
        if os.path.exists(file_0_path):
            sequences['0'] = file_0_path
            logger.info(f"  - 找到序列0文件: {file_0}")
        else:
            logger.warning(f"  - 未找到序列0文件: {file_0}")

        if os.path.exists(file_1_path):
            sequences['1'] = file_1_path
            logger.info(f"  - 找到序列1文件: {file_1}")
        else:
            logger.warning(f"  - 未找到序列1文件: {file_1}")

        return sequences

    def read_multiframe_dicom(self, file_path):
        """
        读取多帧DICOM文件并返回所有帧的像素数组
        """
        try:
            dicom_data = pydicom.dcmread(file_path)

            # 检查是否为多帧图像
            if hasattr(dicom_data, 'NumberOfFrames'):
                num_frames = int(dicom_data.NumberOfFrames)
                logger.info(f"    - DICOM文件包含 {num_frames} 帧")
            else:
                # 如果没有NumberOfFrames属性，尝试从像素数组维度判断
                if len(dicom_data.pixel_array.shape) == 3:
                    num_frames = dicom_data.pixel_array.shape[0]
                else:
                    num_frames = 1

            # 获取像素数组
            pixel_array = dicom_data.pixel_array

            # 处理像素数组
            if len(pixel_array.shape) == 3:
                # 已经是多帧格式 [frames, height, width]
                frames = pixel_array.astype(np.float32)
            elif len(pixel_array.shape) == 2:
                # 单帧图像，包装成多帧格式
                frames = np.expand_dims(pixel_array, axis=0).astype(np.float32)
            else:
                logger.error(f"    - 不支持的图像维度: {pixel_array.shape}")
                return None

            logger.info(f"    - 成功读取 {frames.shape[0]} 帧, 每帧大小: {frames.shape[1]}x{frames.shape[2]}")
            return frames

        except Exception as e:
            logger.error(f"读取多帧DICOM文件失败 {file_path}: {e}")
            return None

    def apply_gaussian_filter(self, image_frames, sigma=1.0):
        """
        对图像序列应用高斯滤波

        Args:
            image_frames: 多帧图像数组 [frames, height, width]
            sigma: 高斯滤波的标准差

        Returns:
            filtered_frames: 滤波后的图像数组
        """
        logger.info(f"    - 应用高斯滤波: sigma={sigma}")

        filtered_frames = np.zeros_like(image_frames)

        for i in range(len(image_frames)):
            # 对每一帧应用高斯滤波
            filtered_frames[i] = gaussian_filter(image_frames[i], sigma=sigma)

        logger.info(f"    - 高斯滤波完成")

        return filtered_frames

    def morphological_postprocessing(self, binary_mask, closing_iterations=1, kernel_size=3):
        """
        对二值掩膜进行形态学后处理

        Args:
            binary_mask: 输入的二值掩膜 (0和1组成的数组)
            closing_iterations: 闭运算迭代次数
            kernel_size: 结构元素大小（必须是奇数）

        Returns:
            processed_mask: 处理后的二值掩膜
        """
        if closing_iterations <= 0:
            return binary_mask

        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            logger.debug(f"     - 调整结构元素大小为奇数: {kernel_size}")

        # 创建结构元素（使用方形结构元素）
        structure = np.ones((kernel_size, kernel_size), dtype=bool)

        processed_mask = binary_mask.copy()

        # 应用多次闭运算
        for i in range(closing_iterations):
            # 闭运算 = 先膨胀后腐蚀
            # 可以填补小的空洞，连接邻近的物体
            processed_mask = ndimage.binary_closing(processed_mask, structure=structure).astype(np.uint8)
            logger.debug(f"     - 第 {i + 1} 次闭运算完成")

        # 统计处理前后的变化
        original_count = np.sum(binary_mask)
        processed_count = np.sum(processed_mask)
        logger.info(f"    - 后处理: 原始像素 {original_count}, 处理后像素 {processed_count}, "
                    f"变化 {processed_count - original_count:+.0f} 像素")

        return processed_mask

    def process_sequence(self, image_frames, threshold_percent, frame_percent,
                         sequence_id, file_name, gaussian_sigma=0,
                         closing_iterations=1, kernel_size=3):
        """
        处理单个DSA序列的指定比例帧数，从第二帧开始，只标记像素值下降超过阈值的区域，
        并进行高斯滤波和形态学后处理

        Args:
            image_frames: 多帧图像数组 [frames, height, width]
            threshold_percent: 变化阈值百分比
            frame_percent: 使用前百分之多少的帧进行标记
            sequence_id: 序列ID ('0' 或 '1')
            file_name: 原始文件名
            gaussian_sigma: 高斯滤波的标准差，设为0则不进行滤波
            closing_iterations: 闭运算迭代次数
            kernel_size: 结构元素大小

        Returns:
            binary_mask: 处理后的二值掩膜图像，失败返回None
        """
        if image_frames is None or len(image_frames) < 2:
            logger.warning(
                f"  - 序列 {sequence_id} 帧数不足2帧 (只有 {len(image_frames) if image_frames is not None else 0} 帧)，跳过")
            return None

        num_frames = len(image_frames)
        logger.info(f"  - 处理序列 {sequence_id}: 共 {num_frames} 帧")

        # 应用高斯滤波（如果sigma > 0）
        if gaussian_sigma > 0:
            image_frames = self.apply_gaussian_filter(image_frames, sigma=gaussian_sigma)

        # 获取图像尺寸
        height, width = image_frames[0].shape

        # 计算全局像素值范围（使用所有帧）
        all_pixels = image_frames.flatten()
        pixel_min, pixel_max = np.min(all_pixels), np.max(all_pixels)
        pixel_range = pixel_max - pixel_min

        if pixel_range == 0:
            pixel_range = 1  # 避免除零错误

        threshold_value = pixel_range * (threshold_percent / 100.0)

        logger.info(f"    - 像素范围: {pixel_min:.0f} - {pixel_max:.0f}, 阈值: {threshold_value:.2f}")

        # 计算要处理的帧数（基于frame_percent）
        frames_to_process = int(num_frames * frame_percent / 100)

        # 确保至少处理2帧
        if frames_to_process < 2:
            frames_to_process = min(2, num_frames)
            logger.info(f"    - 根据比例计算的处理帧数过少，调整为 {frames_to_process} 帧")

        # 确保不超过总帧数
        frames_to_process = min(frames_to_process, num_frames)

        # 从第二帧开始处理（跳过第一帧）
        start_frame = 1  # 从索引1开始（第二帧）
        end_frame = frames_to_process

        logger.info(f"    - 使用帧范围: 第2帧到第{end_frame}帧 (共{end_frame - 1}对比较)")
        logger.info(f"    - 只标记像素值下降的区域")

        # 初始化标记矩阵
        marked_pixels = np.zeros((height, width), dtype=np.uint8)

        # 逐帧比较 - 从第二帧开始，只标记像素值下降的区域
        decrease_count = []
        for i in range(start_frame, end_frame):
            # 比较当前帧与前一帧
            # 计算像素值下降（前一帧比当前帧亮，即对比剂流入导致信号降低）
            decrease = image_frames[i - 1] - image_frames[i]

            # 只标记下降超过阈值的像素点
            significant_decrease = decrease > threshold_value
            marked_pixels[significant_decrease] = 1

            decrease_count_val = np.sum(significant_decrease)
            decrease_count.append(decrease_count_val)

            if decrease_count_val > 0:
                logger.debug(f"      - 帧 {i}-{i + 1}: {decrease_count_val} 个下降像素点")

        if decrease_count:
            logger.info(f"    - 平均每帧下降像素: {np.mean(decrease_count):.0f} 像素")
            logger.info(f"    - 初始标记像素: {np.sum(marked_pixels)}")

        # 应用形态学后处理
        if closing_iterations > 0:
            logger.info(f"    - 应用形态学后处理: 闭运算 {closing_iterations} 次, 核大小 {kernel_size}")
            marked_pixels = self.morphological_postprocessing(
                marked_pixels,
                closing_iterations=closing_iterations,
                kernel_size=kernel_size
            )

        return marked_pixels

    def save_result(self, mask, file_name, sequence_id):
        """
        保存结果图像

        Args:
            mask: 二值掩膜数组
            file_name: 原始文件名
            sequence_id: 序列ID ('0' 或 '1')
        """
        # 构建输出文件名：原始文件名_序列号.png
        output_filename = f"{file_name}_{sequence_id}.png"
        output_path = os.path.join(self.output_dir, output_filename)

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 转换为8位图像（0或255）并保存
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8)
        mask_img.save(output_path)

        return output_path

    def run(self, threshold_percent=30, frame_percent=50, gaussian_sigma=1.0,
            closing_iterations=2, kernel_size=3):
        """
        运行分析流程 - 自动处理所有文件

        Args:
            threshold_percent: 变化阈值百分比（可调整）
            frame_percent: 使用前百分之多少的帧进行标记
            gaussian_sigma: 高斯滤波的标准差，设为0则不进行滤波
            closing_iterations: 闭运算迭代次数
            kernel_size: 形态学操作的结构元素大小
        """
        logger.info("=" * 70)
        logger.info("DSA图像血管标记程序开始运行")
        logger.info(f"阈值: {threshold_percent}% (只标记像素值下降的区域)")
        logger.info(f"使用前 {frame_percent}% 的帧进行标记 (从第二帧开始)")
        if gaussian_sigma > 0:
            logger.info(f"高斯滤波: sigma={gaussian_sigma}")
        else:
            logger.info(f"高斯滤波: 关闭")
        logger.info(f"后处理: 闭运算 {closing_iterations} 次, 核大小 {kernel_size}")
        logger.info(f"Excel文件: {self.excel_path}")
        logger.info(f"图像目录: {self.image_base_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 70)

        # 获取文件名列表
        file_names = self.read_excel_files()
        if not file_names:
            logger.error("没有找到文件名，程序退出")
            return

        self.stats['total'] = len(file_names)
        logger.info(f"将处理全部 {self.stats['total']} 个文件")

        # 处理每个文件名
        for idx, file_name in enumerate(tqdm(file_names, desc="处理进度"), 1):
            logger.info(f"[{idx}/{self.stats['total']}] 处理: {file_name}")

            # 查找该文件名的两个序列文件
            sequence_files = self.find_sequence_files(file_name)

            # 处理序列0
            mask_0 = None
            if sequence_files['0']:
                try:
                    # 读取多帧DICOM
                    frames_0 = self.read_multiframe_dicom(sequence_files['0'])
                    if frames_0 is not None:
                        mask_0 = self.process_sequence(
                            frames_0,
                            threshold_percent,
                            frame_percent,
                            '0',
                            file_name,
                            gaussian_sigma=gaussian_sigma,
                            closing_iterations=closing_iterations,
                            kernel_size=kernel_size
                        )
                        if mask_0 is not None:
                            output_path = self.save_result(mask_0, file_name, '0')
                            logger.info(f"    - 序列0结果已保存: {output_path}")
                            self.stats['sequences_processed'] += 1
                        else:
                            logger.warning(f"    - 序列0处理失败")
                            self.stats['failed'] += 1
                    else:
                        logger.warning(f"    - 序列0读取失败")
                        self.stats['failed'] += 1
                except Exception as e:
                    logger.error(f"    - 处理序列0时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['failed'] += 1
            else:
                self.stats['skipped_missing'] += 1

            # 处理序列1
            mask_1 = None
            if sequence_files['1']:
                try:
                    # 读取多帧DICOM
                    frames_1 = self.read_multiframe_dicom(sequence_files['1'])
                    if frames_1 is not None:
                        mask_1 = self.process_sequence(
                            frames_1,
                            threshold_percent,
                            frame_percent,
                            '1',
                            file_name,
                            gaussian_sigma=gaussian_sigma,
                            closing_iterations=closing_iterations,
                            kernel_size=kernel_size
                        )
                        if mask_1 is not None:
                            output_path = self.save_result(mask_1, file_name, '1')
                            logger.info(f"    - 序列1结果已保存: {output_path}")
                            self.stats['sequences_processed'] += 1
                        else:
                            logger.warning(f"    - 序列1处理失败")
                            self.stats['failed'] += 1
                    else:
                        logger.warning(f"    - 序列1读取失败")
                        self.stats['failed'] += 1
                except Exception as e:
                    logger.error(f"    - 处理序列1时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['failed'] += 1
            else:
                self.stats['skipped_missing'] += 1

            # 统计处理的文件数（只要有任一序列成功就算）
            if (sequence_files['0'] and mask_0 is not None) or \
                    (sequence_files['1'] and mask_1 is not None):
                self.stats['processed'] += 1

            if sequence_files['0'] or sequence_files['1']:
                self.stats['sequences_found'] += (1 if sequence_files['0'] else 0) + (1 if sequence_files['1'] else 0)

        # 输出统计信息
        logger.info("=" * 70)
        logger.info("处理完成！统计信息：")
        logger.info(f"总文件数: {self.stats['total']}")
        logger.info(f"成功处理文件数: {self.stats['processed']}")
        logger.info(f"找到的序列总数: {self.stats['sequences_found']}")
        logger.info(f"成功处理序列数: {self.stats['sequences_processed']}")
        logger.info(f"处理失败的序列数: {self.stats['failed']}")
        logger.info(f"跳过的文件数（无图像）: {self.stats['skipped_missing']}")
        logger.info("=" * 70)

        # 如果有失败的文件，列出详细信息
        if self.stats['failed'] > 0:
            logger.warning("部分序列处理失败，请检查日志了解详情")


def main():
    """主函数 - 自动处理所有文件"""
    # 配置参数
    EXCEL_PATH = r"D:\med_data\ai\classify_500.xlsx"
    IMAGE_BASE_DIR = r"D:\med_data\ANY\0"
    OUTPUT_DIR = r"D:\med_data\ai\vessel"

    # 阈值设置 - 只标记像素值下降超过此阈值的区域
    THRESHOLD_PERCENT = 18  # 范围: 0-100，可根据需要调整

    # 帧数使用比例设置 - 使用前百分之多少的帧进行血管标记
    FRAME_PERCENT = 60  # 使用前30%的帧进行标记（从第二帧开始）

    # 高斯滤波参数
    GAUSSIAN_SIGMA = 1.5  # 高斯滤波标准差，设为0则关闭滤波
    # sigma值建议：
    # - 0.5-1.0: 轻微平滑，保留更多细节
    # - 1.0-2.0: 中等平滑，去除噪声
    # - >2.0: 强平滑，适合噪声大的图像

    # 形态学后处理参数
    CLOSING_ITERATIONS = 5  # 闭运算迭代次数，设为0可关闭后处理
    KERNEL_SIZE = 5  # 结构元素大小（必须是奇数）

    # 创建分析器并运行（自动处理所有文件）
    analyzer = DSAAnalyzer(EXCEL_PATH, IMAGE_BASE_DIR, OUTPUT_DIR)
    analyzer.run(
        threshold_percent=THRESHOLD_PERCENT,
        frame_percent=FRAME_PERCENT,
        gaussian_sigma=GAUSSIAN_SIGMA,
        closing_iterations=CLOSING_ITERATIONS,
        kernel_size=KERNEL_SIZE
    )

    print("\n程序运行完毕！")
    print(f"参数设置:")
    print(f"  - 阈值: {THRESHOLD_PERCENT}%")
    print(f"  - 使用前 {FRAME_PERCENT}% 的帧 (从第二帧开始)")
    if GAUSSIAN_SIGMA > 0:
        print(f"  - 高斯滤波: sigma={GAUSSIAN_SIGMA}")
    else:
        print(f"  - 高斯滤波: 关闭")
    print(f"  - 闭运算: {CLOSING_ITERATIONS} 次, 核大小 {KERNEL_SIZE}")
    print("按任意键退出...")
    input()


if __name__ == "__main__":
    main()