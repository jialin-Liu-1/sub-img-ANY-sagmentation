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
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DSAAnalyzer:
    """DSA图像分析器 - 生成最长中心线与血管提取图的叠加图像"""

    def __init__(self, excel_path, image_base_dir, output_dir,
                 centerline_output_dir, longest_centerline_output_dir,
                 overlay_output_dir):
        self.excel_path = excel_path
        self.image_base_dir = image_base_dir
        self.output_dir = output_dir
        self.centerline_output_dir = centerline_output_dir
        self.longest_centerline_output_dir = longest_centerline_output_dir
        self.overlay_output_dir = overlay_output_dir
        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'skipped_missing': 0,
            'sequences_found': 0,
            'sequences_processed': 0,
            'centerline_generated': 0,
            'longest_centerline_generated': 0,
            'overlay_generated': 0
        }

        # 目标对比度范围
        self.target_min = 0
        self.target_max = 150
        self.target_range = self.target_max - self.target_min

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
            else:
                if len(dicom_data.pixel_array.shape) == 3:
                    num_frames = dicom_data.pixel_array.shape[0]
                else:
                    num_frames = 1

            # 获取像素数组
            pixel_array = dicom_data.pixel_array

            # 处理像素数组
            if len(pixel_array.shape) == 3:
                frames = pixel_array.astype(np.float32)
            elif len(pixel_array.shape) == 2:
                frames = np.expand_dims(pixel_array, axis=0).astype(np.float32)
            else:
                logger.error(f"    - 不支持的图像维度: {pixel_array.shape}")
                return None

            logger.info(f"    - 成功读取 {frames.shape[0]} 帧, 每帧大小: {frames.shape[1]}x{frames.shape[2]}")
            return frames

        except Exception as e:
            logger.error(f"读取多帧DICOM文件失败 {file_path}: {e}")
            return None

    def normalize_contrast_using_frame4(self, image_frames):
        """
        使用第4帧进行对比度归一化
        将图像像素值线性拉伸到目标范围 [0, 150]
        """
        if len(image_frames) < 4:
            logger.warning(f"    - 帧数不足4帧，无法使用第4帧进行归一化")
            return image_frames

        # 获取第4帧（索引3）
        frame4 = image_frames[3]

        # 计算第4帧的最小值和最大值
        frame4_min = frame4.min()
        frame4_max = frame4.max()

        if frame4_max - frame4_min < 1e-6:
            logger.warning(f"    - 第4帧几乎没有变化，跳过归一化")
            return image_frames

        logger.info(f"    - 第4帧原始范围: [{frame4_min:.2f}, {frame4_max:.2f}]")

        # 线性拉伸到目标范围
        normalized_frames = np.zeros_like(image_frames)
        for i in range(len(image_frames)):
            # 使用第4帧的统计信息进行归一化
            normalized = (image_frames[i] - frame4_min) / (frame4_max - frame4_min)
            normalized = normalized * self.target_range + self.target_min
            normalized_frames[i] = np.clip(normalized, self.target_min, self.target_max)

        logger.info(f"    - 归一化后第4帧范围: [{normalized_frames[3].min():.2f}, {normalized_frames[3].max():.2f}]")

        return normalized_frames

    def apply_gaussian_filter(self, image_frames, sigma=1.0):
        """对图像序列应用高斯滤波"""
        filtered_frames = np.zeros_like(image_frames)
        for i in range(len(image_frames)):
            filtered_frames[i] = gaussian_filter(image_frames[i], sigma=sigma)
        return filtered_frames

    def morphological_postprocessing(self, binary_mask, closing_iterations=1, kernel_size=3):
        """对二值掩膜进行形态学后处理"""
        if closing_iterations <= 0:
            return binary_mask

        if kernel_size % 2 == 0:
            kernel_size += 1

        structure = np.ones((kernel_size, kernel_size), dtype=bool)
        processed_mask = binary_mask.copy()

        for i in range(closing_iterations):
            processed_mask = ndimage.binary_closing(processed_mask, structure=structure).astype(np.uint8)

        return processed_mask

    def extract_vessel_centerline(self, binary_mask):
        """提取血管中心线"""
        try:
            if binary_mask.max() > 1:
                binary_mask = (binary_mask > 0).astype(np.uint8)

            skeleton = skeletonize(binary_mask.astype(bool))
            centerline = skeleton.astype(np.uint8)
            return centerline
        except Exception as e:
            logger.error(f"提取中心线时出错: {e}")
            return np.zeros_like(binary_mask)

    def find_endpoints(self, skeleton):
        """
        找到骨架线的端点（只有1个相邻点的点）
        """
        # 使用8邻域卷积计算每个点的邻居数量
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # 排除自身

        # 计算每个点的邻居数量
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

        # 端点：骨架点且邻居数为1
        endpoints = (skeleton > 0) & (neighbors == 1)

        return endpoints

    def find_bottom_candidates(self, skeleton, height_percent=5):
        """
        找到图像底部的候选端点

        Args:
            skeleton: 二值中心线图像
            height_percent: 底部区域高度百分比（相对于图像总高度）

        Returns:
            candidates: 候选端点列表，每个元素为 (y, x, 从该点出发能到达的最高y坐标)
        """
        h, w = skeleton.shape
        endpoints = self.find_endpoints(skeleton)
        endpoint_coords = list(zip(*np.where(endpoints)))

        if not endpoint_coords:
            logger.info(f"    - 未找到任何端点")
            return []

        # 确定底部区域的范围（图像最底部的height_percent%区域）
        bottom_threshold = h - int(h * height_percent / 100)
        logger.info(f"    - 底部区域阈值: y > {bottom_threshold} (图像高度: {h})")

        # 筛选出在底部区域的端点
        bottom_candidates = []
        for y, x in endpoint_coords:
            if y >= bottom_threshold:  # 在底部区域内
                # 计算从这个端点出发能到达的最高点
                highest_y = self.find_highest_point_from_start(skeleton, (y, x))
                bottom_candidates.append((y, x, highest_y))
                logger.info(f"    - 候选端点: ({y}, {x}), 能到达最高y={highest_y}")

        return bottom_candidates

    def find_highest_point_from_start(self, skeleton, start_point):
        """
        从起点出发，找到能到达的最高点（最小y坐标）

        Args:
            skeleton: 二值中心线图像
            start_point: 起点坐标 (y, x)

        Returns:
            highest_y: 能到达的最小y坐标（越靠上y值越小）
        """
        from collections import deque

        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        queue = deque()
        queue.append(start_point)
        visited[start_point] = True

        highest_y = start_point[0]  # 初始化为起点的y坐标

        while queue:
            y, x = queue.popleft()

            # 更新最高点（y值最小）
            if y < highest_y:
                highest_y = y

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        return highest_y

    def select_best_bottom_point(self, candidates):
        """
        从候选底部端点中选择最佳起点

        规则：
        1. 优先选择能到达最高点（y值最小）的端点
        2. 如果多个端点都能到达相近的高度，选择最靠下的那个

        Args:
            candidates: 候选端点列表 [(y, x, highest_y), ...]

        Returns:
            best_point: 最佳起点坐标 (y, x)
        """
        if not candidates:
            return None

        # 按能到达的最高点排序（highest_y越小越好，即越高）
        candidates.sort(key=lambda c: c[2])

        best_highest_y = candidates[0][2]
        best_candidates = [c for c in candidates if c[2] == best_highest_y]

        logger.info(f"    - 最佳能到达高度: y={best_highest_y}")
        logger.info(f"    - 有 {len(best_candidates)} 个端点能到达这个高度")

        if len(best_candidates) == 1:
            # 只有一个端点能到达最高点
            y, x, _ = best_candidates[0]
            logger.info(f"    - 选择唯一能到达最高点的端点: ({y}, {x})")
            return (y, x)
        else:
            # 多个端点都能到达最高点，选择最靠下的那个（y值最大）
            best_candidates.sort(key=lambda c: c[0], reverse=True)
            y, x, _ = best_candidates[0]
            logger.info(f"    - 多个端点都能到达最高点，选择最靠下的: ({y}, {x})")
            return (y, x)

    def find_farthest_point_from_bottom(self, skeleton, bottom_point):
        """
        从底部端点出发，找到最远的点（通过BFS）

        Args:
            skeleton: 二值中心线图像
            bottom_point: 底部端点坐标

        Returns:
            farthest_point: 最远点坐标
            distance_map: 距离图
        """
        from collections import deque

        h, w = skeleton.shape
        distance = -np.ones((h, w), dtype=np.float32)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # BFS
        queue = deque()
        queue.append(bottom_point)
        distance[bottom_point] = 0

        farthest_point = bottom_point
        max_dist = 0

        while queue:
            y, x = queue.popleft()

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] > 0 and distance[ny, nx] < 0:
                        distance[ny, nx] = distance[y, x] + 1
                        queue.append((ny, nx))

                        if distance[ny, nx] > max_dist:
                            max_dist = distance[ny, nx]
                            farthest_point = (ny, nx)

        logger.info(f"    - 最远点坐标: {farthest_point}, 距离: {max_dist} 像素")

        return farthest_point, distance

    def extract_longest_centerline_from_bottom(self, centerline, bottom_height_percent=5):
        """
        从图像底部的候选端点中，选择能到达最高点的最优起点，提取主干血管路径

        Args:
            centerline: 二值中心线图像
            bottom_height_percent: 底部区域高度百分比

        Returns:
            main_path_mask: 只包含主干路径的二值图像
        """
        if np.sum(centerline) == 0:
            return np.zeros_like(centerline)

        # 1. 找到底部区域的候选端点
        candidates = self.find_bottom_candidates(centerline, height_percent=bottom_height_percent)

        if not candidates:
            logger.warning(f"    - 底部区域未找到端点，返回原始中心线")
            return centerline

        # 2. 选择最佳起点
        bottom_point = self.select_best_bottom_point(candidates)
        if bottom_point is None:
            logger.warning(f"    - 未选择到合适的底部端点，返回原始中心线")
            return centerline

        # 3. 找到从底部出发的最远点
        farthest_point, distance_map = self.find_farthest_point_from_bottom(centerline, bottom_point)

        # 4. 使用最短路径算法找到两点之间的路径
        # 创建代价矩阵（中心线上的点代价为1，其余为无穷大）
        cost = np.ones_like(centerline, dtype=np.float32) * 1e9
        cost[centerline > 0] = 1

        try:
            # 找到从bottom_point到farthest_point的路径
            indices, distance = route_through_array(
                cost,
                bottom_point,
                farthest_point,
                fully_connected=True
            )

            # 创建主干路径的二值图像
            main_path_mask = np.zeros_like(centerline, dtype=np.uint8)
            for y, x in indices:
                main_path_mask[y, x] = 1

            logger.info(f"    - 主干路径长度: {distance:.0f} 像素")
            logger.info(f"    - 路径起点(底部): {bottom_point}")
            logger.info(f"    - 路径终点: {farthest_point}")
            logger.info(f"    - 路径最高点: {min([p[0] for p in indices])}")

            return main_path_mask

        except Exception as e:
            logger.error(f"    - 路径提取失败: {e}")
            return np.zeros_like(centerline)

    def create_centerline_overlay(self, original_mask, centerline, file_name, sequence_id):
        """创建中心线叠加图像（红色中心线 + 原始血管标记）"""
        h, w = original_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 原始血管标记用绿色
        green_channel = (original_mask * 255).astype(np.uint8)
        overlay[:, :, 1] = green_channel

        # 中心线用红色
        red_channel = (centerline * 255).astype(np.uint8)
        overlay[:, :, 0] = red_channel
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - centerline)

        return overlay

    def create_longest_centerline_overlay(self, original_mask, longest_centerline, file_name, sequence_id):
        """
        创建最长中心线与血管提取图的叠加图像
        血管区域用绿色，最长中心线用红色

        Args:
            original_mask: 原始血管标记图
            longest_centerline: 最长中心线
            file_name: 文件名
            sequence_id: 序列ID

        Returns:
            overlay_image: RGB叠加图像
        """
        h, w = original_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 原始血管标记用绿色 (半透明效果)
        green_channel = (original_mask * 180).astype(np.uint8)  # 使用180而不是255，使红色更突出
        overlay[:, :, 1] = green_channel

        # 最长中心线用亮红色
        red_channel = (longest_centerline * 255).astype(np.uint8)
        overlay[:, :, 0] = red_channel

        # 在中心线位置，降低绿色通道的强度，使红色更明显
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - longest_centerline * 0.7)

        # 可选：添加蓝色通道增加对比度
        blue_channel = (original_mask * 50).astype(np.uint8)  # 轻微蓝色背景
        overlay[:, :, 2] = blue_channel * (1 - longest_centerline)

        return overlay

    def process_sequence(self, image_frames, threshold_percent, frame_percent,
                         sequence_id, file_name, gaussian_sigma=0,
                         closing_iterations=1, kernel_size=3,
                         extract_centerline=True,
                         extract_longest_centerline=True,
                         enable_contrast_normalization=True,
                         bottom_height_percent=5):  # 新增参数
        """
        处理单个DSA序列
        """
        if image_frames is None or len(image_frames) < 2:
            return None, None, None

        num_frames, height, width = image_frames.shape
        logger.info(f"  - 处理序列 {sequence_id}: 共 {num_frames} 帧")

        # 对比度归一化
        if enable_contrast_normalization:
            logger.info(f"    - 启用对比度归一化")
            image_frames = self.normalize_contrast_using_frame4(image_frames)
        else:
            logger.info(f"    - 未启用对比度归一化")

        # 应用高斯滤波
        if gaussian_sigma > 0:
            image_frames = self.apply_gaussian_filter(image_frames, sigma=gaussian_sigma)

        # 计算全局像素值范围
        all_pixels = image_frames.flatten()
        pixel_min, pixel_max = np.min(all_pixels), np.max(all_pixels)
        pixel_range = pixel_max - pixel_min

        if pixel_range == 0:
            pixel_range = 1

        threshold_value = pixel_range * (threshold_percent / 100.0)

        logger.info(f"    - 像素范围: [{pixel_min:.2f}, {pixel_max:.2f}], 阈值: {threshold_value:.2f}")

        # 计算要处理的帧数
        frames_to_process = int(num_frames * frame_percent / 100)

        if frames_to_process < 2:
            frames_to_process = min(2, num_frames)
            logger.info(f"    - 根据比例计算的处理帧数过少，调整为 {frames_to_process} 帧")

        frames_to_process = min(frames_to_process, num_frames)

        # 从第二帧开始处理
        start_frame = 1
        end_frame = frames_to_process

        logger.info(f"    - 使用帧范围: 第2帧到第{end_frame}帧 (共{end_frame - 1}对比较)")
        logger.info(f"    - 只标记像素值下降的区域")

        # 初始化标记矩阵
        marked_pixels = np.zeros((height, width), dtype=np.uint8)

        # 逐帧比较
        decrease_count = []
        for i in range(start_frame, end_frame):
            decrease = image_frames[i - 1] - image_frames[i]
            significant_decrease = decrease > threshold_value
            marked_pixels[significant_decrease] = 1

            decrease_count_val = np.sum(significant_decrease)
            decrease_count.append(decrease_count_val)

        if decrease_count:
            logger.info(f"    - 平均每帧下降像素: {np.mean(decrease_count):.0f} 像素")
            logger.info(f"    - 初始标记像素: {np.sum(marked_pixels)}")

        # 形态学后处理
        if closing_iterations > 0:
            logger.info(f"    - 应用形态学后处理")
            marked_pixels = self.morphological_postprocessing(
                marked_pixels,
                closing_iterations=closing_iterations,
                kernel_size=kernel_size
            )

        # 提取中心线
        centerline = None
        main_centerline = None

        if extract_centerline and np.sum(marked_pixels) > 100:
            logger.info("    - 提取血管中心线")
            centerline = self.extract_vessel_centerline(marked_pixels)
            if np.sum(centerline) > 0:
                self.stats['centerline_generated'] += 1

                # 提取主干中心线（从底部开始，考虑多个候选）
                if extract_longest_centerline:
                    logger.info(f"    - 提取主干中心线（底部区域高度: {bottom_height_percent}%）")
                    main_centerline = self.extract_longest_centerline_from_bottom(
                        centerline,
                        bottom_height_percent=bottom_height_percent
                    )
                    if np.sum(main_centerline) > 0:
                        self.stats['longest_centerline_generated'] += 1
                        logger.info(f"    - 主干中心线像素: {np.sum(main_centerline)}")

        return marked_pixels, centerline, main_centerline

    def save_result(self, mask, centerline, main_centerline, file_name, sequence_id):
        """保存结果图像"""
        # 保存血管标记图
        output_filename = f"{file_name}_{sequence_id}.png"
        output_path = os.path.join(self.output_dir, output_filename)

        os.makedirs(self.output_dir, exist_ok=True)

        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8)
        mask_img.save(output_path)
        logger.info(f"    - 血管标记图已保存: {output_path}")

        # 保存中心线叠加图
        if centerline is not None and np.sum(centerline) > 0:
            overlay = self.create_centerline_overlay(mask, centerline, file_name, sequence_id)

            overlay_filename = f"{file_name}_{sequence_id}_centerline.png"
            overlay_path = os.path.join(self.centerline_output_dir, overlay_filename)

            os.makedirs(self.centerline_output_dir, exist_ok=True)

            overlay_img = Image.fromarray(overlay)
            overlay_img.save(overlay_path)
            logger.info(f"    - 中心线叠加图已保存: {overlay_path}")

        # 保存主干中心线图
        if main_centerline is not None and np.sum(main_centerline) > 0:
            main_centerline_filename = f"{file_name}_{sequence_id}_main_centerline.png"
            main_centerline_path = os.path.join(self.longest_centerline_output_dir, main_centerline_filename)

            os.makedirs(self.longest_centerline_output_dir, exist_ok=True)

            main_centerline_uint8 = (main_centerline * 255).astype(np.uint8)
            main_centerline_img = Image.fromarray(main_centerline_uint8)
            main_centerline_img.save(main_centerline_path)
            logger.info(f"    - 主干中心线图已保存: {main_centerline_path}")

            # 保存主干中心线与血管的叠加图
            overlay_main = self.create_longest_centerline_overlay(mask, main_centerline, file_name, sequence_id)
            overlay_main_filename = f"{file_name}_{sequence_id}_main_overlay.png"
            overlay_main_path = os.path.join(self.overlay_output_dir, overlay_main_filename)

            os.makedirs(self.overlay_output_dir, exist_ok=True)
            overlay_main_img = Image.fromarray(overlay_main)
            overlay_main_img.save(overlay_main_path)
            logger.info(f"    - 主干中心线叠加图已保存: {overlay_main_path}")
            self.stats['overlay_generated'] += 1

        return output_path

    def run(self,
            threshold_percent=18,
            frame_percent=60,
            gaussian_sigma=1.5,
            closing_iterations=5,
            kernel_size=5,
            extract_centerline=True,
            extract_longest_centerline=True,
            enable_contrast_normalization=True,
            bottom_height_percent=5):  # 新增参数
        """
        运行分析流程
        """
        logger.info("=" * 70)
        logger.info("DSA图像血管标记程序开始运行")
        logger.info("=" * 70)
        logger.info(f"参数设置:")
        logger.info(f"  - 对比度归一化: {'启用' if enable_contrast_normalization else '禁用'}")
        logger.info(f"  - 阈值: {threshold_percent}%")
        logger.info(f"  - 使用前 {frame_percent}% 的帧")
        logger.info(f"  - 高斯滤波: sigma={gaussian_sigma}")
        logger.info(f"  - 后处理: 闭运算 {closing_iterations} 次, 核大小 {kernel_size}")
        logger.info(f"  - 中心线提取: {'开启' if extract_centerline else '关闭'}")
        logger.info(f"  - 主干中心线提取: {'开启' if extract_longest_centerline else '关闭'}")
        logger.info(f"  - 底部区域高度: {bottom_height_percent}% (用于选择主干起点)")
        logger.info(f"Excel文件: {self.excel_path}")
        logger.info(f"图像目录: {self.image_base_dir}")
        logger.info(f"血管标记输出目录: {self.output_dir}")
        logger.info(f"中心线输出目录: {self.centerline_output_dir}")
        logger.info(f"主干中心线输出目录: {self.longest_centerline_output_dir}")
        logger.info(f"主干中心线叠加输出目录: {self.overlay_output_dir}")
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
            centerline_0 = None
            main_centerline_0 = None
            if sequence_files['0']:
                try:
                    frames_0 = self.read_multiframe_dicom(sequence_files['0'])
                    if frames_0 is not None:
                        mask_0, centerline_0, main_centerline_0 = self.process_sequence(
                            frames_0,
                            threshold_percent,
                            frame_percent,
                            '0',
                            file_name,
                            gaussian_sigma=gaussian_sigma,
                            closing_iterations=closing_iterations,
                            kernel_size=kernel_size,
                            extract_centerline=extract_centerline,
                            extract_longest_centerline=extract_longest_centerline,
                            enable_contrast_normalization=enable_contrast_normalization,
                            bottom_height_percent=bottom_height_percent
                        )
                        if mask_0 is not None:
                            self.save_result(mask_0, centerline_0, main_centerline_0, file_name, '0')
                            self.stats['sequences_processed'] += 1
                        else:
                            self.stats['failed'] += 1
                    else:
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
            centerline_1 = None
            main_centerline_1 = None
            if sequence_files['1']:
                try:
                    frames_1 = self.read_multiframe_dicom(sequence_files['1'])
                    if frames_1 is not None:
                        mask_1, centerline_1, main_centerline_1 = self.process_sequence(
                            frames_1,
                            threshold_percent,
                            frame_percent,
                            '1',
                            file_name,
                            gaussian_sigma=gaussian_sigma,
                            closing_iterations=closing_iterations,
                            kernel_size=kernel_size,
                            extract_centerline=extract_centerline,
                            extract_longest_centerline=extract_longest_centerline,
                            enable_contrast_normalization=enable_contrast_normalization,
                            bottom_height_percent=bottom_height_percent
                        )
                        if mask_1 is not None:
                            self.save_result(mask_1, centerline_1, main_centerline_1, file_name, '1')
                            self.stats['sequences_processed'] += 1
                        else:
                            self.stats['failed'] += 1
                    else:
                        self.stats['failed'] += 1
                except Exception as e:
                    logger.error(f"    - 处理序列1时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['failed'] += 1
            else:
                self.stats['skipped_missing'] += 1

            # 统计处理的文件数
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
        logger.info(f"成功生成中心线数: {self.stats['centerline_generated']}")
        logger.info(f"成功生成主干中心线数: {self.stats['longest_centerline_generated']}")
        logger.info(f"成功生成主干叠加图数: {self.stats['overlay_generated']}")
        logger.info(f"处理失败的序列数: {self.stats['failed']}")
        logger.info(f"跳过的文件数（无图像）: {self.stats['skipped_missing']}")
        logger.info("=" * 70)


def main():
    """主函数 - 参数配置"""
    # 基础路径配置
    EXCEL_PATH = r"D:\med_data\ai\classify_500.xlsx"
    IMAGE_BASE_DIR = r"D:\med_data\ANY\0"
    OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con"
    CENTERLINE_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\center"
    MAIN_CENTERLINE_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\center_Long"
    OVERLAY_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\center1"  # 新增：主干中心线叠加图输出目录

    # ========================================
    # 参数配置
    # ========================================

    # 对比度归一化开关
    ENABLE_CONTRAST_NORMALIZATION = True

    # 传统参数
    THRESHOLD_PERCENT = 15
    FRAME_PERCENT = 60
    GAUSSIAN_SIGMA = 1.5
    CLOSING_ITERATIONS = 3
    KERNEL_SIZE = 7
    EXTRACT_CENTERLINE = True
    EXTRACT_MAIN_CENTERLINE = True  # 是否提取主干中心线

    # 新增：底部区域高度百分比
    # 用于选择主干血管起点的底部区域高度
    # 例如：5% 表示图像最底部5%的区域
    BOTTOM_HEIGHT_PERCENT = 10

    print("\n" + "=" * 70)
    print("参数配置")
    print("=" * 70)
    print(f"对比度归一化: {'启用' if ENABLE_CONTRAST_NORMALIZATION else '禁用'}")
    print(f"阈值: {THRESHOLD_PERCENT}%")
    print(f"使用前 {FRAME_PERCENT}% 的帧")
    print(f"高斯滤波: sigma={GAUSSIAN_SIGMA}")
    print(f"后处理: 闭运算 {CLOSING_ITERATIONS} 次, 核大小 {KERNEL_SIZE}")
    print(f"中心线提取: {'开启' if EXTRACT_CENTERLINE else '关闭'}")
    print(f"主干中心线提取: {'开启' if EXTRACT_MAIN_CENTERLINE else '关闭'}")
    print(f"底部区域高度: {BOTTOM_HEIGHT_PERCENT}% (用于选择主干起点)")
    print("=" * 70)

    # 创建分析器并运行
    analyzer = DSAAnalyzer(
        EXCEL_PATH,
        IMAGE_BASE_DIR,
        OUTPUT_DIR,
        CENTERLINE_OUTPUT_DIR,
        MAIN_CENTERLINE_OUTPUT_DIR,
        OVERLAY_OUTPUT_DIR
    )
    analyzer.run(
        threshold_percent=THRESHOLD_PERCENT,
        frame_percent=FRAME_PERCENT,
        gaussian_sigma=GAUSSIAN_SIGMA,
        closing_iterations=CLOSING_ITERATIONS,
        kernel_size=KERNEL_SIZE,
        extract_centerline=EXTRACT_CENTERLINE,
        extract_longest_centerline=EXTRACT_MAIN_CENTERLINE,
        enable_contrast_normalization=ENABLE_CONTRAST_NORMALIZATION,
        bottom_height_percent=BOTTOM_HEIGHT_PERCENT
    )

    print("\n程序运行完毕！")
    print("按任意键退出...")
    input()


if __name__ == "__main__":
    main()
