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
    """DSA图像分析器 - 处理单帧DSA图像，提取血管和最长中心线"""

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
            'centerline_generated': 0,
            'longest_centerline_generated': 0,
            'overlay_generated': 0,
            'contrast_adjusted': 0,
            'edge_paths_rejected': 0
        }

        # 目标对比度范围
        self.target_min = 0
        self.target_max = 255
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

    def find_image_files(self, file_name):
        """
        查找指定文件名的两个单帧DSA图像文件
        返回: dict {'0': 文件路径, '1': 文件路径}
        """
        images = {'0': None, '1': None}

        if not os.path.exists(self.image_base_dir):
            logger.warning(f"图像目录不存在: {self.image_base_dir}")
            return images

        # 构建两个可能的文件名
        file_0 = f"{file_name}_0"
        file_1 = f"{file_name}_1"

        file_0_path = os.path.join(self.image_base_dir, file_0)
        file_1_path = os.path.join(self.image_base_dir, file_1)

        # 检查文件是否存在
        if os.path.exists(file_0_path):
            images['0'] = file_0_path
            logger.info(f"  - 找到图像0文件: {file_0}")
        else:
            logger.warning(f"  - 未找到图像0文件: {file_0}")

        if os.path.exists(file_1_path):
            images['1'] = file_1_path
            logger.info(f"  - 找到图像1文件: {file_1}")
        else:
            logger.warning(f"  - 未找到图像1文件: {file_1}")

        return images

    def read_single_frame_dicom(self, file_path):
        """
        读取单帧DICOM文件并返回像素数组

        Args:
            file_path: DICOM文件路径

        Returns:
            image: 单帧图像数组
        """
        try:
            dicom_data = pydicom.dcmread(file_path)

            # 获取像素数组
            pixel_array = dicom_data.pixel_array

            # 转换为float32类型以便后续处理
            image = pixel_array.astype(np.float32)

            logger.info(f"    - 成功读取单帧图像, 大小: {image.shape[0]}x{image.shape[1]}")

            return image

        except Exception as e:
            logger.error(f"读取单帧DICOM文件失败 {file_path}: {e}")
            return None

    def adaptive_contrast_adjustment(self, image):
        """
        对单帧图像进行自适应对比度调整

        Args:
            image: 单帧图像数组

        Returns:
            adjusted_image: 对比度调整后的图像
            adjustment_info: 调整信息字典
        """
        # 计算图像的最小值和最大值
        img_min = image.min()
        img_max = image.max()
        img_range = img_max - img_min

        logger.info(f"    - 图像原始范围: [{img_min:.2f}, {img_max:.2f}], 范围: {img_range:.2f}")

        # 如果对比度过低，进行拉伸
        if img_range < 50:  # 对比度过低阈值
            logger.info(f"    - 图像对比度不足 ({img_range:.2f} < 50)，进行对比度拉伸")

            # 线性拉伸到目标范围
            normalized = (image - img_min) / (img_range + 1e-6)
            adjusted_image = normalized * self.target_range + self.target_min
            adjusted_image = np.clip(adjusted_image, self.target_min, self.target_max)

            logger.info(f"    - 调整后图像范围: [{adjusted_image.min():.2f}, {adjusted_image.max():.2f}]")

            self.stats['contrast_adjusted'] += 1

            adjustment_info = {
                'adjusted': True,
                'original_min': img_min,
                'original_max': img_max,
                'original_range': img_range,
                'target_range': self.target_range
            }

            return adjusted_image, adjustment_info
        else:
            logger.info(f"    - 图像对比度足够 ({img_range:.2f} >= 50)，保持原始图像")

            adjustment_info = {
                'adjusted': False,
                'original_min': img_min,
                'original_max': img_max,
                'original_range': img_range,
                'reason': 'contrast_sufficient'
            }

            return image, adjustment_info

    def apply_gaussian_filter(self, image, sigma=1.0):
        """对单帧图像应用高斯滤波"""
        filtered_image = gaussian_filter(image, sigma=sigma)
        return filtered_image

    def threshold_image(self, image, threshold_percent=25):
        """
        对单帧图像进行阈值分割

        Args:
            image: 输入图像
            threshold_percent: 阈值百分比（低于图像像素值范围的此百分比标记为血管）

        Returns:
            binary_mask: 二值血管掩膜
        """
        # 计算图像像素值范围
        img_min = image.min()
        img_max = image.max()
        img_range = img_max - img_min

        if img_range == 0:
            img_range = 1

        # 计算阈值：像素值范围的threshold_percent%
        threshold_value = img_min + img_range * (threshold_percent / 100.0)

        logger.info(f"    - 图像范围: [{img_min:.2f}, {img_max:.2f}], 范围: {img_range:.2f}")
        logger.info(f"    - 阈值: {threshold_value:.2f} (低于此值标记为血管)")

        # 创建二值掩膜：低于阈值的像素标记为血管
        binary_mask = (image < threshold_value).astype(np.uint8)

        logger.info(f"    - 初始血管像素数: {np.sum(binary_mask)}")

        return binary_mask

    def morphological_processing(self, binary_mask,
                                 dilation_iterations=0,
                                 erosion_iterations=0,
                                 kernel_size=3):
        """
        对二值掩膜进行自定义的形态学处理
        """
        if dilation_iterations == 0 and erosion_iterations == 0:
            return binary_mask

        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 创建结构元素
        structure = np.ones((kernel_size, kernel_size), dtype=bool)

        processed_mask = binary_mask.copy().astype(np.uint8)

        # 先进行膨胀操作（连续多次）
        if dilation_iterations > 0:
            logger.info(f"      - 执行 {dilation_iterations} 次连续膨胀 (核大小 {kernel_size})")
            for i in range(dilation_iterations):
                processed_mask = ndimage.binary_dilation(processed_mask, structure=structure).astype(np.uint8)
                if i == 0 or i == dilation_iterations - 1:
                    current_sum = np.sum(processed_mask)
                    logger.info(f"        - 第{i + 1}次膨胀后像素数: {current_sum}")

        # 再进行腐蚀操作（连续多次）
        if erosion_iterations > 0:
            logger.info(f"      - 执行 {erosion_iterations} 次连续腐蚀 (核大小 {kernel_size})")
            for i in range(erosion_iterations):
                processed_mask = ndimage.binary_erosion(processed_mask, structure=structure).astype(np.uint8)
                if i == 0 or i == erosion_iterations - 1:
                    current_sum = np.sum(processed_mask)
                    logger.info(f"        - 第{i + 1}次腐蚀后像素数: {current_sum}")

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

    def filter_edge_points(self, centerline, edge_percent=5):
        """
        简单粗暴：移除距离图像左右两侧边缘一定百分比范围内的所有中心线点

        Args:
            centerline: 二值中心线图像
            edge_percent: 排除边缘的百分比（左右两侧各排除这个比例）

        Returns:
            filtered_centerline: 过滤后的中心线图像
        """
        h, w = centerline.shape

        # 计算要排除的边缘宽度
        edge_width = int(w * edge_percent / 100)

        # 创建掩码：只保留中心区域
        mask = np.zeros_like(centerline, dtype=bool)
        mask[:, edge_width:w - edge_width] = True

        # 应用掩码
        filtered_centerline = centerline.copy()
        filtered_centerline[~mask] = 0

        original_count = np.sum(centerline)
        filtered_count = np.sum(filtered_centerline)

        logger.info(f"    - 边缘过滤: 排除左右各{edge_percent}%区域 ({edge_width}像素)")
        logger.info(f"    - 中心线点: {original_count} -> {filtered_count} (减少了{original_count - filtered_count})")

        return filtered_centerline

    def find_bottom_candidates(self, skeleton, height_percent=5, exclude_edge=True, edge_percent=5):
        """
        找到图像底部的候选端点，可以选择排除边缘区域的端点

        Args:
            skeleton: 二值中心线图像
            height_percent: 底部区域高度百分比
            exclude_edge: 是否排除边缘区域的端点
            edge_percent: 边缘区域百分比

        Returns:
            candidates: 候选端点列表，每个元素为 (y, x, highest_y)
        """
        h, w = skeleton.shape

        # 先过滤掉边缘区域的中心线点
        if exclude_edge:
            skeleton = self.filter_edge_points(skeleton, edge_percent)

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

    def extract_longest_centerline_from_bottom(self, centerline, bottom_height_percent=5, exclude_edge=True,
                                               edge_percent=5):
        """
        从图像底部的候选端点中，选择能到达最高点的最优起点，提取主干血管路径

        Args:
            centerline: 二值中心线图像
            bottom_height_percent: 底部区域高度百分比
            exclude_edge: 是否排除边缘区域
            edge_percent: 边缘区域百分比

        Returns:
            main_path_mask: 只包含主干路径的二值图像
        """
        if np.sum(centerline) == 0:
            return np.zeros_like(centerline)

        # 1. 如果启用边缘排除，先过滤掉边缘区域的中心线点
        if exclude_edge:
            filtered_centerline = self.filter_edge_points(centerline, edge_percent)
            if np.sum(filtered_centerline) < 10:  # 如果过滤后中心线太少，可能误删了真实血管
                logger.warning(f"    - 过滤后中心线点太少 ({np.sum(filtered_centerline)})，可能误删，使用原始中心线")
                filtered_centerline = centerline
        else:
            filtered_centerline = centerline

        # 2. 找到底部区域的候选端点（使用过滤后的中心线）
        candidates = self.find_bottom_candidates(filtered_centerline, height_percent=bottom_height_percent,
                                                 exclude_edge=False)  # 这里不再重复排除边缘

        if not candidates:
            logger.warning(f"    - 底部区域未找到端点，返回原始中心线")
            return centerline

        # 3. 选择最佳起点
        bottom_point = self.select_best_bottom_point(candidates)
        if bottom_point is None:
            logger.warning(f"    - 未选择到合适的底部端点，返回原始中心线")
            return centerline

        # 4. 找到从底部出发的最远点
        farthest_point, distance_map = self.find_farthest_point_from_bottom(filtered_centerline, bottom_point)

        # 5. 使用最短路径算法找到两点之间的路径
        # 创建代价矩阵（中心线上的点代价为1，其余为无穷大）
        cost = np.ones_like(filtered_centerline, dtype=np.float32) * 1e9
        cost[filtered_centerline > 0] = 1

        try:
            # 找到从bottom_point到farthest_point的路径
            indices, distance = route_through_array(
                cost,
                bottom_point,
                farthest_point,
                fully_connected=True
            )

            # 创建主干路径的二值图像
            main_path_mask = np.zeros_like(centerline, dtype=np.uint8)  # 使用原始尺寸
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

    def create_centerline_overlay(self, original_mask, centerline, file_name, image_id):
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

    def create_longest_centerline_overlay(self, original_mask, longest_centerline, file_name, image_id):
        """
        创建最长中心线与血管提取图的叠加图像
        """
        h, w = original_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 原始血管标记用绿色 (半透明效果)
        green_channel = (original_mask * 180).astype(np.uint8)
        overlay[:, :, 1] = green_channel

        # 最长中心线用亮红色
        red_channel = (longest_centerline * 255).astype(np.uint8)
        overlay[:, :, 0] = red_channel

        # 在中心线位置，降低绿色通道的强度，使红色更明显
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - longest_centerline * 0.7)

        # 添加蓝色通道增加对比度
        blue_channel = (original_mask * 50).astype(np.uint8)
        overlay[:, :, 2] = blue_channel * (1 - longest_centerline)

        return overlay

    def process_single_image(self, image_array,
                             threshold_percent=25,
                             gaussian_sigma=0,
                             dilation_iterations=0,
                             erosion_iterations=0,
                             kernel_size=3,
                             extract_centerline=True,
                             extract_longest_centerline=True,
                             enable_adaptive_contrast=True,
                             bottom_height_percent=5,
                             exclude_edge=True,
                             edge_percent=5):
        """
        处理单帧DSA图像

        Args:
            image_array: 输入图像数组
            threshold_percent: 阈值百分比
            gaussian_sigma: 高斯滤波参数
            dilation_iterations: 膨胀次数
            erosion_iterations: 腐蚀次数
            kernel_size: 形态学核大小
            extract_centerline: 是否提取中心线
            extract_longest_centerline: 是否提取主干中心线
            enable_adaptive_contrast: 是否启用自适应对比度调整
            bottom_height_percent: 底部区域高度百分比
            exclude_edge: 是否排除边缘区域
            edge_percent: 边缘区域百分比

        Returns:
            binary_mask: 二值血管掩膜
            centerline: 中心线
            main_centerline: 主干中心线
        """
        if image_array is None:
            return None, None, None

        height, width = image_array.shape
        logger.info(f"  - 处理单帧图像, 大小: {height}x{width}")

        # 自适应对比度调整
        if enable_adaptive_contrast:
            logger.info(f"    - 启用自适应对比度调整")
            image_array, contrast_info = self.adaptive_contrast_adjustment(image_array)
            if contrast_info['adjusted']:
                logger.info(
                    f"    - 对比度已调整: 原始范围 {contrast_info['original_range']:.2f} -> 目标范围 {contrast_info['target_range']}")
        else:
            logger.info(f"    - 未启用自适应对比度调整")

        # 应用高斯滤波
        if gaussian_sigma > 0:
            logger.info(f"    - 应用高斯滤波: sigma={gaussian_sigma}")
            image_array = self.apply_gaussian_filter(image_array, sigma=gaussian_sigma)

        # 阈值分割
        binary_mask = self.threshold_image(image_array, threshold_percent)

        # 形态学后处理
        if dilation_iterations > 0 or erosion_iterations > 0:
            logger.info(f"    - 应用自定义形态学处理:")
            logger.info(f"      - 膨胀次数: {dilation_iterations}")
            logger.info(f"      - 腐蚀次数: {erosion_iterations}")
            logger.info(f"      - 核大小: {kernel_size}")
            binary_mask = self.morphological_processing(
                binary_mask,
                dilation_iterations=dilation_iterations,
                erosion_iterations=erosion_iterations,
                kernel_size=kernel_size
            )
            logger.info(f"    - 形态学处理后像素: {np.sum(binary_mask)}")

        # 提取中心线
        centerline = None
        main_centerline = None

        if extract_centerline and np.sum(binary_mask) > 100:
            logger.info("    - 提取血管中心线")
            centerline = self.extract_vessel_centerline(binary_mask)
            if np.sum(centerline) > 0:
                self.stats['centerline_generated'] += 1

                # 提取主干中心线
                if extract_longest_centerline:
                    logger.info(
                        f"    - 提取主干中心线（底部区域高度: {bottom_height_percent}%，排除边缘: {exclude_edge}，边缘宽度: {edge_percent}%）")
                    main_centerline = self.extract_longest_centerline_from_bottom(
                        centerline,
                        bottom_height_percent=bottom_height_percent,
                        exclude_edge=exclude_edge,
                        edge_percent=edge_percent
                    )
                    if np.sum(main_centerline) > 0:
                        self.stats['longest_centerline_generated'] += 1
                        logger.info(f"    - 主干中心线像素: {np.sum(main_centerline)}")
                    else:
                        logger.warning(f"    - 主干中心线提取失败，使用原始中心线")
                        main_centerline = centerline

        return binary_mask, centerline, main_centerline

    def save_result(self, mask, centerline, main_centerline, file_name, image_id):
        """保存结果图像"""
        # 保存血管标记图
        output_filename = f"{file_name}_{image_id}.png"
        output_path = os.path.join(self.output_dir, output_filename)

        os.makedirs(self.output_dir, exist_ok=True)

        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8)
        mask_img.save(output_path)
        logger.info(f"    - 血管标记图已保存: {output_path}")

        # 保存中心线叠加图
        if centerline is not None and np.sum(centerline) > 0:
            overlay = self.create_centerline_overlay(mask, centerline, file_name, image_id)

            overlay_filename = f"{file_name}_{image_id}_centerline.png"
            overlay_path = os.path.join(self.centerline_output_dir, overlay_filename)

            os.makedirs(self.centerline_output_dir, exist_ok=True)

            overlay_img = Image.fromarray(overlay)
            overlay_img.save(overlay_path)
            logger.info(f"    - 中心线叠加图已保存: {overlay_path}")

        # 保存主干中心线图
        if main_centerline is not None and np.sum(main_centerline) > 0:
            main_centerline_filename = f"{file_name}_{image_id}_main_centerline.png"
            main_centerline_path = os.path.join(self.longest_centerline_output_dir, main_centerline_filename)

            os.makedirs(self.longest_centerline_output_dir, exist_ok=True)

            main_centerline_uint8 = (main_centerline * 255).astype(np.uint8)
            main_centerline_img = Image.fromarray(main_centerline_uint8)
            main_centerline_img.save(main_centerline_path)
            logger.info(f"    - 主干中心线图已保存: {main_centerline_path}")

            # 保存主干中心线与血管的叠加图
            overlay_main = self.create_longest_centerline_overlay(mask, main_centerline, file_name, image_id)
            overlay_main_filename = f"{file_name}_{image_id}_main_overlay.png"
            overlay_main_path = os.path.join(self.overlay_output_dir, overlay_main_filename)

            os.makedirs(self.overlay_output_dir, exist_ok=True)
            overlay_main_img = Image.fromarray(overlay_main)
            overlay_main_img.save(overlay_main_path)
            logger.info(f"    - 主干中心线叠加图已保存: {overlay_main_path}")
            self.stats['overlay_generated'] += 1

        return output_path

    def run(self,
            threshold_percent=25,
            gaussian_sigma=1.5,
            dilation_iterations=2,
            erosion_iterations=1,
            kernel_size=5,
            extract_centerline=True,
            extract_longest_centerline=True,
            enable_adaptive_contrast=True,
            bottom_height_percent=5,
            exclude_edge=True,
            edge_percent=5):
        """
        运行分析流程 - 处理单帧DSA图像
        """
        logger.info("=" * 70)
        logger.info("DSA单帧图像血管标记程序开始运行")
        logger.info("=" * 70)
        logger.info(f"参数设置:")
        logger.info(f"  - 自适应对比度调整: {'启用' if enable_adaptive_contrast else '禁用'}")
        logger.info(f"  - 阈值: {threshold_percent}% (低于图像范围的此百分比标记为血管)")
        logger.info(f"  - 高斯滤波: sigma={gaussian_sigma}")
        logger.info(f"  - 形态学处理:")
        logger.info(f"      - 膨胀次数: {dilation_iterations}")
        logger.info(f"      - 腐蚀次数: {erosion_iterations}")
        logger.info(f"      - 核大小: {kernel_size}")
        logger.info(f"  - 中心线提取: {'开启' if extract_centerline else '关闭'}")
        logger.info(f"  - 主干中心线提取: {'开启' if extract_longest_centerline else '关闭'}")
        logger.info(f"  - 底部区域高度: {bottom_height_percent}% (用于选择主干起点)")
        logger.info(f"  - 排除边缘区域: {'是' if exclude_edge else '否'} (排除左右各{edge_percent}%区域)")
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

            # 查找该文件名的两个单帧图像文件
            image_files = self.find_image_files(file_name)

            # 处理图像0
            mask_0 = None
            centerline_0 = None
            main_centerline_0 = None
            if image_files['0']:
                try:
                    image_0 = self.read_single_frame_dicom(image_files['0'])
                    if image_0 is not None:
                        mask_0, centerline_0, main_centerline_0 = self.process_single_image(
                            image_0,
                            threshold_percent=threshold_percent,
                            gaussian_sigma=gaussian_sigma,
                            dilation_iterations=dilation_iterations,
                            erosion_iterations=erosion_iterations,
                            kernel_size=kernel_size,
                            extract_centerline=extract_centerline,
                            extract_longest_centerline=extract_longest_centerline,
                            enable_adaptive_contrast=enable_adaptive_contrast,
                            bottom_height_percent=bottom_height_percent,
                            exclude_edge=exclude_edge,
                            edge_percent=edge_percent
                        )
                        if mask_0 is not None:
                            self.save_result(mask_0, centerline_0, main_centerline_0, file_name, '0')
                            self.stats['processed'] += 1
                        else:
                            self.stats['failed'] += 1
                    else:
                        self.stats['failed'] += 1
                except Exception as e:
                    logger.error(f"    - 处理图像0时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['failed'] += 1
            else:
                self.stats['skipped_missing'] += 1

            # 处理图像1
            mask_1 = None
            centerline_1 = None
            main_centerline_1 = None
            if image_files['1']:
                try:
                    image_1 = self.read_single_frame_dicom(image_files['1'])
                    if image_1 is not None:
                        mask_1, centerline_1, main_centerline_1 = self.process_single_image(
                            image_1,
                            threshold_percent=threshold_percent,
                            gaussian_sigma=gaussian_sigma,
                            dilation_iterations=dilation_iterations,
                            erosion_iterations=erosion_iterations,
                            kernel_size=kernel_size,
                            extract_centerline=extract_centerline,
                            extract_longest_centerline=extract_longest_centerline,
                            enable_adaptive_contrast=enable_adaptive_contrast,
                            bottom_height_percent=bottom_height_percent,
                            exclude_edge=exclude_edge,
                            edge_percent=edge_percent
                        )
                        if mask_1 is not None:
                            self.save_result(mask_1, centerline_1, main_centerline_1, file_name, '1')
                            self.stats['processed'] += 1
                        else:
                            self.stats['failed'] += 1
                    else:
                        self.stats['failed'] += 1
                except Exception as e:
                    logger.error(f"    - 处理图像1时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['failed'] += 1
            else:
                self.stats['skipped_missing'] += 1

        # 输出统计信息
        logger.info("=" * 70)
        logger.info("处理完成！统计信息：")
        logger.info(f"总文件数: {self.stats['total']}")
        logger.info(f"成功处理图像数: {self.stats['processed']}")
        logger.info(f"成功生成中心线数: {self.stats['centerline_generated']}")
        logger.info(f"成功生成主干中心线数: {self.stats['longest_centerline_generated']}")
        logger.info(f"成功生成主干叠加图数: {self.stats['overlay_generated']}")
        logger.info(f"对比度调整次数: {self.stats['contrast_adjusted']}")
        logger.info(f"处理失败的图像数: {self.stats['failed']}")
        logger.info(f"跳过的文件数（无图像）: {self.stats['skipped_missing']}")
        logger.info("=" * 70)


def main():
    """主函数 - 参数配置"""
    # 基础路径配置
    EXCEL_PATH = r"D:\med_data\ai\classify_500.xlsx"
    IMAGE_BASE_DIR = r"D:\med_data\ai\train11"  # 修改为单帧图像目录
    OUTPUT_DIR = r"D:\med_data\ai\vessel\vessel_single_frame"
    CENTERLINE_OUTPUT_DIR = r"D:\med_data\ai\vessel\vessel_single_frame\center"
    MAIN_CENTERLINE_OUTPUT_DIR = r"D:\med_data\ai\vessel\vessel_single_frame\center_Long"
    OVERLAY_OUTPUT_DIR = r"D:\med_data\ai\vessel\vessel_single_frame\center1"

    # ========================================
    # 参数配置
    # ========================================

    # 自适应对比度调整开关
    ENABLE_ADAPTIVE_CONTRAST = True

    # ===== 阈值分割相关参数 =====
    THRESHOLD_PERCENT = 50  # 阈值百分比（低于图像范围的40%标记为血管）

    # 图像预处理参数
    GAUSSIAN_SIGMA = 1.5

    # 形态学处理参数（自定义膨胀和腐蚀）
    DILATION_ITERATIONS = 4
    EROSION_ITERATIONS = 3
    KERNEL_SIZE = 5

    # 中心线提取参数
    EXTRACT_CENTERLINE = True
    EXTRACT_MAIN_CENTERLINE = True
    BOTTOM_HEIGHT_PERCENT = 10  # 底部区域高度百分比

    # 边缘排除参数
    EXCLUDE_EDGE = True  # 是否排除边缘区域
    EDGE_PERCENT = 5  # 排除左右两侧各5%的区域

    print("\n" + "=" * 70)
    print("参数配置")
    print("=" * 70)
    print(f"自适应对比度调整: {'启用' if ENABLE_ADAPTIVE_CONTRAST else '禁用'}")
    print(f"阈值: {THRESHOLD_PERCENT}% (低于图像范围的此百分比标记为血管)")
    print(f"高斯滤波: sigma={GAUSSIAN_SIGMA}")
    print(f"形态学处理:")
    print(f"  - 膨胀次数: {DILATION_ITERATIONS}")
    print(f"  - 腐蚀次数: {EROSION_ITERATIONS}")
    print(f"  - 核大小: {KERNEL_SIZE}")
    print(f"中心线提取: {'开启' if EXTRACT_CENTERLINE else '关闭'}")
    print(f"主干中心线提取: {'开启' if EXTRACT_MAIN_CENTERLINE else '关闭'}")
    print(f"底部区域高度: {BOTTOM_HEIGHT_PERCENT}% (用于选择主干起点)")
    print(f"排除边缘区域: {'是' if EXCLUDE_EDGE else '否'} (排除左右各{EDGE_PERCENT}%区域)")
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
        gaussian_sigma=GAUSSIAN_SIGMA,
        dilation_iterations=DILATION_ITERATIONS,
        erosion_iterations=EROSION_ITERATIONS,
        kernel_size=KERNEL_SIZE,
        extract_centerline=EXTRACT_CENTERLINE,
        extract_longest_centerline=EXTRACT_MAIN_CENTERLINE,
        enable_adaptive_contrast=ENABLE_ADAPTIVE_CONTRAST,
        bottom_height_percent=BOTTOM_HEIGHT_PERCENT,
        exclude_edge=EXCLUDE_EDGE,
        edge_percent=EDGE_PERCENT
    )

    print("\n程序运行完毕！")
    print("按任意键退出...")
    input()


if __name__ == "__main__":
    main()