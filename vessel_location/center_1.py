import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import logging
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from scipy.spatial import KDTree
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DSAAnalyzer:
    """DSA图像分析器 - 血管中心线提取与动脉瘤分析"""

    def __init__(self, location_excel_path, image_base_dir, output_dir,
                 centerline_output_dir, longest_centerline_output_dir,
                 overlay_output_dir, aneurysm_mask_output_dir):
        self.location_excel_path = location_excel_path
        self.image_base_dir = image_base_dir
        self.output_dir = output_dir
        self.centerline_output_dir = centerline_output_dir
        self.longest_centerline_output_dir = longest_centerline_output_dir
        self.overlay_output_dir = overlay_output_dir
        self.aneurysm_mask_output_dir = aneurysm_mask_output_dir
        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'skipped_missing': 0,
            'sequences_found': 0,
            'sequences_processed': 0,
            'centerline_generated': 0,
            'longest_centerline_generated': 0,
            'overlay_generated': 0,
            'aneurysm_masks_generated': 0,
            'contrast_adjusted': 0
        }

        # 目标对比度范围
        self.target_min = 0
        self.target_max = 150
        self.target_range = self.target_max - self.target_min

        # 对比度调整阈值
        self.min_contrast_threshold = 100

        # 存储动脉瘤位置信息
        self.aneurysm_locations = {}

    def read_aneurysm_locations(self):
        """读取动脉瘤位置信息Excel文件"""
        try:
            location_df = pd.read_excel(self.location_excel_path)
            for idx, row in location_df.iterrows():
                case_id = str(row.iloc[0])  # 第一列：病历号
                x_ratio = float(row.iloc[1])  # 第二列：x坐标比例
                y_ratio = float(row.iloc[2])  # 第三列：y坐标比例
                self.aneurysm_locations[case_id] = (x_ratio, y_ratio)

            logger.info(f"从位置文件读取到 {len(self.aneurysm_locations)} 个动脉瘤位置信息")
            return list(self.aneurysm_locations.keys())
        except Exception as e:
            logger.error(f"读取位置文件失败: {e}")
            return []

    def find_sequence_files(self, file_name):
        """查找指定文件名的两个DSA序列文件"""
        sequences = {'0': None, '1': None}

        if not os.path.exists(self.image_base_dir):
            logger.warning(f"图像目录不存在: {self.image_base_dir}")
            return sequences

        # 构建两个可能的文件名
        file_0 = f"{file_name}"
        file_1 = f"{file_name}"

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
        """读取多帧DICOM文件并返回所有帧的像素数组"""
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

    def adaptive_contrast_adjustment(self, image_frames):
        """基于第5帧的自适应对比度调整"""
        if len(image_frames) < 5:
            logger.warning(f"    - 帧数不足5帧，无法使用第5帧进行对比度调整")
            return image_frames, {'adjusted': False, 'reason': 'insufficient_frames'}

        # 获取第5帧（索引4）
        frame5 = image_frames[4]

        # 计算第5帧的最小值和最大值
        frame5_min = frame5.min()
        frame5_max = frame5.max()
        frame5_range = frame5_max - frame5_min

        logger.info(f"    - 第5帧原始范围: [{frame5_min:.2f}, {frame5_max:.2f}], 范围: {frame5_range:.2f}")

        # 判断是否需要调整对比度
        if frame5_range < self.min_contrast_threshold:
            logger.info(f"    - 第5帧对比度不足 ({frame5_range:.2f} < {self.min_contrast_threshold})，进行对比度拉伸")

            # 线性拉伸到目标范围
            adjusted_frames = np.zeros_like(image_frames)
            for i in range(len(image_frames)):
                # 使用第5帧的统计信息进行归一化
                normalized = (image_frames[i] - frame5_min) / (frame5_range + 1e-6)
                normalized = normalized * self.target_range + self.target_min
                adjusted_frames[i] = np.clip(normalized, self.target_min, self.target_max)

            self.stats['contrast_adjusted'] += 1

            adjustment_info = {
                'adjusted': True,
                'original_min': frame5_min,
                'original_max': frame5_max,
                'original_range': frame5_range,
                'target_range': self.target_range
            }

            return adjusted_frames, adjustment_info
        else:
            logger.info(f"    - 第5帧对比度足够 ({frame5_range:.2f} >= {self.min_contrast_threshold})，保持原始图像")
            return image_frames, {'adjusted': False, 'original_range': frame5_range}

    def morphological_processing(self, binary_mask, dilation_iterations=0, erosion_iterations=0, kernel_size=3):
        """对二值掩膜进行自定义的形态学处理"""
        if dilation_iterations == 0 and erosion_iterations == 0:
            return binary_mask

        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 创建结构元素
        structure = np.ones((kernel_size, kernel_size), dtype=bool)
        processed_mask = binary_mask.copy().astype(np.uint8)

        # 先进行膨胀操作
        if dilation_iterations > 0:
            for i in range(dilation_iterations):
                processed_mask = ndimage.binary_dilation(processed_mask, structure=structure).astype(np.uint8)

        # 再进行腐蚀操作
        if erosion_iterations > 0:
            for i in range(erosion_iterations):
                processed_mask = ndimage.binary_erosion(processed_mask, structure=structure).astype(np.uint8)

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
        """找到骨架线的端点"""
        # 使用8邻域卷积计算每个点的邻居数量
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # 排除自身

        # 计算每个点的邻居数量
        neighbors = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

        # 端点：骨架点且邻居数为1
        endpoints = (skeleton > 0) & (neighbors == 1)

        return endpoints

    def filter_edge_points(self, centerline, edge_percent=5):
        """移除距离图像左右两侧边缘一定百分比范围内的所有中心线点"""
        h, w = centerline.shape

        # 计算要排除的边缘宽度
        edge_width = int(w * edge_percent / 100)

        # 创建掩码：只保留中心区域
        mask = np.zeros_like(centerline, dtype=bool)
        mask[:, edge_width:w - edge_width] = True

        # 应用掩码
        filtered_centerline = centerline.copy()
        filtered_centerline[~mask] = 0

        return filtered_centerline

    def find_bottom_candidates(self, skeleton, height_percent=5, exclude_edge=True, edge_percent=5):
        """找到图像底部的候选端点"""
        h, w = skeleton.shape

        # 先过滤掉边缘区域的中心线点
        if exclude_edge:
            skeleton = self.filter_edge_points(skeleton, edge_percent)

        endpoints = self.find_endpoints(skeleton)
        endpoint_coords = list(zip(*np.where(endpoints)))

        if not endpoint_coords:
            return []

        # 确定底部区域的范围
        bottom_threshold = h - int(h * height_percent / 100)

        # 筛选出在底部区域的端点
        bottom_candidates = []
        for y, x in endpoint_coords:
            if y >= bottom_threshold:
                # 计算从这个端点出发能到达的最高点
                highest_y = self.find_highest_point_from_start(skeleton, (y, x))
                bottom_candidates.append((y, x, highest_y))

        return bottom_candidates

    def find_highest_point_from_start(self, skeleton, start_point):
        """从起点出发，找到能到达的最高点（最小y坐标）"""
        from collections import deque

        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        queue = deque()
        queue.append(start_point)
        visited[start_point] = True

        highest_y = start_point[0]

        while queue:
            y, x = queue.popleft()

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
        """从候选底部端点中选择最佳起点"""
        if not candidates:
            return None

        # 按能到达的最高点排序
        candidates.sort(key=lambda c: c[2])
        best_highest_y = candidates[0][2]
        best_candidates = [c for c in candidates if c[2] == best_highest_y]

        if len(best_candidates) == 1:
            y, x, _ = best_candidates[0]
            return (y, x)
        else:
            # 多个端点都能到达最高点，选择最靠下的那个
            best_candidates.sort(key=lambda c: c[0], reverse=True)
            y, x, _ = best_candidates[0]
            return (y, x)

    def find_farthest_point_from_bottom(self, skeleton, bottom_point):
        """从底部端点出发，找到最远的点"""
        from collections import deque

        h, w = skeleton.shape
        distance = -np.ones((h, w), dtype=np.float32)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

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

        return farthest_point, distance

    def extract_longest_centerline_from_bottom(self, centerline, bottom_height_percent=5, exclude_edge=True,
                                               edge_percent=5):
        """从图像底部的候选端点中提取主干血管路径"""
        if np.sum(centerline) == 0:
            return np.zeros_like(centerline)

        # 如果启用边缘排除，先过滤掉边缘区域的中心线点
        if exclude_edge:
            filtered_centerline = self.filter_edge_points(centerline, edge_percent)
            if np.sum(filtered_centerline) < 10:
                filtered_centerline = centerline
        else:
            filtered_centerline = centerline

        # 找到底部区域的候选端点
        candidates = self.find_bottom_candidates(filtered_centerline, height_percent=bottom_height_percent,
                                                 exclude_edge=False)

        if not candidates:
            return centerline

        # 选择最佳起点
        bottom_point = self.select_best_bottom_point(candidates)
        if bottom_point is None:
            return centerline

        # 找到从底部出发的最远点
        farthest_point, distance_map = self.find_farthest_point_from_bottom(filtered_centerline, bottom_point)

        # 使用最短路径算法找到两点之间的路径
        cost = np.ones_like(filtered_centerline, dtype=np.float32) * 1e9
        cost[filtered_centerline > 0] = 1

        try:
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

            return main_path_mask

        except Exception as e:
            logger.error(f"    - 路径提取失败: {e}")
            return np.zeros_like(centerline)

    def find_nearest_point_on_centerline(self, centerline_points, target_point):
        """找到中心线上距离目标点最近的点"""
        if len(centerline_points) == 0:
            return None

        kdtree = KDTree(centerline_points)
        distance, index = kdtree.query(target_point)
        return centerline_points[index], distance

    def get_centerline_points(self, centerline_mask):
        """获取中心线上所有点的坐标"""
        points = np.where(centerline_mask > 0)
        return list(zip(points[0], points[1]))

    def get_perpendicular_direction(self, centerline_points, point_idx):
        """计算中心线上某点的垂直方向"""
        if point_idx == 0:
            # 如果是起点，使用下一个点
            p1 = np.array(centerline_points[point_idx])
            p2 = np.array(centerline_points[point_idx + 1])
        elif point_idx == len(centerline_points) - 1:
            # 如果是终点，使用前一个点
            p1 = np.array(centerline_points[point_idx - 1])
            p2 = np.array(centerline_points[point_idx])
        else:
            # 使用前后点计算切线方向
            p1 = np.array(centerline_points[point_idx - 1])
            p2 = np.array(centerline_points[point_idx + 1])

        # 计算切线方向
        tangent = p2 - p1
        tangent_norm = tangent / (np.linalg.norm(tangent) + 1e-6)

        # 垂直方向（在图像平面内）
        perpendicular = np.array([-tangent_norm[1], tangent_norm[0]])

        return perpendicular

    def create_aneurysm_mask(self, centerline_mask, aneurysm_center, height, width, y_ratio):
        """创建动脉瘤相关mask"""
        # 获取中心线上所有点
        centerline_points = self.get_centerline_points(centerline_mask)
        if len(centerline_points) == 0:
            logger.warning("    - 中心线上没有点，无法创建动脉瘤mask")
            return None, None, None

        # 找到距离动脉瘤中心最近的中心线点
        target_point = (aneurysm_center[0], aneurysm_center[1])
        nearest_point, distance = self.find_nearest_point_on_centerline(centerline_points, target_point)

        if nearest_point is None:
            logger.warning("    - 无法找到最近的动脉瘤中心点")
            return None, None, None

        logger.info(f"    - 最近的中心线点: {nearest_point}, 距离: {distance:.2f}像素")

        # 计算inlet点位置：y + (1-y) * 0.4
        inlet_y = int(aneurysm_center[0] + (height - aneurysm_center[0]) * 0.4)
        inlet_target = (inlet_y, aneurysm_center[1])

        # 找到inlet点最近的中心线点
        inlet_point, inlet_distance = self.find_nearest_point_on_centerline(centerline_points, inlet_target)

        if inlet_point is None:
            logger.warning("    - 无法找到inlet点")
            return None, None, None

        logger.info(f"    - Inlet点: {inlet_point}, 距离: {inlet_distance:.2f}像素")

        # 找到inlet点在中心线路径上的索引
        try:
            inlet_idx = centerline_points.index(inlet_point)
        except ValueError:
            # 如果找不到精确索引，使用最近的点
            inlet_idx = 0
            min_dist = float('inf')
            for i, pt in enumerate(centerline_points):
                dist = np.sqrt((pt[0] - inlet_point[0]) ** 2 + (pt[1] - inlet_point[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    inlet_idx = i

        # 计算垂直于中心线的方向
        perpendicular_dir = self.get_perpendicular_direction(centerline_points, inlet_idx)

        # 计算血管直径
        radius = self.calculate_vessel_radius(centerline_mask, inlet_point, perpendicular_dir)

        logger.info(f"    - 血管半径: {radius:.2f}像素")

        # 创建圆形mask
        circle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(circle_mask, (int(inlet_point[1]), int(inlet_point[0])),
                   int(max(1, radius)), 1, -1)

        return circle_mask, inlet_point, radius

    def calculate_vessel_radius(self, centerline_mask, center_point, perpendicular_dir):
        """计算血管半径（从中心点到血管边缘的距离）"""
        # 使用距离变换计算到最近背景像素的距离
        h, w = centerline_mask.shape
        y, x = center_point

        # 在垂直方向上搜索血管边缘
        radius = 0
        max_search = 100

        # 正方向搜索
        for r in range(1, max_search):
            ny = int(y + perpendicular_dir[0] * r)
            nx = int(x + perpendicular_dir[1] * r)
            if 0 <= ny < h and 0 <= nx < w:
                if centerline_mask[ny, nx] == 0:
                    radius = r
                    break
            else:
                radius = r
                break

        # 负方向搜索
        radius_neg = 0
        for r in range(1, max_search):
            ny = int(y - perpendicular_dir[0] * r)
            nx = int(x - perpendicular_dir[1] * r)
            if 0 <= ny < h and 0 <= nx < w:
                if centerline_mask[ny, nx] == 0:
                    radius_neg = r
                    break
            else:
                radius_neg = r
                break

        # 取两个方向的平均值
        if radius > 0 and radius_neg > 0:
            final_radius = (radius + radius_neg) / 2.0
        elif radius > 0:
            final_radius = radius
        elif radius_neg > 0:
            final_radius = radius_neg
        else:
            final_radius = 20  # 默认半径

        # 限制最大半径
        max_radius = 50
        if final_radius > max_radius:
            final_radius = max_radius

        return final_radius

    def create_longest_centerline_overlay(self, original_mask, longest_centerline):
        """创建最长中心线与血管提取图的叠加图像"""
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

    def create_final_overlay(self, original_mask, longest_centerline, circle_mask):
        """创建最终的叠加图像：血管mask + 中心线 + 圆形mask"""
        h, w = original_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 原始血管标记用绿色 (半透明)
        green_channel = (original_mask * 180).astype(np.uint8)
        overlay[:, :, 1] = green_channel

        # 最长中心线用亮红色
        red_channel = (longest_centerline * 255).astype(np.uint8)
        overlay[:, :, 0] = red_channel
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - longest_centerline * 0.7)

        # 蓝色通道用于背景
        blue_channel = (original_mask * 50).astype(np.uint8)
        overlay[:, :, 2] = blue_channel * (1 - longest_centerline)

        # 圆形mask用黄色（红色+绿色）
        if circle_mask is not None:
            circle_yellow = np.zeros_like(overlay)
            circle_yellow[:, :, 0] = circle_mask * 255  # 红色通道
            circle_yellow[:, :, 1] = circle_mask * 255  # 绿色通道

            # 叠加圆形mask（使用OR操作）
            overlay = np.maximum(overlay, circle_yellow)

        return overlay

    def save_aneurysm_related_results(self, circle_mask, original_mask, longest_centerline,
                                      inlet_point, radius, aneurysm_center,
                                      file_name, sequence_id, image_frames):
        """保存动脉瘤相关结果"""
        h, w = original_mask.shape

        # 1. 保存圆形mask
        if circle_mask is not None:
            mask_filename = f"{file_name}_{sequence_id}_aneurysm_mask.png"
            mask_path = os.path.join(self.aneurysm_mask_output_dir, mask_filename)
            os.makedirs(self.aneurysm_mask_output_dir, exist_ok=True)

            # 保存为PNG
            mask_img = Image.fromarray((circle_mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
            logger.info(f"    - 动脉瘤mask已保存: {mask_path}")
            self.stats['aneurysm_masks_generated'] += 1

            # 2. 保存叠加图像：血管mask + 中心线 + 圆形mask
            final_overlay = self.create_final_overlay(original_mask, longest_centerline, circle_mask)
            overlay_filename = f"{file_name}_{sequence_id}_final_overlay.png"
            overlay_path = os.path.join(self.overlay_output_dir, overlay_filename)
            os.makedirs(self.overlay_output_dir, exist_ok=True)

            overlay_img = Image.fromarray(final_overlay)
            overlay_img.save(overlay_path)
            logger.info(f"    - 最终叠加图已保存: {overlay_path}")

            # 3. 保存血管中心线叠加图像（不含圆形mask）
            centerline_overlay = self.create_longest_centerline_overlay(original_mask, longest_centerline)
            centerline_filename = f"{file_name}_{sequence_id}_centerline_overlay.png"
            centerline_path = os.path.join(self.centerline_output_dir, centerline_filename)
            os.makedirs(self.centerline_output_dir, exist_ok=True)

            centerline_img = Image.fromarray(centerline_overlay)
            centerline_img.save(centerline_path)
            logger.info(f"    - 中心线叠加图已保存: {centerline_path}")

            # 4. 额外保存DSA图像（使用第一帧作为背景）
            if image_frames is not None and len(image_frames) > 0:
                dsa_frame = image_frames[0].copy()
                # 归一化到0-255
                dsa_frame_min = dsa_frame.min()
                dsa_frame_max = dsa_frame.max()
                if dsa_frame_max > dsa_frame_min:
                    dsa_frame = ((dsa_frame - dsa_frame_min) / (dsa_frame_max - dsa_frame_min + 1e-6) * 255).astype(
                        np.uint8)
                else:
                    dsa_frame = np.zeros_like(dsa_frame, dtype=np.uint8)

                dsa_with_mask = cv2.cvtColor(dsa_frame, cv2.COLOR_GRAY2BGR)

                # 在DSA图像上叠加圆形mask
                if circle_mask is not None:
                    # 创建黄色半透明mask
                    mask_overlay = np.zeros_like(dsa_with_mask)
                    mask_overlay[:, :, 0] = circle_mask * 200  # 红色
                    mask_overlay[:, :, 1] = circle_mask * 200  # 绿色

                    # 半透明叠加
                    alpha = 0.4
                    dsa_with_mask = cv2.addWeighted(dsa_with_mask, 1 - alpha, mask_overlay, alpha, 0)

                # 标记inlet点（绿色）
                if inlet_point is not None:
                    cv2.circle(dsa_with_mask, (int(inlet_point[1]), int(inlet_point[0])),
                               6, (0, 255, 0), -1)
                    # 添加文字标签
                    cv2.putText(dsa_with_mask, 'Inlet',
                                (int(inlet_point[1]) - 20, int(inlet_point[0]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 标记动脉瘤中心点（蓝色）
                if aneurysm_center is not None:
                    cv2.circle(dsa_with_mask, (int(aneurysm_center[1]), int(aneurysm_center[0])),
                               6, (255, 0, 0), -1)
                    cv2.putText(dsa_with_mask, 'Aneurysm',
                                (int(aneurysm_center[1]) - 30, int(aneurysm_center[0]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                dsa_filename = f"{file_name}_{sequence_id}_dsa_with_mask.png"
                dsa_path = os.path.join(self.aneurysm_mask_output_dir, dsa_filename)
                dsa_img = Image.fromarray(dsa_with_mask)
                dsa_img.save(dsa_path)
                logger.info(f"    - DSA图像已保存: {dsa_path}")

    def process_sequence(self, image_frames, threshold_percent, frame_percent,
                         sequence_id, file_name, gaussian_sigma=0,
                         dilation_iterations=0, erosion_iterations=0, kernel_size=3,
                         extract_centerline=True,
                         extract_longest_centerline=True,
                         enable_adaptive_contrast=True,
                         bottom_height_percent=5,
                         exclude_edge=True,
                         edge_percent=5):
        """处理单个DSA序列"""
        if image_frames is None or len(image_frames) < 2:
            return None, None, None, None

        num_frames, height, width = image_frames.shape
        logger.info(f"  - 处理序列 {sequence_id}: 共 {num_frames} 帧")

        # 自适应对比度调整（基于第5帧）
        if enable_adaptive_contrast:
            image_frames, contrast_info = self.adaptive_contrast_adjustment(image_frames)

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
        if dilation_iterations > 0 or erosion_iterations > 0:
            marked_pixels = self.morphological_processing(
                marked_pixels,
                dilation_iterations=dilation_iterations,
                erosion_iterations=erosion_iterations,
                kernel_size=kernel_size
            )
            logger.info(f"    - 形态学处理后像素: {np.sum(marked_pixels)}")

        # 提取中心线
        centerline = None
        main_centerline = None

        if extract_centerline and np.sum(marked_pixels) > 100:
            logger.info("    - 提取血管中心线")
            centerline = self.extract_vessel_centerline(marked_pixels)
            if np.sum(centerline) > 0:
                self.stats['centerline_generated'] += 1

                # 提取主干中心线
                if extract_longest_centerline:
                    logger.info(f"    - 提取主干中心线")
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

        return marked_pixels, centerline, main_centerline, image_frames

    def apply_gaussian_filter(self, image_frames, sigma=1.0):
        """对图像序列应用高斯滤波"""
        from scipy.ndimage import gaussian_filter
        filtered_frames = np.zeros_like(image_frames)
        for i in range(len(image_frames)):
            filtered_frames[i] = gaussian_filter(image_frames[i], sigma=sigma)
        return filtered_frames

    def process_aneurysm_case(self, file_name, sequence_id, mask, main_centerline, image_frames):
        """处理单个病例的动脉瘤分析"""
        height, width = mask.shape

        # 获取动脉瘤位置信息
        x_ratio, y_ratio = self.aneurysm_locations[file_name]

        # 计算像素坐标
        aneurysm_x = int(x_ratio * width)
        aneurysm_y = int(y_ratio * height)
        aneurysm_center = (aneurysm_y, aneurysm_x)

        logger.info(f"    - 动脉瘤位置: ({aneurysm_x}, {aneurysm_y}) (比例: {x_ratio:.3f}, {y_ratio:.3f})")

        # 创建动脉瘤mask
        circle_mask, inlet_point, radius = self.create_aneurysm_mask(
            main_centerline, aneurysm_center, height, width, y_ratio
        )

        if circle_mask is not None and inlet_point is not None:
            # 保存结果
            self.save_aneurysm_related_results(
                circle_mask, mask, main_centerline,
                inlet_point, radius, aneurysm_center,
                file_name, sequence_id, image_frames
            )
            self.stats['overlay_generated'] += 1
        else:
            logger.warning(f"    - 动脉瘤mask创建失败")

    def run(self,
            threshold_percent=18,
            frame_percent=60,
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
        """运行分析流程"""
        logger.info("=" * 70)
        logger.info("DSA图像分析程序开始运行")
        logger.info("=" * 70)
        logger.info(f"参数设置:")
        logger.info(f"  - 自适应对比度调整: {'启用' if enable_adaptive_contrast else '禁用'}")
        logger.info(f"  - 阈值: {threshold_percent}%")
        logger.info(f"  - 使用前 {frame_percent}% 的帧")
        logger.info(f"  - 高斯滤波: sigma={gaussian_sigma}")
        logger.info(f"  - 形态学处理: 膨胀{dilation_iterations}次, 腐蚀{erosion_iterations}次")
        logger.info(f"  - 主干中心线提取: {'开启' if extract_longest_centerline else '关闭'}")
        logger.info(f"位置文件: {self.location_excel_path}")
        logger.info(f"图像目录: {self.image_base_dir}")
        logger.info("=" * 70)

        # 读取动脉瘤位置信息
        case_ids = self.read_aneurysm_locations()
        if not case_ids:
            logger.error("没有找到动脉瘤位置信息，程序退出")
            return

        self.stats['total'] = len(case_ids)
        logger.info(f"将处理 {self.stats['total']} 个病例")

        # 处理每个病例
        for idx, file_name in enumerate(tqdm(case_ids, desc="处理进度"), 1):
            logger.info(f"[{idx}/{self.stats['total']}] 处理: {file_name}")

            # 查找该文件名的两个序列文件
            sequence_files = self.find_sequence_files(file_name)

            # 处理序列0
            if sequence_files['0']:
                try:
                    frames_0 = self.read_multiframe_dicom(sequence_files['0'])
                    if frames_0 is not None:
                        mask_0, centerline_0, main_centerline_0, frames_0 = self.process_sequence(
                            frames_0,
                            threshold_percent,
                            frame_percent,
                            '0',
                            file_name,
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
                        if mask_0 is not None and main_centerline_0 is not None:
                            # 处理动脉瘤相关操作
                            self.process_aneurysm_case(file_name, '0', mask_0, main_centerline_0, frames_0)
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
            if sequence_files['1']:
                try:
                    frames_1 = self.read_multiframe_dicom(sequence_files['1'])
                    if frames_1 is not None:
                        mask_1, centerline_1, main_centerline_1, frames_1 = self.process_sequence(
                            frames_1,
                            threshold_percent,
                            frame_percent,
                            '1',
                            file_name,
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
                        if mask_1 is not None and main_centerline_1 is not None:
                            # 处理动脉瘤相关操作
                            self.process_aneurysm_case(file_name, '1', mask_1, main_centerline_1, frames_1)
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
        logger.info(f"总病例数: {self.stats['total']}")
        logger.info(f"成功处理病例数: {self.stats['processed']}")
        logger.info(f"找到的序列总数: {self.stats['sequences_found']}")
        logger.info(f"成功处理序列数: {self.stats['sequences_processed']}")
        logger.info(f"成功生成主干中心线数: {self.stats['longest_centerline_generated']}")
        logger.info(f"成功生成动脉瘤mask数: {self.stats['aneurysm_masks_generated']}")
        logger.info(f"对比度调整次数: {self.stats['contrast_adjusted']}")
        logger.info(f"处理失败的序列数: {self.stats['failed']}")
        logger.info(f"跳过的文件数: {self.stats['skipped_missing']}")
        logger.info("=" * 70)


def main():
    """主函数 - 参数配置"""
    # 基础路径配置
    LOCATION_EXCEL_PATH = r"D:\med_data\multi\location_vessel.xlsx"
    IMAGE_BASE_DIR = r"D:\med_data\ANY\0"
    OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1"
    CENTERLINE_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\center"
    MAIN_CENTERLINE_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\center_Long"
    OVERLAY_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\center1"
    ANEURYSM_MASK_OUTPUT_DIR = r"D:\med_data\ai\vessel_time_con1\aneurysm_masks"

    # ========================================
    # 参数配置
    # ========================================

    ENABLE_ADAPTIVE_CONTRAST = True
    THRESHOLD_PERCENT = 15
    FRAME_PERCENT = 55
    GAUSSIAN_SIGMA = 1.5

    DILATION_ITERATIONS = 4
    EROSION_ITERATIONS = 3
    KERNEL_SIZE = 5

    EXTRACT_CENTERLINE = True
    EXTRACT_MAIN_CENTERLINE = True
    BOTTOM_HEIGHT_PERCENT = 10
    EXCLUDE_EDGE = True
    EDGE_PERCENT = 5

    print("\n" + "=" * 70)
    print("参数配置")
    print("=" * 70)
    print(f"自适应对比度调整: {'启用' if ENABLE_ADAPTIVE_CONTRAST else '禁用'}")
    print(f"阈值: {THRESHOLD_PERCENT}%")
    print(f"使用前 {FRAME_PERCENT}% 的帧")
    print(f"高斯滤波: sigma={GAUSSIAN_SIGMA}")
    print(f"形态学处理: 膨胀{DILATION_ITERATIONS}次, 腐蚀{EROSION_ITERATIONS}次")
    print(f"主干中心线提取: {'开启' if EXTRACT_MAIN_CENTERLINE else '关闭'}")
    print(f"动脉瘤位置文件: {LOCATION_EXCEL_PATH}")
    print(f"图像目录: {IMAGE_BASE_DIR}")
    print("=" * 70)

    # 创建分析器并运行
    analyzer = DSAAnalyzer(
        LOCATION_EXCEL_PATH,
        IMAGE_BASE_DIR,
        OUTPUT_DIR,
        CENTERLINE_OUTPUT_DIR,
        MAIN_CENTERLINE_OUTPUT_DIR,
        OVERLAY_OUTPUT_DIR,
        ANEURYSM_MASK_OUTPUT_DIR
    )
    analyzer.run(
        threshold_percent=THRESHOLD_PERCENT,
        frame_percent=FRAME_PERCENT,
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
