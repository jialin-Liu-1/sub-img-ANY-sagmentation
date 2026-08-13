import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import warnings
import pydicom
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')


# ==============================
# AttentionMaskGenerator 类
# ==============================

class AttentionMaskGenerator(nn.Module):
    """生成注意力mask"""

    def __init__(self, image_size: Tuple[int, int] = (512, 512), min_radius_ratio: float = 0.06):
        super().__init__()
        self.H, self.W = image_size
        self.min_radius_ratio = min_radius_ratio
        self.device = torch.device('cpu')

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, height_ratio: torch.Tensor, width_ratio: torch.Tensor) -> torch.Tensor:
        B = height_ratio.shape[0]
        H, W = self.H, self.W

        batch_masks = []

        for b in range(B):
            h_ratio = height_ratio[b].item()
            w_ratio = width_ratio[b].item()

            # 应用最小半径比例限制
            if w_ratio < self.min_radius_ratio:
                w_ratio = self.min_radius_ratio

            # 计算中心高度位置
            y_center = int(h_ratio * (H - 1))
            y_center = max(0, min(y_center, H - 1))

            # 计算窗口宽度
            window_height = int(w_ratio * H)
            if window_height < 1:
                window_height = 1
            elif window_height > H:
                window_height = H

            # 计算矩形边界
            half_height = window_height // 2
            y_min = max(0, y_center - half_height)
            y_max = min(H - 1, y_center + half_height)

            # 如果窗口宽度为奇数，调整边界
            if window_height % 2 == 1:
                if y_min > 0:
                    y_min -= 1
                elif y_max < H - 1:
                    y_max += 1

            # 创建矩形mask
            attention_mask = torch.zeros(H, W, device=self.device)
            attention_mask[y_min:y_max + 1, :] = 1.0

            batch_masks.append(attention_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(batch_masks, dim=0)

    def generate_mask_numpy(self, height_ratio: float, width_ratio: float) -> np.ndarray:
        """生成numpy格式的mask（方便直接调用）"""
        height_tensor = torch.tensor([height_ratio], dtype=torch.float32)
        width_tensor = torch.tensor([width_ratio], dtype=torch.float32)

        mask_tensor = self.forward(height_tensor, width_tensor)
        return mask_tensor.squeeze().cpu().numpy()


# ==============================
# DSA图像处理类
# ==============================

class DSAImageProcessor:
    """处理DSA图像，应用注意力mask"""

    def __init__(self, dsa_dir: str = "D:\\med_data\\ai\\train11"):
        self.dsa_dir = Path(dsa_dir)
        self.dsa_files_index = self._build_file_index()

    def _build_file_index(self) -> Dict[str, Path]:
        """建立DSA文件索引，key为文件名（无扩展名），value为完整路径"""
        index = {}

        if not self.dsa_dir.exists():
            print(f"警告: DSA目录不存在 {self.dsa_dir}")
            return index

        for file_path in self.dsa_dir.iterdir():
            if file_path.is_file():
                # 获取文件名（无扩展名）
                stem = file_path.stem
                index[stem] = file_path

                # 如果文件无扩展名，也添加完整文件名
                if file_path.suffix == '':
                    index[file_path.name] = file_path

        print(f"建立DSA文件索引: 找到 {len(index)} 个文件")
        return index

    def find_dsa_by_filename(self, base_name: str) -> Optional[Path]:
        """根据基础文件名查找对应的DSA文件"""
        # 方法1: 直接匹配无扩展名
        if base_name in self.dsa_files_index:
            return self.dsa_files_index[base_name]

        # 方法2: 尝试常见扩展名
        common_extensions = ['', '.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.tif', '.tiff']
        for ext in common_extensions:
            full_name = f"{base_name}{ext}"
            if full_name in self.dsa_files_index:
                return self.dsa_files_index[full_name]

        # 方法3: 尝试提取数字部分进行模糊匹配
        import re
        match = re.search(r'(\d+)', base_name)
        if match:
            num_part = match.group(1)
            for key in self.dsa_files_index.keys():
                if num_part in key:
                    print(f"  模糊匹配: {base_name} -> {key}")
                    return self.dsa_files_index[key]

        return None

    def load_dsa_image(self, file_path: Path) -> Optional[np.ndarray]:
        """加载DSA图像（支持DICOM和常见图像格式）"""
        try:
            # 尝试加载DICOM
            if file_path.suffix.lower() in ['.dcm', '.dicom'] or file_path.suffix == '':
                try:
                    dicom_data = pydicom.dcmread(str(file_path), force=True)
                    image = dicom_data.pixel_array.astype(np.float32)
                except:
                    # 如果不是DICOM，尝试作为图像加载
                    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        return None
                    image = image.astype(np.float32)
            else:
                # 加载普通图像
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return None
                image = image.astype(np.float32)

            # 归一化到0-1
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)

            return image

        except Exception as e:
            print(f"  加载DSA图像失败 {file_path.name}: {e}")
            return None

    def apply_attention_mask(self, dsa_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        应用注意力mask
        mask=0的位置设为DSA图像最大值，mask=1的位置像素值不变
        """
        # 创建结果图像的副本
        result_image = dsa_image.copy()

        # 找到mask为0的区域
        zero_mask_area = mask < 0.5

        # 将这些区域设为DSA图像的最大值
        if np.any(zero_mask_area):
            max_value = dsa_image.max()
            result_image[zero_mask_area] = max_value

        return result_image

    def save_as_dicom(self, image: np.ndarray, original_filename: str, save_dir: Path):
        """保存为DICOM格式（无后缀）"""
        try:
            # 创建基本的DICOM数据集
            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds = pydicom.Dataset()
            ds.file_meta = file_meta

            # 设置必要的DICOM标签
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()

            # 图像数据
            ds.Rows, ds.Columns = image.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15

            # 转换图像到16位
            image_16bit = (image * 65535).astype(np.uint16)
            ds.PixelData = image_16bit.tobytes()

            # 保存（无后缀）
            base_name = os.path.splitext(original_filename)[0]
            save_path = save_dir / base_name  # 无后缀

            ds.save_as(str(save_path), write_like_original=False)
            return True

        except Exception as e:
            print(f"  保存DICOM失败: {e}")
            return False

    def save_as_png(self, image: np.ndarray, filename: str, save_dir: Path):
        """保存为PNG格式"""
        # 转换为8位
        image_8bit = (image * 255).astype(np.uint8)

        base_name = os.path.splitext(filename)[0]
        save_path = save_dir / f"{base_name}_attended.png"
        cv2.imwrite(str(save_path), image_8bit)
        return save_path


# ==============================
# MaskLocationExtractor（改进版）
# ==============================

class MaskLocationExtractor:
    """从mask图像中提取动脉瘤位置信息（使用最小外接圆确保完全覆盖）"""

    def __init__(self, mask_dir="D:\\med_data\\ai\\test2",
                 dsa_dir="D:\\med_data\\ai\\train11",
                 output_excel="D:\\med_data\\ai\\location4.xlsx",
                 visualization_dir="D:\\med_data\\ai\\mask4",
                 attention_dir="D:\\med_data\\ai\\attention_images"):

        self.mask_dir = Path(mask_dir)
        self.dsa_dir = Path(dsa_dir)
        self.output_excel = Path(output_excel)
        self.visualization_dir = Path(visualization_dir)
        self.attention_dir = Path(attention_dir)

        # 创建输出目录
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.attention_dir.mkdir(parents=True, exist_ok=True)

        # 创建注意力图像子目录
        self.attention_dicom_dir = self.attention_dir / "dicom"
        self.attention_png_dir = self.attention_dir / "png"
        self.attention_comparison_dir = self.attention_dir / "comparison"

        self.attention_dicom_dir.mkdir(exist_ok=True)
        self.attention_png_dir.mkdir(exist_ok=True)
        self.attention_comparison_dir.mkdir(exist_ok=True)

        # 存储结果
        self.location_data = []

        # 初始化注意力mask生成器
        self.mask_generator = AttentionMaskGenerator(image_size=(512, 512))

        # 初始化DSA处理器
        self.dsa_processor = DSAImageProcessor(dsa_dir)

        # 建立mask文件与DSA文件的对应关系
        self.mask_to_dsa_mapping = self._build_mask_to_dsa_mapping()

    def _build_mask_to_dsa_mapping(self) -> Dict[str, Optional[Path]]:
        """建立mask文件名到DSA文件路径的映射"""
        mapping = {}

        # 获取所有mask文件
        mask_files = list(self.mask_dir.glob("*.tif"))

        for mask_path in mask_files:
            mask_stem = mask_path.stem  # 无扩展名的mask文件名

            # 查找对应的DSA文件
            dsa_path = self.dsa_processor.find_dsa_by_filename(mask_stem)

            if dsa_path:
                mapping[mask_stem] = dsa_path
                print(f"匹配成功: {mask_stem}.tif <-> {dsa_path.name}")
            else:
                mapping[mask_stem] = None
                print(f"警告: 未找到 {mask_stem}.tif 对应的DSA文件")

        print(f"\n建立映射: 共 {len(mapping)} 个mask文件，{sum(1 for v in mapping.values() if v)} 个找到对应DSA")
        return mapping

    def find_mask_files(self):
        """查找所有TIF格式的mask文件"""
        mask_files = list(self.mask_dir.glob("*.tif"))
        print(f"找到 {len(mask_files)} 个TIF格式的mask文件")

        # 按文件名排序
        mask_files.sort()

        # 显示前10个文件
        print("前10个文件:")
        for i, file in enumerate(mask_files[:10]):
            match_status = "✓" if file.stem in self.mask_to_dsa_mapping and self.mask_to_dsa_mapping[file.stem] else "✗"
            print(f"  {i + 1}. {file.name} [{match_status}]")

        return mask_files

    def load_mask(self, mask_path):
        """加载mask图像并进行预处理"""
        try:
            # 使用cv2读取TIF文件
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"警告: 无法读取 {mask_path.name}")
                return None

            # 转换为0-255范围
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)

            # 二值化（确保是0和255）
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # 转换为0-1
            mask_normalized = (mask_binary > 127).astype(np.uint8)

            return mask_normalized

        except Exception as e:
            print(f"加载mask {mask_path.name} 失败: {e}")
            return None

    def find_main_aneurysm_region(self, mask):
        """找到主要的动脉瘤区域"""
        try:
            # 标记连通区域
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

            if num_labels == 0:
                return None

            # 获取所有区域属性
            regions = regionprops(labeled_mask)

            if not regions:
                return None

            # 选择面积最大的区域（假设最大的区域是动脉瘤）
            largest_region = max(regions, key=lambda r: r.area)

            # 过滤掉太小的区域
            if largest_region.area < 10:
                return None

            return largest_region

        except Exception as e:
            print(f"查找主要动脉瘤区域失败: {e}")
            return None

    def calculate_minimum_enclosing_circle(self, region):
        """计算最小外接圆（确保完全覆盖动脉瘤）"""
        try:
            # 获取动脉瘤的所有像素坐标
            coords = region.coords  # [y, x]格式

            if len(coords) < 3:
                # 如果点太少，使用边界框计算
                min_row, min_col, max_row, max_col = region.bbox
                center_x = (min_col + max_col) / 2
                center_y = (min_row + max_row) / 2
                radius = max((max_row - min_row), (max_col - min_col)) / 2
                return center_y, center_x, radius

            # 使用OpenCV的minEnclosingCircle计算最小外接圆
            points = coords[:, [1, 0]].astype(np.float32)  # 转换为[x, y]格式

            (center_x, center_y), radius = cv2.minEnclosingCircle(points)

            # 添加10%的余量以确保完全覆盖
            radius = radius * 1.1

            return center_y, center_x, radius

        except Exception as e:
            print(f"计算最小外接圆失败: {e}")
            # 备用方案：使用边界框计算
            min_row, min_col, max_row, max_col = region.bbox
            center_x = (min_col + max_col) / 2
            center_y = (min_row + max_row) / 2
            radius = max((max_row - min_row), (max_col - min_col)) / 2 * 1.2
            return center_y, center_x, radius

    def create_circle_mask(self, image_shape, center_y, center_x, radius):
        """创建圆形mask"""
        h, w = image_shape
        y_coords, x_coords = np.ogrid[:h, :w]

        # 计算距离
        dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # 创建圆形mask
        circle_mask = (dist <= radius).astype(np.uint8)

        return circle_mask

    def check_coverage_and_adjust(self, original_mask, center_y, center_x, radius):
        """检查覆盖率并调整半径以确保完全覆盖"""
        # 创建初始圆
        h, w = original_mask.shape
        initial_circle = self.create_circle_mask((h, w), center_y, center_x, radius)

        # 检查覆盖率
        aneurysm_area = np.sum(original_mask > 0)
        overlap_area = np.sum(np.logical_and(original_mask > 0, initial_circle > 0))
        initial_coverage = overlap_area / aneurysm_area if aneurysm_area > 0 else 0

        # 如果覆盖率已经很高（>95%），直接返回
        if initial_coverage > 0.95:
            return center_y, center_x, radius, initial_coverage

        # 如果覆盖率不够，逐步扩大半径
        max_radius = min(h, w) / 2  # 最大半径为图像尺寸的一半
        adjusted_radius = radius

        for i in range(20):  # 最多尝试20次
            if adjusted_radius >= max_radius:
                break

            # 稍微增加半径
            adjusted_radius = adjusted_radius * 1.05

            # 创建新的圆
            new_circle = self.create_circle_mask((h, w), center_y, center_x, adjusted_radius)

            # 计算新的覆盖率
            new_overlap = np.sum(np.logical_and(original_mask > 0, new_circle > 0))
            new_coverage = new_overlap / aneurysm_area if aneurysm_area > 0 else 0

            # 如果达到满意覆盖率或不能再提高，停止
            if new_coverage > 0.99 or (i > 5 and new_coverage - initial_coverage < 0.01):
                return center_y, center_x, adjusted_radius, new_coverage

        return center_y, center_x, adjusted_radius, new_coverage

    def process_attention_image(self, mask_stem: str, height_ratio: float, radius_ratio: float):
        """
        根据mask文件名处理对应的DSA图像

        Args:
            mask_stem: mask文件名（无扩展名）
            height_ratio: 高度比例
            radius_ratio: 半径比例
        """
        print(f"\n  处理注意力图像: {mask_stem}")

        # 1. 查找对应的DSA文件
        dsa_path = self.mask_to_dsa_mapping.get(mask_stem)

        if dsa_path is None:
            print(f"  警告: 未找到 {mask_stem} 对应的DSA文件，跳过注意力图像生成")
            return None

        print(f"  找到对应DSA文件: {dsa_path.name}")

        # 2. 加载DSA图像
        dsa_image = self.dsa_processor.load_dsa_image(dsa_path)
        if dsa_image is None:
            print(f"  警告: 无法加载DSA图像 {dsa_path.name}，跳过注意力图像生成")
            return None

        # 3. 生成注意力mask
        attention_mask = self.mask_generator.generate_mask_numpy(height_ratio, radius_ratio)

        # 4. 应用注意力mask
        attended_image = self.dsa_processor.apply_attention_mask(dsa_image, attention_mask)

        # 5. 保存DICOM格式（无后缀）
        dicom_success = self.dsa_processor.save_as_dicom(
            attended_image,
            dsa_path.name,
            self.attention_dicom_dir
        )
        if dicom_success:
            print(f"    保存DICOM: {self.attention_dicom_dir / mask_stem}")

        # 6. 保存PNG格式
        png_path = self.dsa_processor.save_as_png(
            attended_image,
            dsa_path.name,
            self.attention_png_dir
        )
        print(f"    保存PNG: {png_path}")

        # 7. 创建对比图像（原始DSA vs 注意力mask vs 应用后）
        self.create_comparison_image(
            dsa_image,
            attention_mask,
            attended_image,
            dsa_path.name,
            height_ratio,
            radius_ratio
        )

        return attended_image

    def create_comparison_image(self, dsa_image, attention_mask, attended_image,
                                filename, height_ratio, radius_ratio):
        """创建对比图像：原始DSA、注意力mask、应用后"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 原始DSA
            axes[0].imshow(dsa_image, cmap='gray')
            axes[0].set_title(f'原始DSA图像\n{filename}')
            axes[0].axis('off')

            # 注意力mask
            axes[1].imshow(attention_mask, cmap='gray')
            axes[1].set_title(f'注意力mask\n高度比例={height_ratio:.3f}, 半径比例={radius_ratio:.3f}')
            axes[1].axis('off')

            # 应用后的图像
            axes[2].imshow(attended_image, cmap='gray')
            axes[2].set_title('应用注意力后\n(mask=0区域设为最大值)')
            axes[2].axis('off')

            plt.suptitle(f'注意力mask应用效果对比 - {filename}', fontsize=14)
            plt.tight_layout()

            # 保存对比图像
            base_name = os.path.splitext(filename)[0]
            save_path = self.attention_comparison_dir / f"{base_name}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    保存对比图: {save_path}")

        except Exception as e:
            print(f"    创建对比图像失败: {e}")

    def visualize_results(self, original_mask, circle_mask, filename,
                          center_y, center_x, radius, region_bbox,
                          height_ratio, radius_ratio, dsa_attended_image=None):
        """可视化结果并保存（包含注意力mask应用效果）"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))

            # 1. 原始mask
            axes[0, 0].imshow(original_mask, cmap='gray')
            axes[0, 0].set_title(f'原始mask: {filename}')
            axes[0, 0].axis('off')

            # 标记边界框
            min_row, min_col, max_row, max_col = region_bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=2)
            axes[0, 0].add_patch(rect)

            # 2. 最小外接圆分析
            axes[0, 1].imshow(original_mask, cmap='gray')
            axes[0, 1].set_title('最小外接圆分析')
            axes[0, 1].axis('off')

            # 绘制圆心
            axes[0, 1].plot(center_x, center_y, 'go', markersize=8, linewidth=2, label='圆心')

            # 绘制圆形边界
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center_x + radius * np.cos(theta)
            circle_y = center_y + radius * np.sin(theta)
            axes[0, 1].plot(circle_x, circle_y, 'g-', linewidth=2, label=f'半径={radius:.1f}')

            # 绘制边界框
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=2, linestyle='--',
                                 label='边界框')
            axes[0, 1].add_patch(rect)

            axes[0, 1].legend(loc='upper right')

            # 3. 生成的圆
            axes[0, 2].imshow(circle_mask, cmap='gray')
            axes[0, 2].set_title(f'最小外接圆 (半径={radius:.1f}像素)')
            axes[0, 2].axis('off')
            axes[0, 2].plot(center_x, center_y, 'go', markersize=8, linewidth=2)

            # 4. 重叠图像（彩色）
            overlap = np.zeros((*original_mask.shape, 3), dtype=np.uint8)

            # 动脉瘤区域：绿色
            aneurysm_mask = original_mask > 0
            overlap[aneurysm_mask, 1] = 255  # 绿色

            # 圆区域：红色（半透明）
            circle_area = circle_mask > 0
            overlap[circle_area, 0] = 128  # 红色

            # 重叠部分：黄色
            overlap_area = np.logical_and(aneurysm_mask, circle_area)
            overlap[overlap_area, 0] = 255  # 红色
            overlap[overlap_area, 1] = 255  # 绿色
            overlap[overlap_area, 2] = 0  # 无蓝色

            axes[0, 3].imshow(overlap)
            axes[0, 3].set_title('重叠图像 (绿:动脉瘤, 红:圆, 黄:重叠)')
            axes[0, 3].axis('off')

            # 5. 覆盖率分析
            axes[1, 0].imshow(original_mask, cmap='gray')

            # 标记未覆盖的区域
            uncovered = np.logical_and(aneurysm_mask, ~circle_area)
            if np.any(uncovered):
                # 找到未覆盖区域的轮廓
                uncovered_uint8 = uncovered.astype(np.uint8) * 255
                contours, _ = cv2.findContours(uncovered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 5:  # 只显示面积大于5的区域
                        contour = contour.squeeze()
                        axes[1, 0].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.7)

            axes[1, 0].set_title('未覆盖区域分析 (红色:未覆盖)')
            axes[1, 0].axis('off')

            # 6. 注意力mask示意图
            attention_mask = self.mask_generator.generate_mask_numpy(height_ratio, radius_ratio)
            axes[1, 1].imshow(attention_mask, cmap='gray')
            axes[1, 1].set_title(f'注意力mask\n高度比例={height_ratio:.3f}\n半径比例={radius_ratio:.3f}')
            axes[1, 1].axis('off')

            # 7. 应用注意力后的DSA图像（如果有）
            if dsa_attended_image is not None:
                axes[1, 2].imshow(dsa_attended_image, cmap='gray')
                axes[1, 2].set_title('应用注意力后的DSA图像')
                axes[1, 2].axis('off')
            else:
                axes[1, 2].axis('off')
                axes[1, 2].set_title('无DSA图像')

            # 8. 统计信息
            axes[1, 3].axis('off')

            # 计算统计信息
            aneurysm_area = np.sum(aneurysm_mask)
            circle_area_sum = np.sum(circle_area)
            overlap_area_sum = np.sum(overlap_area)

            coverage = overlap_area_sum / aneurysm_area if aneurysm_area > 0 else 0
            uncovered_area = aneurysm_area - overlap_area_sum

            bbox_width = max_col - min_col
            bbox_height = max_row - min_row

            # 获取对应DSA文件状态
            dsa_status = "有对应DSA" if self.mask_to_dsa_mapping.get(filename.replace('.tif', '')) else "无对应DSA"
            attention_status = "已生成" if dsa_attended_image is not None else "未生成"

            info_text = f"文件: {filename}\n"
            info_text += "=" * 40 + "\n"
            info_text += f"圆心: ({center_x:.1f}, {center_y:.1f})\n"
            info_text += f"圆半径: {radius:.1f}像素\n"
            info_text += f"边界框: {bbox_width:.1f}×{bbox_height:.1f}\n"
            info_text += f"动脉瘤面积: {aneurysm_area}像素\n"
            info_text += f"圆面积: {circle_area_sum}像素\n"
            info_text += f"重叠面积: {overlap_area_sum}像素\n"
            info_text += f"未覆盖面积: {uncovered_area}像素\n"
            info_text += f"覆盖率: {coverage:.1%}\n"
            info_text += f"高度比例: {height_ratio:.3f}\n"
            info_text += f"半径比例: {radius_ratio:.3f}\n"
            info_text += f"DSA状态: {dsa_status}\n"
            info_text += f"注意力图像: {attention_status}"

            if uncovered_area > 0:
                info_text += f"\n未覆盖率: {(1 - coverage):.1%}"

            axes[1, 3].text(0.1, 0.5, info_text, transform=axes[1, 3].transAxes,
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f'动脉瘤位置提取与注意力应用: {filename}', fontsize=16, y=1.02)
            plt.tight_layout()

            # 保存可视化结果
            output_name = filename.replace('.tif', '_full_analysis.png')
            output_path = self.visualization_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 保存重叠图像单独文件
            overlap_name = filename.replace('.tif', '_overlap.png')
            overlap_path = self.visualization_dir / overlap_name
            cv2.imwrite(str(overlap_path), overlap)

            return coverage, uncovered_area

        except Exception as e:
            print(f"可视化保存失败: {e}")
            return 0.0, 0

    def process_single_mask(self, mask_path):
        """处理单个mask文件 - 使用最小外接圆法"""
        filename = mask_path.name
        mask_stem = mask_path.stem
        print(f"\n处理: {filename}")

        # 1. 加载mask
        mask = self.load_mask(mask_path)
        if mask is None:
            return None

        # 获取图像尺寸
        h, w = mask.shape

        # 2. 找到主要动脉瘤区域
        region = self.find_main_aneurysm_region(mask)

        if region is None:
            print(f"  {filename}: 未找到有效的动脉瘤区域")
            return None

        # 3. 计算最小外接圆
        center_y, center_x, radius = self.calculate_minimum_enclosing_circle(region)

        print(f"  初始圆心: ({center_x:.1f}, {center_y:.1f})")
        print(f"  初始半径: {radius:.1f}像素")

        # 4. 检查并调整半径以确保完全覆盖
        center_y, center_x, final_radius, coverage = self.check_coverage_and_adjust(
            mask, center_y, center_x, radius
        )

        print(f"  调整后圆心: ({center_x:.1f}, {center_y:.1f})")
        print(f"  调整后半径: {final_radius:.1f}像素")
        print(f"  预计覆盖率: {coverage:.1%}")

        # 5. 创建圆形mask
        circle_mask = self.create_circle_mask((h, w), center_y, center_x, final_radius)

        # 6. 计算归一化参数
        # 高度比例：从上到下的位置，0=顶部，1=底部
        height_ratio = center_y / h

        # 半径比例：相对于图像尺寸的比例
        max_dim = max(h, w)
        radius_ratio = final_radius / (max_dim / 2)  # 最大半径为图像尺寸的一半

        # 设置最小半径比例为0.06
        MIN_RADIUS_RATIO = 0.06
        if radius_ratio < MIN_RADIUS_RATIO:
            print(
                f"  警告: 计算出的半径比例 {radius_ratio:.3f} 小于最小值 {MIN_RADIUS_RATIO}，将设为 {MIN_RADIUS_RATIO}")
            radius_ratio = MIN_RADIUS_RATIO

        # 限制在0-1范围内
        height_ratio = max(0.0, min(1.0, height_ratio))
        radius_ratio = max(0.0, min(1.0, radius_ratio))

        # 7. 根据文件名匹配处理对应的DSA图像
        dsa_attended_image = self.process_attention_image(mask_stem, height_ratio, radius_ratio)

        # 8. 可视化并保存（包含注意力mask应用效果）
        bbox = region.bbox
        final_coverage, uncovered_area = self.visualize_results(
            mask, circle_mask, filename,
            center_y, center_x, final_radius, bbox,
            height_ratio, radius_ratio, dsa_attended_image
        )

        # 9. 记录结果
        result = {
            'filename': mask_stem,
            'height_ratio': height_ratio,
            'radius_ratio': radius_ratio,
            'center_x': center_x,
            'center_y': center_y,
            'pixel_radius': final_radius,
            'initial_radius': radius,
            'coverage': final_coverage,
            'uncovered_area': uncovered_area,
            'aneurysm_area': region.area,
            'image_width': w,
            'image_height': h,
            'bbox_width': bbox[3] - bbox[1],
            'bbox_height': bbox[2] - bbox[0],
            'has_dsa': self.mask_to_dsa_mapping.get(mask_stem) is not None,
            'attention_image_generated': dsa_attended_image is not None
        }

        print(f"  高度比例: {height_ratio:.3f}")
        print(f"  半径比例: {radius_ratio:.3f}")
        print(f"  实际覆盖率: {final_coverage:.1%}")
        print(f"  未覆盖面积: {uncovered_area}像素")
        print(f"  存在对应DSA: {'是' if result['has_dsa'] else '否'}")
        print(f"  注意力图像生成: {'成功' if dsa_attended_image is not None else '失败'}")

        if final_coverage < 0.95:
            print(f"  ⚠️ 警告: 覆盖率较低 ({final_coverage:.1%})")

        return result

    def process_all_masks(self):
        """处理所有mask文件"""
        print("开始提取动脉瘤位置信息（最小外接圆法）...")
        print("=" * 60)

        # 查找所有mask文件
        mask_files = self.find_mask_files()

        if not mask_files:
            print("错误: 未找到mask文件")
            return False

        # 处理每个文件
        successful_count = 0
        failed_count = 0
        low_coverage_count = 0
        has_dsa_count = 0
        attention_generated_count = 0

        for mask_path in mask_files:
            result = self.process_single_mask(mask_path)

            if result:
                self.location_data.append(result)
                successful_count += 1

                # 统计低覆盖率情况
                if result['coverage'] < 0.95:
                    low_coverage_count += 1

                # 统计有对应DSA的文件数
                if result['has_dsa']:
                    has_dsa_count += 1

                # 统计成功生成注意力图像的数量
                if result['attention_image_generated']:
                    attention_generated_count += 1
            else:
                failed_count += 1

        print(f"\n处理完成!")
        print(f"成功处理: {successful_count} 个文件")
        print(f"处理失败: {failed_count} 个文件")
        if successful_count > 0:
            print(f"覆盖率<95%: {low_coverage_count} 个 ({low_coverage_count / successful_count * 100:.1f}%)")
            print(f"有对应DSA: {has_dsa_count} 个 ({has_dsa_count / successful_count * 100:.1f}%)")
            print(
                f"生成注意力图像: {attention_generated_count} 个 ({attention_generated_count / successful_count * 100:.1f}%)")

        return successful_count > 0

    def save_to_excel(self):
        """保存结果到Excel文件"""
        if not self.location_data:
            print("错误: 没有数据可保存")
            return False

        try:
            # 创建DataFrame
            df = pd.DataFrame(self.location_data)

            # 重新排列列，只保留需要的列
            output_df = df[['filename', 'height_ratio', 'radius_ratio', 'has_dsa', 'attention_image_generated']].copy()

            # 保存到Excel
            output_df.to_excel(self.output_excel, index=False)

            print(f"\n位置信息已保存到: {self.output_excel}")
            print(f"总记录数: {len(output_df)}")
            print(f"有对应DSA: {output_df['has_dsa'].sum()} 个")
            print(f"成功生成注意力图像: {output_df['attention_image_generated'].sum()} 个")

            # 显示前几行
            print("\n前10条记录:")
            print(output_df.head(10))

            # 保存完整数据到CSV（包含更多信息）
            full_csv_path = self.output_excel.with_suffix('.csv')
            df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
            print(f"完整数据已保存到: {full_csv_path}")

            return True

        except Exception as e:
            print(f"保存Excel文件失败: {e}")
            return False

    def generate_summary_report(self):
        """生成统计报告"""
        if not self.location_data:
            print("没有数据生成报告")
            return

        df = pd.DataFrame(self.location_data)

        print("\n" + "=" * 60)
        print("位置信息统计报告（最小外接圆法）")
        print("=" * 60)

        print(f"总样本数: {len(df)}")
        print(f"平均高度比例: {df['height_ratio'].mean():.3f} (±{df['height_ratio'].std():.3f})")
        print(f"平均半径比例: {df['radius_ratio'].mean():.3f} (±{df['radius_ratio'].std():.3f})")
        print(f"平均覆盖率: {df['coverage'].mean():.1%} (±{df['coverage'].std():.1%})")
        print(f"有对应DSA: {df['has_dsa'].sum()} 个 ({df['has_dsa'].mean() * 100:.1f}%)")
        print(
            f"生成注意力图像: {df['attention_image_generated'].sum()} 个 ({df['attention_image_generated'].mean() * 100:.1f}%)")

        # 覆盖率分布统计
        coverage_stats = {
            '优秀 (≥95%)': len(df[df['coverage'] >= 0.95]),
            '良好 (85-95%)': len(df[(df['coverage'] >= 0.85) & (df['coverage'] < 0.95)]),
            '一般 (75-85%)': len(df[(df['coverage'] >= 0.75) & (df['coverage'] < 0.85)]),
            '较差 (<75%)': len(df[df['coverage'] < 0.75])
        }

        print("\n覆盖率分布:")
        for category, count in coverage_stats.items():
            percentage = count / len(df) * 100
            print(f"  {category}: {count} 个 ({percentage:.1f}%)")

        # 生成可视化图表
        self.create_statistics_plots(df)

    def create_statistics_plots(self, df):
        """创建统计图表"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. 高度比例分布
            axes[0, 0].hist(df['height_ratio'], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(df['height_ratio'].mean(), color='red', linestyle='--',
                               label=f'均值: {df["height_ratio"].mean():.3f}')
            axes[0, 0].set_xlabel('高度比例 (0=顶部, 1=底部)')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].set_title('高度比例分布')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 半径比例分布
            axes[0, 1].hist(df['radius_ratio'], bins=20, edgecolor='black', alpha=0.7, color='orange')
            axes[0, 1].axvline(df['radius_ratio'].mean(), color='red', linestyle='--',
                               label=f'均值: {df["radius_ratio"].mean():.3f}')
            axes[0, 1].set_xlabel('半径比例')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].set_title('半径比例分布')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 覆盖率分布
            axes[0, 2].hist(df['coverage'], bins=20, edgecolor='black', alpha=0.7, color='green')
            axes[0, 2].axvline(df['coverage'].mean(), color='red', linestyle='--',
                               label=f'均值: {df["coverage"].mean():.1%}')
            axes[0, 2].axvline(0.95, color='blue', linestyle=':', label='95%阈值')
            axes[0, 2].set_xlabel('覆盖率')
            axes[0, 2].set_ylabel('频数')
            axes[0, 2].set_title('覆盖率分布')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # 4. 高度 vs 半径散点图（颜色表示覆盖率）
            scatter = axes[1, 0].scatter(df['height_ratio'], df['radius_ratio'],
                                         c=df['coverage'], cmap='RdYlGn', alpha=0.7, s=50,
                                         vmin=0.7, vmax=1.0)
            axes[1, 0].set_xlabel('高度比例')
            axes[1, 0].set_ylabel('半径比例')
            axes[1, 0].set_title('高度 vs 半径 (颜色=覆盖率)')
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('覆盖率')
            axes[1, 0].grid(True, alpha=0.3)

            # 5. DSA匹配统计
            dsa_counts = [df['has_dsa'].sum(), len(df) - df['has_dsa'].sum()]
            axes[1, 1].bar(['有对应DSA', '无对应DSA'], dsa_counts, color=['green', 'red'], alpha=0.7)
            axes[1, 1].set_ylabel('样本数')
            axes[1, 1].set_title('DSA文件匹配统计')
            for i, v in enumerate(dsa_counts):
                axes[1, 1].text(i, v + 0.1, str(v), ha='center', va='bottom')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            # 6. 注意力图像生成统计
            attention_counts = [df['attention_image_generated'].sum(), len(df) - df['attention_image_generated'].sum()]
            axes[1, 2].bar(['生成成功', '生成失败'], attention_counts, color=['green', 'red'], alpha=0.7)
            axes[1, 2].set_ylabel('样本数')
            axes[1, 2].set_title('注意力图像生成统计')
            for i, v in enumerate(attention_counts):
                axes[1, 2].text(i, v + 0.1, str(v), ha='center', va='bottom')
            axes[1, 2].grid(True, alpha=0.3, axis='y')

            plt.suptitle('动脉瘤位置信息统计 - 最小外接圆法', fontsize=16, y=1.02)
            plt.tight_layout()

            # 保存图表
            stats_path = self.visualization_dir / 'statistics_summary.png'
            plt.savefig(stats_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n统计图表已保存到: {stats_path}")

        except Exception as e:
            print(f"生成统计图表失败: {e}")


def main():
    """主函数"""
    print("动脉瘤位置信息提取程序 - 最小外接圆法（确保完全覆盖）")
    print("=" * 60)

    # 配置参数
    mask_dir = "D:\\med_data\\ai\\translate\\train_all_trans(2)"  # mask图像目录
    dsa_dir = "D:\\med_data\\ai\\translate\\train_all_trans(1)"  # DSA图像目录
    output_excel = "D:\\med_data\\ai\\translate\\location_trans(2).xlsx"
    visualization_dir = "D:\\med_data\\ai\\translate\\attention_images\\mask4"
    attention_dir = "D:\\med_data\\ai\\translate\\attention_images"  # 注意力图像保存目录

    # 创建提取器
    extractor = MaskLocationExtractor(
        mask_dir=mask_dir,
        dsa_dir=dsa_dir,
        output_excel=output_excel,
        visualization_dir=visualization_dir,
        attention_dir=attention_dir
    )

    try:
        # 处理所有mask文件
        success = extractor.process_all_masks()

        if success:
            # 保存到Excel
            extractor.save_to_excel()

            # 生成统计报告
            extractor.generate_summary_report()

            print("\n" + "=" * 60)
            print("处理完成!")
            print("=" * 60)
            print(f"位置表格: {output_excel}")
            print(f"可视化结果: {visualization_dir}")
            print(f"注意力图像: {attention_dir}")
            print(f"  - DICOM格式: {extractor.attention_dicom_dir}")
            print(f"  - PNG格式: {extractor.attention_png_dir}")
            print(f"  - 对比图像: {extractor.attention_comparison_dir}")

            # 显示关键统计数据
            df = pd.DataFrame(extractor.location_data)
            if len(df) > 0:
                high_coverage = len(df[df['coverage'] >= 0.95])
                has_dsa = df['has_dsa'].sum()
                attention_success = df['attention_image_generated'].sum()
                print(f"\n高覆盖率(≥95%)样本: {high_coverage}/{len(df)} ({high_coverage / len(df) * 100:.1f}%)")
                print(f"有对应DSA: {has_dsa}/{len(df)} ({has_dsa / len(df) * 100:.1f}%)")
                print(f"生成注意力图像: {attention_success}/{len(df)} ({attention_success / len(df) * 100:.1f}%)")

                # 显示低覆盖率样本
                low_coverage = df[df['coverage'] < 0.85]
                if len(low_coverage) > 0:
                    print(f"\n低覆盖率(<85%)样本 ({len(low_coverage)}个):")
                    for idx, row in low_coverage.head(5).iterrows():
                        print(
                            f"  {row['filename']}: 覆盖率={row['coverage']:.1%}, DSA={'✓' if row['has_dsa'] else '✗'}, 注意力={'✓' if row['attention_image_generated'] else '✗'}")
                    if len(low_coverage) > 5:
                        print(f"  ... 还有{len(low_coverage) - 5}个")

        else:
            print("处理失败，请检查输入文件")

    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()