import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List, Union, Any
import warnings
import json
from datetime import datetime
import re
import random
import pickle
import shutil
from multi.locate_net_2channle import CoarseInfoExtractor_Simplified  # 导入新模型

warnings.filterwarnings('ignore')


# ==============================
# 1. 通用工具函数（保持不变）
# ==============================

def setup_output_directory(base_dir: str, prefix: str = "") -> Path:
    """创建带日期和随机数的输出目录"""
    date_str = datetime.now().strftime('%Y%m%d')
    random_num = random.randint(100, 999)
    dir_name = f"{prefix}_{date_str}_{random_num}" if prefix else f"{date_str}_{random_num}"
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_image(file_path: Path, target_size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
    """通用图像加载函数"""
    try:
        if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom']:
            dicom_data = pydicom.dcmread(str(file_path), force=True)
            image = dicom_data.pixel_array.astype(np.float32)
        else:
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            image = image.astype(np.float32)

        # 归一化
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            image = np.zeros_like(image)

        # 调整大小
        if image.shape != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        return image

    except Exception as e:
        print(f"加载图像 {file_path.name} 失败: {e}")
        return None


def save_image(image: np.ndarray, path: Path, normalize: bool = True):
    """通用图像保存函数"""
    if normalize and image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    cv2.imwrite(str(path), image)


def extract_case_id(filename: str) -> Optional[str]:
    """从文件名提取病历号"""
    basename = os.path.splitext(filename)[0]

    # 模式: 字母_数字 或 数字
    match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
    if match:
        return match.group(0)

    # 只匹配数字部分
    match = re.search(r'(\d+)', basename)
    if match:
        return match.group(1)

    return None


def compute_overlap_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """计算重叠指标"""
    intersection = np.logical_and(pred_mask > threshold, gt_mask > threshold).sum()
    union = np.logical_or(pred_mask > threshold, gt_mask > threshold).sum()
    iou = intersection / (union + 1e-8)

    coverage = (gt_mask * pred_mask).sum() / (gt_mask.sum() + 1e-8)

    return {
        'iou': iou,
        'coverage': coverage
    }


def create_overlay_image(dsa_image: np.ndarray,
                         attention_mask: np.ndarray,
                         aneurysm_mask: np.ndarray,
                         attention_color: str = 'red') -> np.ndarray:
    """创建叠加图像"""
    if len(dsa_image.shape) == 2:
        dsa_rgb = np.stack([dsa_image] * 3, axis=-1)
    else:
        dsa_rgb = dsa_image.copy()

    dsa_rgb = np.clip(dsa_rgb, 0, 1)

    attention_overlay = np.zeros_like(dsa_rgb)
    if attention_color == 'red':
        attention_overlay[:, :, 0] = attention_mask * 0.5
    elif attention_color == 'blue':
        attention_overlay[:, :, 2] = attention_mask * 0.5
    elif attention_color == 'green':
        attention_overlay[:, :, 1] = attention_mask * 0.5

    aneurysm_overlay = np.zeros_like(dsa_rgb)
    aneurysm_overlay[:, :, 1] = aneurysm_mask

    overlay_image = dsa_rgb.copy()
    overlay_image = overlay_image * (1 - 0.3 * attention_mask[:, :, np.newaxis]) + attention_overlay * 0.3
    overlay_image = overlay_image + aneurysm_overlay * 0.7
    overlay_image = np.clip(overlay_image, 0, 1)

    return overlay_image


def plot_comparison_grid(images: List[np.ndarray],
                         titles: List[str],
                         suptitle: str,
                         save_path: Path,
                         figsize: Tuple[int, int] = (25, 5)):
    """绘制对比网格图"""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_history(history: Dict[str, List[float]], output_dir: Path, config: Dict[str, Any]):
    """保存训练历史"""
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_height_loss': history['train_height_loss'],
        'train_width_reg_loss': history.get('train_width_reg_loss', []),
        'train_width_cls_loss': history.get('train_width_cls_loss', []),
        'val_height_loss': history['val_height_loss'],
        'val_width_reg_loss': history.get('val_width_reg_loss', []),
        'val_width_cls_loss': history.get('val_width_cls_loss', []),
        'train_accuracy': history.get('train_accuracy', []),
        'val_accuracy': history.get('val_accuracy', []),
        'learning_rate': history['learning_rate']
    })

    history_df.to_csv(output_dir / 'training_history.csv', index=False)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4, default=str)


def plot_training_curves(history: Dict[str, List[float]], save_dir: Path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 总损失
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('总损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 高度损失
    axes[0, 1].plot(epochs, history['train_height_loss'], 'b-', label='训练高度损失')
    axes[0, 1].plot(epochs, history['val_height_loss'], 'r-', label='验证高度损失')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].set_title('高度比例损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 宽度回归损失
    axes[0, 2].plot(epochs, history.get('train_width_reg_loss', []), 'b-', label='训练宽度回归损失')
    axes[0, 2].plot(epochs, history.get('val_width_reg_loss', []), 'r-', label='验证宽度回归损失')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('损失')
    axes[0, 2].set_title('宽度回归损失')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 宽度分类损失
    axes[1, 0].plot(epochs, history.get('train_width_cls_loss', []), 'b-', label='训练宽度分类损失')
    axes[1, 0].plot(epochs, history.get('val_width_cls_loss', []), 'r-', label='验证宽度分类损失')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].set_title('宽度分类损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 分类准确率
    axes[1, 1].plot(epochs, history.get('train_accuracy', []), 'b-', label='训练准确率')
    axes[1, 1].plot(epochs, history.get('val_accuracy', []), 'r-', label='验证准确率')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_title('宽度分类准确率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 学习率
    axes[1, 2].plot(epochs, history['learning_rate'], 'g-', marker='o')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('学习率')
    axes[1, 2].set_title('学习率调度')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


# ==============================
# 2. Early Stopping
# ==============================
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


# ==============================
# 3. AttentionMaskGenerator
# ==============================
class AttentionMaskGenerator(nn.Module):
    def __init__(self, image_size: Tuple[int, int] = (512, 512), min_radius_ratio: float = 0.06):
        super().__init__()
        self.H, self.W = image_size
        self.min_radius_ratio = min_radius_ratio
        self.device = torch.device('cpu')

        print(f"AttentionMaskGenerator初始化: 图像尺寸={image_size}, 最小半径比例={min_radius_ratio}")

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, height_ratio: torch.Tensor,
                width_ratio: torch.Tensor) -> torch.Tensor:

        B = height_ratio.shape[0]
        H, W = self.H, self.W

        if height_ratio.device != self.device:
            self.device = height_ratio.device

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

            # 创建矩形掩膜
            attention_mask = torch.zeros(H, W, device=self.device)
            attention_mask[y_min:y_max + 1, :] = 1.0

            batch_masks.append(attention_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(batch_masks, dim=0)


# ==============================
# 4. PositionInfoLoader
# ==============================
class PositionInfoLoader:
    """加载动脉瘤类别信息（6类）"""

    def __init__(self, excel_path: str = None):
        self.position_dict = {}  # 存储文件名到类别索引的映射
        self.case_to_position = {}  # 存储病历号到类别索引的映射
        self.class_names = {
            0: "Segment 1 (原1类)",
            1: "Segment 2 (原2类)",
            2: "Segment 4 (原4类)",
            3: "Segment 5 (原5类)",
            4: "Segment 6 (原6类)",
            5: "Segment 7 (原7类)"
        }
        self.valid_classes = [1, 2, 4, 5, 6, 7]  # 有效原始类别

        if excel_path:
            self._load_position_info(excel_path)

    def _load_position_info(self, excel_path: str):
        """加载类别信息"""
        try:
            df = pd.read_excel(excel_path)
            print(f"类别信息Excel列名: {df.columns.tolist()}")
            print(f"前5行数据:\n{df.head()}")

            # 假设第一列是文件名，第二列是类别
            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()

                # 提取基础文件名
                base_name = os.path.splitext(filename)[0]

                # 获取类别编号
                try:
                    original_class = int(row.iloc[1]) if not pd.isna(row.iloc[1]) else None

                    if original_class is None:
                        continue

                    # 只保留有效类别 (1,2,4,5,6,7)
                    if original_class in self.valid_classes:
                        # 映射到新索引 (0-5)
                        new_index = self.valid_classes.index(original_class)

                        # 保存到字典
                        self.position_dict[base_name] = {
                            'original_class': original_class,
                            'new_index': new_index
                        }

                        # 提取病历号
                        case_id = extract_case_id(filename)
                        if case_id:
                            if case_id not in self.case_to_position:
                                self.case_to_position[case_id] = {
                                    'original_class': original_class,
                                    'new_index': new_index
                                }
                    else:
                        print(f"跳过类别 {original_class} (不在有效类别[1,2,4,5,6,7]中)")

                except Exception as e:
                    print(f"处理行数据时出错 {filename}: {e}")
                    continue

            print(f"成功加载 {len(self.position_dict)} 个文件的类别信息")
            print(f"成功加载 {len(self.case_to_position)} 个病历号的类别信息")

            # 统计类别分布
            new_indices = [v['new_index'] for v in self.position_dict.values()]
            dist = {}
            for idx in new_indices:
                dist[idx] = dist.get(idx, 0) + 1

            print("类别分布(新索引):")
            for idx, count in sorted(dist.items()):
                print(f"  新类别{idx} (原{self.valid_classes[idx]}): {count}个样本")

        except Exception as e:
            print(f"加载类别信息失败: {e}")
            import traceback
            traceback.print_exc()

    def get_position_for_image(self, filename: str):
        """获取图像对应的类别信息"""
        basename = os.path.splitext(filename)[0]

        # 方法1: 直接匹配
        if basename in self.position_dict:
            pos_info = self.position_dict[basename]
            position_tensor = torch.zeros(6)
            position_tensor[pos_info['new_index']] = 1.0
            return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 方法2: 提取病历号进行匹配
        case_id = extract_case_id(filename)
        if case_id and case_id in self.case_to_position:
            pos_info = self.case_to_position[case_id]
            position_tensor = torch.zeros(6)
            position_tensor[pos_info['new_index']] = 1.0
            return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 方法3: 只匹配数字部分
        match = re.search(r'(\d+)', basename)
        if match:
            num_part = match.group(1)
            for stored_case_id in self.case_to_position.keys():
                if num_part in stored_case_id:
                    pos_info = self.case_to_position[stored_case_id]
                    position_tensor = torch.zeros(6)
                    position_tensor[pos_info['new_index']] = 1.0
                    return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 返回默认值
        print(f"警告: 未找到图像 {filename} 的类别信息，使用默认类别0")
        position_tensor = torch.zeros(6)
        position_tensor[0] = 1.0  # 默认使用类别0
        return position_tensor, 0, "Segment 1 (default)"


# ==============================
# 5. SizeRangeInfoLoader
# ==============================
class SizeRangeInfoLoader:
    """加载尺寸分类信息"""

    # 尺寸分类边界（真实值范围）
    SIZE_CATEGORIES = {
        'tiny': (0, 0.1),  # < 0.1
        'small': (0.1, 0.2),  # 0.1-0.2
        'medium': (0.2, 0.475),  # 0.2-0.475
        'large': (0.475, 1.0)  # > 0.475
    }

    # 分类标签映射
    CATEGORY_TO_CLASS = {
        'tiny': 0,
        'small': 1,
        'medium': 2,
        'large': 3
    }

    CLASS_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_CLASS.items()}

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}  # 存储文件名到位置信息的映射
        self.case_to_location = {}  # 存储病历号到位置信息的映射
        self.size_distribution = {category: 0 for category in self.CATEGORY_TO_CLASS.keys()}
        self._load_location_info()

    def _get_size_category(self, width_ratio: float) -> str:
        """根据宽度比例获取尺寸分类"""
        if width_ratio < 0.1:
            return 'tiny'
        elif width_ratio < 0.2:
            return 'small'
        elif width_ratio < 0.475:
            return 'medium'
        else:
            return 'large'

    def _get_size_class(self, width_ratio: float) -> int:
        """根据宽度比例获取尺寸类别索引"""
        if width_ratio < 0.1:
            return 0  # tiny
        elif width_ratio < 0.2:
            return 1  # small
        elif width_ratio < 0.475:
            return 2  # medium
        else:
            return 3  # large

    def _parse_size_value(self, size_value: Any) -> float:
        """
        解析尺寸值
        返回实际的宽度比例值（真实值，范围0-0.4）
        """
        try:
            if isinstance(size_value, (int, float)):
                return float(size_value)
            elif isinstance(size_value, str):
                # 尝试直接转换
                try:
                    return float(size_value)
                except:
                    # 如果是分类标签，返回该类别区间的中值
                    if size_value in ['0.04', '0.08', '0.19', '0.4']:
                        mapping = {'0.04': 0.02, '0.08': 0.06, '0.19': 0.135, '0.4': 0.295}
                        return mapping[size_value]
                    else:
                        print(f"警告: 无法解析尺寸值 '{size_value}'，使用默认值0.1")
                        return 0.1
            else:
                return 0.1
        except:
            print(f"警告: 无法解析尺寸值 '{size_value}'，使用默认值0.1")
            return 0.1

    def _load_location_info(self):
        """从Excel加载位置信息"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"位置信息Excel列名: {df.columns.tolist()}")
            print(f"前5行数据:\n{df.head()}")

            # 假设前三列：Filename, Height Ratio, [忽略]，第五列：Size Range
            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()

                # 获取高度比例（第二列）
                try:
                    height_ratio = float(row.iloc[1])
                    height_ratio = max(0.0, min(1.0, height_ratio))
                except:
                    print(f"警告: 无法解析高度比例 '{row.iloc[1]}'，使用默认值0.5")
                    height_ratio = 0.5

                # 获取尺寸值（第五列） - 真实值，范围0-0.4
                size_value = row.iloc[4] if len(row) > 4 else 0.19
                width_ratio_true = self._parse_size_value(size_value)  # 真实值 0-0.4

                # 确保宽度比例在0-0.4范围内
                width_ratio_true = max(0.0, min(0.4, width_ratio_true))

                # 获取尺寸分类名称
                size_category = self._get_size_category(width_ratio_true)
                size_class = self._get_size_class(width_ratio_true)

                # 更新分布统计
                self.size_distribution[size_category] += 1

                # 去除可能的扩展名
                basename = os.path.splitext(filename)[0]

                location_info = {
                    'height_ratio': height_ratio,
                    'width_ratio_true': width_ratio_true,  # 真实值 (0-0.4)
                    'width_ratio_scaled': width_ratio_true * 2.5,  # 缩放后的值 (0-1) 用于回归
                    'size_category': size_category,
                    'size_class': size_class,  # 类别索引 (0-3)
                    'original_size_value': size_value
                }

                self.location_dict[basename] = location_info

                # 提取病历号
                case_id = extract_case_id(filename)
                if case_id:
                    if case_id not in self.case_to_location:
                        self.case_to_location[case_id] = location_info

            print(f"成功加载 {len(self.location_dict)} 个文件的位置记录")
            print(f"成功加载 {len(self.case_to_location)} 个病历号的位置记录")
            print("\n尺寸分类分布:")
            for category, count in self.size_distribution.items():
                if count > 0:
                    range_min, range_max = self.SIZE_CATEGORIES[category]
                    class_idx = self.CATEGORY_TO_CLASS[category]
                    print(f"  {category}({class_idx}): {count}个样本 (范围: {range_min}-{range_max})")

        except Exception as e:
            print(f"加载位置信息失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_location_for_image(self, filename: str) -> Dict:
        """获取图像对应的位置信息"""
        basename = os.path.splitext(filename)[0]

        # 方法1: 直接匹配
        if basename in self.location_dict:
            return self.location_dict[basename]

        # 方法2: 提取病历号匹配
        case_id = extract_case_id(filename)
        if case_id and case_id in self.case_to_location:
            return self.case_to_location[case_id]

        # 方法3: 只匹配数字部分
        match = re.search(r'(\d+)', basename)
        if match:
            num_part = match.group(1)
            for stored_case_id in self.case_to_location.keys():
                if num_part in stored_case_id:
                    return self.case_to_location[stored_case_id]

        # 返回默认值
        print(f"警告: 未找到图像 {filename} 的位置信息，使用默认值")
        return {
            'height_ratio': 0.5,
            'width_ratio_true': 0.1,
            'width_ratio_scaled': 0.25,
            'size_category': 'medium',
            'size_class': 2,
            'original_size_value': '0.19'
        }


# ==============================
# 6. StreamDataManager
# ==============================
class StreamDataManager:
    """流式数据管理器 - 负责将数据保存为.npz格式并管理缓存"""

    def __init__(self, cache_root: str = "D:/med_data/ai/stream_cache"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def create_cache_session(self, prefix: str = "dataset") -> str:
        """创建新的缓存会话文件夹"""
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        session_name = f"{date_str}_{random_num}"
        session_path = self.cache_root / session_name
        session_path.mkdir(parents=True, exist_ok=True)

        # 创建训练和测试子文件夹
        (session_path / "train").mkdir(exist_ok=True)
        (session_path / "test").mkdir(exist_ok=True)

        print(f"创建缓存会话: {session_path}")
        return str(session_path)

    def get_existing_sessions(self) -> List[str]:
        """获取所有现有的缓存会话"""
        sessions = []
        for item in self.cache_root.iterdir():
            if item.is_dir() and re.match(r'\d{8}_\d{3}', item.name):
                sessions.append(str(item))
        return sorted(sessions)

    def save_sample(self, data: Dict, save_path: str, filename: str):
        """保存单个样本为.npz文件"""
        # 准备要保存的数据
        save_dict = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                save_dict[key] = value.numpy()
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
            elif isinstance(value, (int, float)):
                save_dict[key] = np.array([value])
            elif isinstance(value, str):
                save_dict[key] = np.array([value])
            else:
                save_dict[key] = np.array([str(value)])

        # 保存
        np.savez(save_path, **save_dict)

    def load_sample(self, npz_path: str) -> Dict:
        """加载.npz文件中的样本"""
        data = np.load(npz_path, allow_pickle=True)

        result = {}
        for key in data.files:
            arr = data[key]
            if arr.dtype == np.object_ and len(arr) == 1:
                # 尝试转换字符串
                try:
                    result[key] = str(arr[0])
                except:
                    result[key] = arr[0]
            elif arr.ndim == 0:
                result[key] = arr.item()
            else:
                # 转换为tensor
                result[key] = torch.from_numpy(arr).float()

        return result


# ==============================
# 7. StreamCoarseExtractorDataset
# ==============================
class StreamCoarseExtractorDataset(Dataset):
    """使用流式数据的CoarseExtractor数据集"""

    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform

        # 获取所有.npz文件
        self.sample_files = list(self.data_dir.glob("*.npz"))
        self.sample_files.sort()

        print(f"加载{split}数据集: {len(self.sample_files)}个样本")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # 从磁盘加载样本
        sample_path = self.sample_files[idx]
        data = np.load(sample_path, allow_pickle=True)

        # 转换为tensor
        image = torch.from_numpy(data['image']).float()
        position = torch.from_numpy(data['position']).float()
        height_ratio = torch.tensor(data['height_ratio'].item(), dtype=torch.float32)
        width_ratio_true = torch.tensor(data['width_ratio_true'].item(), dtype=torch.float32)
        width_ratio_scaled = torch.tensor(data['width_ratio_scaled'].item(), dtype=torch.float32)
        size_class = torch.tensor(data['size_class'].item(), dtype=torch.long)

        # 获取元数据
        filename = str(data['filename'].item()) if data['filename'].ndim == 0 else str(data['filename'][0])
        position_name = str(data['position_name'].item()) if data['position_name'].ndim == 0 else str(
            data['position_name'][0])
        position_num = int(data['position_num'].item()) if data['position_num'].ndim == 0 else int(
            data['position_num'][0])
        size_category = str(data['size_category'].item()) if 'size_category' in data.files else 'medium'

        return {
            'image': image,
            'position': position,
            'height_ratio': height_ratio,
            'width_ratio_true': width_ratio_true,
            'width_ratio_scaled': width_ratio_scaled,
            'size_class': size_class,
            'filename': filename,
            'position_name': position_name,
            'position_num': position_num,
            'size_category': size_category
        }


# ==============================
# 8. DataPreprocessor
# ==============================
class DataPreprocessor:
    """数据预处理器 - 负责将原始数据转换为流式存储格式"""

    def __init__(self,
                 image_dir: str,
                 location_excel_path: str,
                 position_excel_path: str = None,
                 image_size: Tuple[int, int] = (512, 512)):

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # 加载位置信息
        self.location_loader = SizeRangeInfoLoader(location_excel_path)

        # 加载类别信息
        self.position_loader = PositionInfoLoader(position_excel_path)

        # 流式数据管理器
        self.stream_manager = StreamDataManager()

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """加载图像"""
        return load_image(file_path, self.image_size)

    def process_and_save(self,
                         output_session: str = None,
                         train_ratio: float = 0.8,
                         max_samples: int = None,
                         force_new: bool = True) -> str:
        """
        处理数据并保存为流式格式
        """
        # 创建或获取输出会话
        if output_session is None:
            output_session = self.stream_manager.create_cache_session()
        else:
            output_path = Path(output_session)
            if output_path.exists() and not force_new:
                print(f"使用现有会话: {output_session}")
                return output_session
            else:
                # 创建新会话
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "train").mkdir(exist_ok=True)
                (output_path / "test").mkdir(exist_ok=True)

        output_path = Path(output_session)
        train_dir = output_path / "train"
        test_dir = output_path / "test"

        print(f"\n开始处理数据，保存到: {output_session}")
        print("=" * 60)

        # 获取所有图像文件
        image_files = []
        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']:
                    image_files.append(file_path)

        if max_samples:
            image_files = image_files[:max_samples]

        print(f"找到 {len(image_files)} 个图像文件")

        # 统计信息
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'position_dist': {},
            'size_dist': {}
        }

        # 处理每个图像
        processed_samples = []

        for file_path in tqdm(image_files, desc="处理图像"):
            filename = file_path.name
            stats['total'] += 1

            try:
                # 加载图像
                image = self._load_image(file_path)
                if image is None:
                    stats['failed'] += 1
                    continue

                # 添加通道维度
                image = np.expand_dims(image, axis=0)

                # 获取位置信息
                location_info = self.location_loader.get_location_for_image(filename)

                # 获取类别信息
                position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)

                # 更新统计
                stats['success'] += 1
                stats['position_dist'][position_name] = stats['position_dist'].get(position_name, 0) + 1
                stats['size_dist'][location_info['size_category']] = stats['size_dist'].get(
                    location_info['size_category'], 0) + 1

                # 准备保存的数据
                sample_data = {
                    'image': image,
                    'position': position_tensor.numpy(),
                    'height_ratio': np.array([location_info['height_ratio']]),
                    'width_ratio_true': np.array([location_info['width_ratio_true']]),  # 真实值 (0-0.4)
                    'width_ratio_scaled': np.array([location_info['width_ratio_scaled']]),  # 缩放值 (0-1)
                    'size_class': np.array([location_info['size_class']]),  # 类别索引 (0-3)
                    'filename': np.array([filename]),
                    'position_name': np.array([position_name]),
                    'position_num': np.array([position_num]),
                    'size_category': np.array([location_info['size_category']])
                }

                processed_samples.append((file_path.stem, sample_data))

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                stats['failed'] += 1

        # 随机划分训练集和测试集
        random.shuffle(processed_samples)
        split_idx = int(len(processed_samples) * train_ratio)
        train_samples = processed_samples[:split_idx]
        test_samples = processed_samples[split_idx:]

        print(f"\n保存训练集 ({len(train_samples)}个样本) 到 {train_dir}")
        for stem, sample_data in tqdm(train_samples, desc="保存训练集"):
            save_path = train_dir / f"{stem}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path), stem)

        print(f"保存测试集 ({len(test_samples)}个样本) 到 {test_dir}")
        for stem, sample_data in tqdm(test_samples, desc="保存测试集"):
            save_path = test_dir / f"{stem}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path), stem)

        # 保存元数据
        metadata = {
            'total_files': stats['total'],
            'successful': stats['success'],
            'failed': stats['failed'],
            'train_samples': len(train_samples),
            'test_samples': len(test_samples),
            'train_ratio': train_ratio,
            'position_distribution': stats['position_dist'],
            'size_distribution': stats['size_dist'],
            'image_size': self.image_size,
            'created_at': datetime.now().isoformat()
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"总文件数: {stats['total']}")
        print(f"成功处理: {stats['success']}")
        print(f"处理失败: {stats['failed']}")
        print(f"训练集: {len(train_samples)}个样本")
        print(f"测试集: {len(test_samples)}个样本")
        print("\n类别分布:")
        for pos_name, count in stats['position_dist'].items():
            print(f"  {pos_name}: {count}个样本")
        print("\n尺寸分类分布:")
        for size_name, count in stats['size_dist'].items():
            print(f"  {size_name}: {count}个样本")

        return output_session


# ==============================
# 9. 新模型的损失函数
# ==============================
class NewModelLoss(nn.Module):
    """
    新模型损失函数

    新模型可以同时输出：
    - height_ratio: 高度回归值
    - width_value: 宽度连续值 (0-1，对应真实值乘以2.5)
    - width_class: 宽度类别 (0-3)

    损失权重：
    - 高度回归：weight=1.0
    - 宽度分类：weight=1.5 (主要任务)
    - 宽度回归：weight=0.3 (辅助任务)

    宽度分类阈值（真实值）：
        0: < 0.1  (tiny)
        1: 0.1-0.2 (small)
        2: 0.2-0.475 (medium)
        3: 0.475-1.0 (large)
    """

    def __init__(self,
                 height_weight: float = 1.0,  # 高度回归损失权重
                 width_cls_weight: float = 1.5,  # 宽度分类损失权重（主要任务）
                 width_reg_weight: float = 0.3,  # 宽度回归损失权重（辅助任务）
                 thresholds: List[float] = None):  # 分类阈值（真实值）
        super().__init__()

        # 分类阈值（真实值范围）
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.475, 1.0]
        self.thresholds = thresholds
        self.num_classes = len(thresholds)

        # 损失函数组件
        self.mse_loss = nn.MSELoss()  # 回归损失
        self.ce_loss = nn.CrossEntropyLoss()  # 分类损失

        # 权重参数
        self.height_weight = height_weight
        self.width_cls_weight = width_cls_weight
        self.width_reg_weight = width_reg_weight

        print(f"新模型损失函数初始化:")
        print(f"  分类阈值(真实值): {thresholds}")
        print(f"  类别数: {self.num_classes}")
        print(f"  权重 - 高度: {height_weight}, 宽度分类: {width_cls_weight}, 宽度回归: {width_reg_weight}")

    def forward(self,
                outputs: Dict[str, torch.Tensor],  # 模型输出
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播 - 计算组合损失

        Args:
            outputs: 模型输出字典，包含:
                - 'height_ratio': [batch_size] 高度预测
                - 'width_value': [batch_size] 宽度预测值 (0-1，对应真实值*2.5)
                - 'width_class': [batch_size] 宽度预测类别 (可选)
            targets: 目标字典，包含:
                - 'height_ratio': [batch_size] 真实高度
                - 'width_ratio_true': [batch_size] 真实宽度 (0-0.4)
                - 'width_ratio_scaled': [batch_size] 缩放后的宽度 (0-1)
                - 'size_class': [batch_size] 真实尺寸类别 (0-3)

        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        # 1. 高度回归损失
        height_loss = self.mse_loss(outputs['height_ratio'], targets['height_ratio'])

        # 2. 宽度分类损失（主要任务）
        # 使用模型的宽度预测值生成分类logits
        # 方法：根据预测值到各阈值中心的距离计算
        batch_size = outputs['width_value'].shape[0]
        device = outputs['width_value'].device

        # 将预测的宽度值(0-1)转换回真实值范围(0-0.4)
        pred_width_true = outputs['width_value'] / 2.5

        # 构建分类logits（基于距离的软标签）
        centers = torch.tensor([0.05, 0.15, 0.3375, 0.7375], device=device)  # 各区间中点（真实值）

        # 计算每个样本到每个类别中心的距离
        # [batch_size, num_classes]
        distances = torch.abs(pred_width_true.unsqueeze(1) - centers.unsqueeze(0))

        # 将距离转换为logits（距离越小，logits越大）
        # 使用负距离作为logits，并添加温度参数
        temperature = 0.1
        logits = -distances / temperature

        # 使用交叉熵损失
        width_cls_loss = self.ce_loss(logits, targets['size_class'])

        # 3. 宽度回归损失（辅助任务）
        # 使用缩放后的目标值进行回归
        width_reg_loss = self.mse_loss(outputs['width_value'], targets['width_ratio_scaled'])

        # 总损失
        total_loss = (self.height_weight * height_loss +
                      self.width_cls_weight * width_cls_loss +
                      self.width_reg_weight * width_reg_loss)

        # 计算分类准确率
        pred_classes = torch.argmax(logits, dim=1)
        accuracy = (pred_classes == targets['size_class']).float().mean().item()

        loss_dict = {
            'height_loss': height_loss.item(),
            'width_cls_loss': width_cls_loss.item(),
            'width_reg_loss': width_reg_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy
        }

        return total_loss, loss_dict


# ==============================
# 10. 训练器
# ==============================
class CoarseExtractorTrainer:
    """CoarseInfoExtractor_Improved训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录
        self.output_dir = setup_output_directory(
            config.get('model_save_root', "D:/med_data/ai/pre_loc"),
            prefix="train"
        )

        # 子目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"

        for dir_path in [self.checkpoint_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"模型保存目录: {self.output_dir}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_height_loss': [],
            'train_width_reg_loss': [],
            'train_width_cls_loss': [],
            'val_height_loss': [],
            'val_width_reg_loss': [],
            'val_width_cls_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # 创建损失函数
        self.criterion = NewModelLoss(
            height_weight=config.get('height_weight', 1.0),
            width_cls_weight=config.get('width_cls_weight', 1.5),
            width_reg_weight=config.get('width_reg_weight', 0.3)
        )

    def _create_model(self):
        """创建新模型"""
        model = CoarseInfoExtractor_Simplified(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=6,
            num_width_classes=4,  # 4个宽度类别
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        return model

    def _create_dataloaders(self, cache_session: str):
        """从缓存创建数据加载器"""
        train_dataset = StreamCoarseExtractorDataset(
            data_dir=cache_session,
            split="train"
        )

        val_dataset = StreamCoarseExtractorDataset(
            data_dir=cache_session,
            split="test"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        print(f"训练集: {len(train_dataset)}个样本")
        print(f"验证集: {len(val_dataset)}个样本")

        return train_loader, val_loader

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        train_loss_total = 0
        train_height_loss = 0
        train_width_reg_loss = 0
        train_width_cls_loss = 0
        train_accuracy = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [训练]')
        for batch in train_bar:
            # 将数据移动到设备
            images = batch['image'].to(self.device)
            positions = batch['position'].to(self.device)

            targets = {
                'height_ratio': batch['height_ratio'].to(self.device),
                'width_ratio_true': batch['width_ratio_true'].to(self.device),
                'width_ratio_scaled': batch['width_ratio_scaled'].to(self.device),
                'size_class': batch['size_class'].to(self.device)
            }

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, positions)

            # 计算损失
            loss, loss_dict = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录损失
            train_loss_total += loss.item()
            train_height_loss += loss_dict['height_loss']
            train_width_reg_loss += loss_dict['width_reg_loss']
            train_width_cls_loss += loss_dict['width_cls_loss']
            train_accuracy += loss_dict['accuracy']

            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{loss_dict["accuracy"]:.2%}'
            })

        # 计算平均损失和准确率
        num_batches = len(train_loader)
        avg_metrics = {
            'total': train_loss_total / num_batches,
            'height': train_height_loss / num_batches,
            'width_reg': train_width_reg_loss / num_batches,
            'width_cls': train_width_cls_loss / num_batches,
            'accuracy': train_accuracy / num_batches
        }

        return avg_metrics

    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        val_loss_total = 0
        val_height_loss = 0
        val_width_reg_loss = 0
        val_width_cls_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='[验证]')
            for batch in val_bar:
                # 将数据移动到设备
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)

                targets = {
                    'height_ratio': batch['height_ratio'].to(self.device),
                    'width_ratio_true': batch['width_ratio_true'].to(self.device),
                    'width_ratio_scaled': batch['width_ratio_scaled'].to(self.device),
                    'size_class': batch['size_class'].to(self.device)
                }

                # 前向传播
                outputs = model(images, positions)

                # 计算损失
                loss, loss_dict = criterion(outputs, targets)

                # 记录损失
                val_loss_total += loss.item()
                val_height_loss += loss_dict['height_loss']
                val_width_reg_loss += loss_dict['width_reg_loss']
                val_width_cls_loss += loss_dict['width_cls_loss']
                val_accuracy += loss_dict['accuracy']

                # 更新进度条
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{loss_dict["accuracy"]:.2%}'
                })

        # 计算平均损失和准确率
        num_batches = len(val_loader)
        avg_metrics = {
            'total': val_loss_total / num_batches,
            'height': val_height_loss / num_batches,
            'width_reg': val_width_reg_loss / num_batches,
            'width_cls': val_width_cls_loss / num_batches,
            'accuracy': val_accuracy / num_batches
        }

        return avg_metrics

    def save_checkpoint(self, model, optimizer, epoch, train_metrics, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'history': self.history
        }

        if is_best:
            path = self.checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'

        torch.save(checkpoint, path)
        return path

    def train(self, cache_session: str):
        """训练模型"""
        print("\n" + "=" * 60)
        print(f"开始 CoarseInfoExtractor_Improved 训练")
        print("=" * 60)

        # 创建模型
        model = self._create_model()
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 创建数据加载器
        train_loader, val_loader = self._create_dataloaders(cache_session)

        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 早停机制
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=self.config.get('early_stopping_min_delta', 0.001)
        )

        # 训练循环
        best_val_loss = float('inf')
        best_model_path = None

        for epoch in range(self.config['num_epochs']):
            # 训练
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, self.criterion, epoch
            )

            # 验证
            val_metrics = self.validate_epoch(
                model, val_loader, self.criterion
            )

            # 更新学习率
            scheduler.step(val_metrics['total'])

            # 记录历史
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['total'])
            self.history['train_height_loss'].append(train_metrics['height'])
            self.history['train_width_reg_loss'].append(train_metrics['width_reg'])
            self.history['train_width_cls_loss'].append(train_metrics['width_cls'])
            self.history['val_height_loss'].append(val_metrics['height'])
            self.history['val_width_reg_loss'].append(val_metrics['width_reg'])
            self.history['val_width_cls_loss'].append(val_metrics['width_cls'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 打印统计信息
            print(f"\nEpoch {epoch + 1} 统计:")
            print(f"  训练总损失: {train_metrics['total']:.4f}")
            print(f"    高度损失: {train_metrics['height']:.4f}")
            print(f"    宽度分类损失: {train_metrics['width_cls']:.4f}")
            print(f"    宽度回归损失: {train_metrics['width_reg']:.4f}")
            print(f"    训练准确率: {train_metrics['accuracy']:.2%}")
            print(f"  验证总损失: {val_metrics['total']:.4f}")
            print(f"    高度损失: {val_metrics['height']:.4f}")
            print(f"    宽度分类损失: {val_metrics['width_cls']:.4f}")
            print(f"    宽度回归损失: {val_metrics['width_reg']:.4f}")
            print(f"    验证准确率: {val_metrics['accuracy']:.2%}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                best_model_path = self.save_checkpoint(
                    model, optimizer, epoch, train_metrics, val_metrics, is_best=True
                )
                print(f"  ✓ 保存最佳模型到: {best_model_path}")

            # 每5个epoch保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.save_checkpoint(
                    model, optimizer, epoch, train_metrics, val_metrics, is_best=False
                )
                print(f"  ✓ 保存检查点到: {checkpoint_path}")

            # 早停检查
            if early_stopping(val_metrics['total']):
                print(f"\n🚨 触发早停！连续 {early_stopping.patience} 个epoch没有改善")
                break

        # 保存最终模型
        final_model_path = self.checkpoint_dir / 'final_model.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': best_val_loss
        }, final_model_path)

        print(f"\n训练完成！共训练 {epoch + 1} 个epoch")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"最佳验证准确率: {max(self.history['val_accuracy']):.2%}")
        print(f"最终模型保存到: {final_model_path}")

        # 绘制训练曲线
        plot_training_curves(self.history, self.plots_dir)

        # 保存训练历史
        save_training_history(self.history, self.output_dir, self.config)

        return final_model_path


# ==============================
# 11. 测试器
# ==============================
class CoarseExtractorTester:
    """CoarseInfoExtractor_Improved测试器"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 创建注意力掩码生成器
        self.mask_generator = AttentionMaskGenerator(
            image_size=config['image_size']
        ).to(self.device)

        # 加载位置信息
        self.location_loader = SizeRangeInfoLoader(config['location_excel_path'])
        self.position_loader = PositionInfoLoader(config.get('position_excel_path'))

        # 分类边界和类别名称
        self.class_boundaries = [0.1, 0.2, 0.475, 1.0]
        self.class_names = ['tiny', 'small', 'medium', 'large']
        self.num_classes = len(self.class_names)

        # 创建输出目录
        self.output_dir = setup_output_directory(
            "D:/med_data/ai/test_loc",
            prefix="test"
        )

        self.masks_dir = self.output_dir / "masks"
        self.focused_dir = self.output_dir / "focused_images"
        self.comparison_dir = self.output_dir / "comparison"
        self.overlay_dir = self.output_dir / "overlay"
        self.gt_overlay_dir = self.output_dir / "gt_overlay"
        self.standard_masks_dir = self.output_dir / "standard_masks"

        for dir_path in [self.masks_dir, self.focused_dir, self.comparison_dir,
                         self.overlay_dir, self.gt_overlay_dir, self.standard_masks_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"测试结果保存到: {self.output_dir}")
        print(f"分类边界(真实值): {self.class_boundaries}")
        print(f"类别名称: {self.class_names}")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = CoarseInfoExtractor_Simplified(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=6,
            num_width_classes=4,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', None)
        if val_loss is None and 'val_metrics' in checkpoint:
            val_loss = checkpoint['val_metrics'].get('total', 'Unknown')

        print(f"模型加载成功，训练epochs: {epoch}")
        if isinstance(val_loss, (int, float)):
            print(f"验证损失: {val_loss:.4f}")
        else:
            print(f"验证损失: {val_loss}")

        return model

    def _load_image(self, image_path):
        """加载图像"""
        return load_image(image_path, self.config['image_size'])

    def _load_aneurysm_mask(self, filename: str) -> Optional[np.ndarray]:
        """加载动脉瘤掩膜图像"""
        try:
            basename = os.path.splitext(filename)[0]
            mask_dir = Path("D:/med_data/ai/test2")

            possible_names = [
                f"{basename}.tif",
                f"{basename}.tiff",
                f"{basename}.png",
                f"{basename}.jpg",
                f"{basename}_mask.tif",
                f"{basename}_mask.tiff"
            ]

            mask_path = None
            for mask_name in possible_names:
                test_path = mask_dir / mask_name
                if test_path.exists():
                    mask_path = test_path
                    break

            if mask_path is None:
                return None

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                from PIL import Image
                pil_image = Image.open(str(mask_path))
                mask = np.array(pil_image)

            if mask is None:
                return None

            if mask.max() > 1.0:
                mask = mask / 255.0

            mask = (mask > 0.5).astype(np.float32)

            if mask.shape != self.config['image_size']:
                mask = cv2.resize(mask, self.config['image_size'], interpolation=cv2.INTER_NEAREST)

            return mask

        except Exception as e:
            print(f"加载动脉瘤掩膜 {filename} 失败: {e}")
            return None

    def test_single_image(self, image_path):
        """测试单张图像"""
        filename = image_path.name
        basename = os.path.splitext(filename)[0]

        print(f"\n处理图像: {filename}")

        # 加载图像
        image = self._load_image(image_path)
        if image is None:
            print(f"  跳过图像 {filename}")
            return None

        # 获取位置类别信息
        position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)
        print(f"  动脉瘤类别: {position_name}")

        # 模型推理
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        position_tensor = position_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor, position_tensor)
            height_pred = outputs['height_ratio'].item()
            width_pred_scaled = outputs['width_value'].item()  # 缩放值 (0-1)
            width_pred_true = width_pred_scaled / 2.5  # 真实值 (0-0.4)
            width_class = outputs['width_class'].item() if 'width_class' in outputs else None

        print(
            f"  预测 - 高度: {height_pred:.4f}, 宽度(缩放): {width_pred_scaled:.4f}, 宽度(真实): {width_pred_true:.4f}")

        # 获取真实值
        true_info = self.location_loader.get_location_for_image(filename)
        height_true = true_info['height_ratio']
        width_true = true_info['width_ratio_true']  # 真实值 (0-0.4)
        width_scaled_true = true_info['width_ratio_scaled']  # 缩放值 (0-1)
        size_category = true_info['size_category']
        size_class_true = true_info['size_class']

        # 预测的类别
        if width_class is None:
            # 如果没有直接输出类别，根据阈值计算
            if width_pred_true < 0.1:
                pred_class = 0
            elif width_pred_true < 0.2:
                pred_class = 1
            elif width_pred_true < 0.475:
                pred_class = 2
            else:
                pred_class = 3
        else:
            pred_class = width_class

        pred_class_name = self.class_names[pred_class]
        true_class_name = self.class_names[size_class_true]
        size_correct = (pred_class == size_class_true)

        print(f"  真实值 - 高度: {height_true:.4f}, 宽度(真实): {width_true:.4f}, 尺寸分类: {size_category}")
        print(f"  误差 - 高度: {abs(height_pred - height_true):.4f}, 宽度: {abs(width_pred_true - width_true):.4f}")
        print(f"  尺寸分类 - 预测: {pred_class_name} ({pred_class}), 真实: {true_class_name} ({size_class_true})")
        print(f"  分类正确: {'✓' if size_correct else '✗'}")

        # 生成注意力掩膜
        height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
        width_tensor_pred = torch.tensor([width_pred_true], dtype=torch.float32).to(self.device)

        attention_mask_pred = self.mask_generator(height_tensor_pred, width_tensor_pred)
        attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

        focused_image_pred = image * attention_mask_pred_np

        # 生成标准注意力掩膜
        attention_mask_gt_np = None
        focused_image_gt = None
        if height_true is not None and width_true is not None:
            height_tensor_gt = torch.tensor([height_true], dtype=torch.float32).to(self.device)
            width_tensor_gt = torch.tensor([width_true], dtype=torch.float32).to(self.device)

            attention_mask_gt = self.mask_generator(height_tensor_gt, width_tensor_gt)
            attention_mask_gt_np = attention_mask_gt.squeeze().cpu().numpy()
            focused_image_gt = image * attention_mask_gt_np

        # 加载动脉瘤掩膜
        aneurysm_mask = self._load_aneurysm_mask(filename)
        has_aneurysm_mask = aneurysm_mask is not None

        # 计算重叠指标
        pred_overlap_metrics = None
        gt_overlap_metrics = None

        if has_aneurysm_mask:
            pred_overlap_metrics = compute_overlap_metrics(attention_mask_pred_np, aneurysm_mask)

            if attention_mask_gt_np is not None:
                gt_overlap_metrics = compute_overlap_metrics(attention_mask_gt_np, aneurysm_mask)

                print(
                    f"  预测掩膜 - IoU: {pred_overlap_metrics['iou']:.4f}, 覆盖率: {pred_overlap_metrics['coverage']:.2%}")
                print(
                    f"  标准掩膜 - IoU: {gt_overlap_metrics['iou']:.4f}, 覆盖率: {gt_overlap_metrics['coverage']:.2%}")
            else:
                print(
                    f"  预测掩膜 - IoU: {pred_overlap_metrics['iou']:.4f}, 覆盖率: {pred_overlap_metrics['coverage']:.2%}")

        # 保存图像
        self._save_output_images(basename, image, attention_mask_pred_np, attention_mask_gt_np,
                                 focused_image_pred, focused_image_gt, aneurysm_mask,
                                 pred_overlap_metrics, gt_overlap_metrics,
                                 height_pred, width_pred_true, height_true, width_true,
                                 position_name, size_category, size_correct,
                                 pred_class_name, true_class_name)

        # 准备结果字典
        result = {
            'filename': filename,
            'height_pred': height_pred,
            'width_pred_scaled': width_pred_scaled,
            'width_pred_true': width_pred_true,
            'height_true': height_true,
            'width_true': width_true,
            'height_error': abs(height_pred - height_true),
            'width_error': abs(width_pred_true - width_true),
            'position_name': position_name,
            'position_num': position_num,
            'size_category': size_category,
            'pred_class': pred_class,
            'true_class': size_class_true,
            'pred_class_name': pred_class_name,
            'true_class_name': true_class_name,
            'size_correct': size_correct,
            'has_aneurysm_mask': has_aneurysm_mask,
            'pred_iou': pred_overlap_metrics['iou'] if has_aneurysm_mask and pred_overlap_metrics else None,
            'pred_coverage': pred_overlap_metrics['coverage'] if has_aneurysm_mask and pred_overlap_metrics else None,
            'gt_iou': gt_overlap_metrics['iou'] if has_aneurysm_mask and gt_overlap_metrics else None,
            'gt_coverage': gt_overlap_metrics['coverage'] if has_aneurysm_mask and gt_overlap_metrics else None
        }

        return result

    def _save_output_images(self, basename, image, pred_mask, gt_mask,
                            pred_focused, gt_focused, aneurysm_mask,
                            pred_metrics, gt_metrics,
                            height_pred, width_pred, height_true, width_true,
                            position_name, size_category, size_correct,
                            pred_class_name, true_class_name):
        """保存各种输出图像"""
        # 保存原始图像
        save_image(image, self.output_dir / f"{basename}_original.png")

        # 保存预测掩膜
        pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
        save_image(pred_mask_uint8, self.masks_dir / f"{basename}_pred_mask.png", normalize=False)

        # 保存标准掩膜
        if gt_mask is not None:
            gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
            save_image(gt_mask_uint8, self.standard_masks_dir / f"{basename}_standard_mask.png", normalize=False)

        # 保存聚焦图像
        save_image(pred_focused, self.focused_dir / f"{basename}_pred_focused.png")
        if gt_focused is not None:
            save_image(gt_focused, self.focused_dir / f"{basename}_standard_focused.png")

        # 保存动脉瘤掩膜和叠加图像
        if aneurysm_mask is not None:
            aneurysm_uint8 = (aneurysm_mask * 255).astype(np.uint8)
            save_image(aneurysm_uint8, self.output_dir / f"{basename}_aneurysm_mask.png", normalize=False)

            # 预测叠加图像
            overlay_image_pred = create_overlay_image(image, pred_mask, aneurysm_mask, 'red')
            overlay_pred_uint8 = (overlay_image_pred * 255).astype(np.uint8)
            save_image(overlay_pred_uint8, self.overlay_dir / f"{basename}_pred_overlay.png", normalize=False)

            # 标准叠加图像
            if gt_mask is not None:
                overlay_image_gt = create_overlay_image(image, gt_mask, aneurysm_mask, 'blue')
                overlay_gt_uint8 = (overlay_image_gt * 255).astype(np.uint8)
                save_image(overlay_gt_uint8, self.gt_overlay_dir / f"{basename}_standard_overlay.png", normalize=False)

        # 创建对比图
        self._create_comparison_plot(basename, image, pred_mask, gt_mask, aneurysm_mask,
                                     pred_metrics, gt_metrics,
                                     height_pred, width_pred, height_true, width_true,
                                     position_name, size_category, size_correct,
                                     pred_class_name, true_class_name)

    def _create_comparison_plot(self, basename, image, pred_mask, gt_mask, aneurysm_mask,
                                pred_metrics, gt_metrics,
                                height_pred, width_pred, height_true, width_true,
                                position_name, size_category, size_correct,
                                pred_class_name, true_class_name):
        """创建对比图"""
        images_to_plot = [image, pred_mask]

        # 预测掩膜标题
        pred_title = f'预测掩膜\n高度: {height_pred:.3f}\n宽度: {width_pred:.3f}'
        titles_to_plot = ['原始图像', pred_title]

        # 标准掩膜
        if gt_mask is not None:
            gt_title = f'标准掩膜\n高度: {height_true:.3f}\n宽度: {width_true:.3f}'
            images_to_plot.append(gt_mask)
            titles_to_plot.append(gt_title)
        else:
            images_to_plot.append(np.zeros_like(image))
            titles_to_plot.append('无标准掩膜')

        # 预测叠加
        if aneurysm_mask is not None:
            overlay_display = create_overlay_image(image, pred_mask, aneurysm_mask, 'red')
            images_to_plot.append(overlay_display)
            title = '预测叠加\n'
            if pred_metrics:
                title += f'IoU: {pred_metrics["iou"]:.3f}\n'
                title += f'覆盖率: {pred_metrics["coverage"]:.1%}'
            titles_to_plot.append(title)
        else:
            images_to_plot.append(image)
            titles_to_plot.append('无动脉瘤掩膜')

        # 标准叠加
        if aneurysm_mask is not None and gt_mask is not None:
            gt_overlay_display = create_overlay_image(image, gt_mask, aneurysm_mask, 'blue')
            images_to_plot.append(gt_overlay_display)
            title = '标准叠加\n'
            if gt_metrics:
                title += f'IoU: {gt_metrics["iou"]:.3f}\n'
                title += f'覆盖率: {gt_metrics["coverage"]:.1%}'
            titles_to_plot.append(title)
        else:
            if len(images_to_plot) < 5:
                images_to_plot.append(image)
                titles_to_plot.append('无叠加')

        # 总标题
        suptitle = f"{basename}\n动脉瘤类型: {position_name}"
        if height_true is not None:
            suptitle += f"\n预测: 高度={height_pred:.3f}, 宽度={width_pred:.3f} | "
            suptitle += f"标准: 高度={height_true:.3f}, 宽度={width_true:.3f} | "
            suptitle += f"尺寸: {size_category}"
            suptitle += f"\n分类: {pred_class_name} vs {true_class_name} - {'✓ 正确' if size_correct else '✗ 错误'}"
        else:
            suptitle += f"\n预测: 高度={height_pred:.3f}, 宽度={width_pred:.3f}"

        plot_comparison_grid(
            images_to_plot,
            titles_to_plot,
            suptitle,
            self.comparison_dir / f"{basename}_comparison.png",
            figsize=(25, 5)
        )

    def test_all_images(self, test_image_dir):
        """测试所有图像"""
        test_dir = Path(test_image_dir)

        image_files = []
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']:
                    image_files.append(file_path)

        print(f"\n开始测试 {len(image_files)} 张图像...")
        print("=" * 60)

        results = []
        stats = {
            'total': 0,
            'size_correct': 0,
            'height_error_sum': 0,
            'width_error_sum': 0,
            'class_0_correct': 0,
            'class_0_total': 0,
            'class_1_correct': 0,
            'class_1_total': 0,
            'class_2_correct': 0,
            'class_2_total': 0,
            'class_3_correct': 0,
            'class_3_total': 0,
            'iou_above_05': 0,
            'iou_above_07': 0,
            'iou_total': 0,
        }

        for i, image_path in enumerate(tqdm(image_files, desc="测试进度")):
            result = self.test_single_image(image_path)

            if result:
                results.append(result)
                stats['total'] += 1

                # 尺寸分类正确统计
                if result['size_correct']:
                    stats['size_correct'] += 1

                # 误差累加
                stats['height_error_sum'] += result['height_error']
                stats['width_error_sum'] += result['width_error']

                # 按类别统计
                true_class = result['true_class']
                stats[f'class_{true_class}_total'] += 1
                if result['size_correct']:
                    stats[f'class_{true_class}_correct'] += 1

                # IoU统计
                if result.get('pred_iou') is not None:
                    stats['iou_total'] += 1
                    if result['pred_iou'] > 0.5:
                        stats['iou_above_05'] += 1
                    if result['pred_iou'] > 0.7:
                        stats['iou_above_07'] += 1

        if results:
            self._save_test_results(results, stats)

        return results

    def _save_test_results(self, results, stats):
        """保存测试结果"""
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        total = stats['total']
        if total == 0:
            return

        # 计算统计指标
        size_accuracy = stats['size_correct'] / total
        avg_height_error = stats['height_error_sum'] / total
        avg_width_error = stats['width_error_sum'] / total

        # 各类别准确率
        class_accuracies = {}
        for i in range(self.num_classes):
            class_total = stats[f'class_{i}_total']
            class_correct = stats[f'class_{i}_correct']
            class_accuracies[self.class_names[i]] = {
                'accuracy': class_correct / class_total if class_total > 0 else 0,
                'correct': class_correct,
                'total': class_total
            }

        # IoU统计
        iou_above_05_rate = stats['iou_above_05'] / stats['iou_total'] if stats['iou_total'] > 0 else 0
        iou_above_07_rate = stats['iou_above_07'] / stats['iou_total'] if stats['iou_total'] > 0 else 0

        # 保存统计报告
        with open(self.output_dir / 'test_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("测试结果统计报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"测试图像总数: {total}\n\n")

            f.write("1. 尺寸分类准确率\n")
            f.write("-" * 40 + "\n")
            f.write(f"  总体准确率: {size_accuracy:.2%} ({stats['size_correct']}/{total})\n\n")

            f.write("  各类别准确率:\n")
            for class_name in self.class_names:
                acc = class_accuracies[class_name]['accuracy']
                correct = class_accuracies[class_name]['correct']
                total_class = class_accuracies[class_name]['total']
                f.write(f"    {class_name.capitalize():6}: {acc:.2%} ({correct}/{total_class})\n")
            f.write("\n")

            f.write("2. 回归误差\n")
            f.write("-" * 40 + "\n")
            f.write(f"  平均高度误差: {avg_height_error:.4f}\n")
            f.write(f"  平均宽度误差: {avg_width_error:.4f}\n\n")

            if stats['iou_total'] > 0:
                f.write("3. 掩膜重叠指标\n")
                f.write("-" * 40 + "\n")
                f.write(f"  IoU > 0.5: {iou_above_05_rate:.2%} ({stats['iou_above_05']}/{stats['iou_total']})\n")
                f.write(f"  IoU > 0.7: {iou_above_07_rate:.2%} ({stats['iou_above_07']}/{stats['iou_total']})\n")

        # 打印简明统计
        print("\n" + "=" * 60)
        print("测试结果统计")
        print("=" * 60)
        print(f"尺寸分类准确率: {size_accuracy:.2%} ({stats['size_correct']}/{total})")
        print(f"平均高度误差: {avg_height_error:.4f}")
        print(f"平均宽度误差: {avg_width_error:.4f}")
        if stats['iou_total'] > 0:
            print(f"IoU > 0.5: {iou_above_05_rate:.2%}")

        print(f"\n详细报告已保存到: {self.output_dir / 'test_report.txt'}")


# ==============================
# 12. 主函数
# ==============================
def main():
    """主函数"""
    print("CoarseInfoExtractor_Improved 训练与测试程序")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/translate/contrast",
        'val_image_dir': "D:/med_data/ai/translate/test1",
        'test_image_dir': "D:/med_data/ai/translate/test1",
        'location_excel_path': "D:/med_data/ai/translate/location_contrast_size.xlsx",
        'position_excel_path': "D:/med_data/ai/translate/contrast/classify_all_trans_updated.xlsx",

        # 模型参数
        'image_size': (512, 512),
        'base_channels': 32,
        'num_position_classes': 6,
        'num_width_classes': 4,
        'dropout_rate': 0.2,

        # 损失函数权重
        'height_weight': 1.5,  # 高度回归损失权重
        'width_cls_weight': 1.2,  # 宽度分类损失权重（主要任务）
        'width_reg_weight': 0.4,  # 宽度回归损失权重（辅助任务）

        # 训练参数
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,

        # 早停参数
        'early_stopping_patience': 6,
        'early_stopping_min_delta': 0.001,

        # 其他参数
        'num_workers': 2,
        'max_train_samples': None,
        'max_val_samples': None,

        # 保存路径配置
        'cache_root': "D:/med_data/ai/stream_cache",
        'model_save_root': "D:/med_data/ai/pre_loc",
    }

    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # ========== 流式数据预处理 ==========
    print("\n步骤 0: 数据预处理 (流式存储)")

    # 创建数据预处理器
    preprocessor = DataPreprocessor(
        image_dir=config['train_image_dir'],
        location_excel_path=config['location_excel_path'],
        position_excel_path=config['position_excel_path'],
        image_size=config['image_size']
    )

    # 选择是否新建数据或使用已有数据
    use_existing_data = False  # 设置为True使用已有数据，False新建数据

    if use_existing_data:
        # 获取现有缓存会话
        stream_manager = StreamDataManager(cache_root=config['cache_root'])
        existing_sessions = stream_manager.get_existing_sessions()

        if existing_sessions:
            print(f"找到 {len(existing_sessions)} 个现有缓存会话:")
            for i, session in enumerate(existing_sessions):
                print(f"  [{i}] {session}")

            # 选择最新的会话
            cache_session = existing_sessions[-1]
            print(f"使用最新会话: {cache_session}")
        else:
            print("未找到现有缓存会话，将创建新数据")
            cache_session = preprocessor.process_and_save(
                force_new=True,
                train_ratio=0.8,
                max_samples=config['max_train_samples']
            )
    else:
        # 新建数据
        cache_session = preprocessor.process_and_save(
            force_new=True,
            train_ratio=0.8,
            max_samples=config['max_train_samples']
        )

    print(f"缓存会话路径: {cache_session}")

    # ========== 训练模型 ==========
    print("\n步骤 1: 训练 CoarseInfoExtractor_Improved 模型")
    trainer = CoarseExtractorTrainer(config)
    trained_model_path = trainer.train(cache_session)

    # ========== 测试模型 ==========
    print("\n" + "=" * 60)
    print("步骤 2: 测试训练好的模型")

    if isinstance(trained_model_path, Path):
        trained_model_path = str(trained_model_path)

    tester = CoarseExtractorTester(trained_model_path, config)
    test_results = tester.test_all_images(config['test_image_dir'])

    print("\n" + "=" * 60)
    print("程序完成!")
    print(f"训练输出目录: {trainer.output_dir}")
    print(f"测试输出目录: {tester.output_dir}")

    if test_results:
        print(f"\n测试了 {len(test_results)} 张图像")

        print("\n前5个测试结果:")
        for i, result in enumerate(test_results[:5]):
            print(f"  {i + 1}. {result['filename']}:")
            print(f"      动脉瘤类别: {result['position_name']}, 尺寸分类: {result['size_category']}")
            print(f"      预测 - 高度: {result['height_pred']:.4f}, 宽度: {result['width_pred_true']:.4f}")
            if result['height_true']:
                print(f"      标准 - 高度: {result['height_true']:.4f}, 宽度: {result['width_true']:.4f}")
                print(f"      误差 - 高度: {result['height_error']:.4f}, 宽度: {result['width_error']:.4f}")
                print(f"      分类 - 预测: {result['pred_class_name']}, 标准: {result['true_class_name']}")
            if result['has_aneurysm_mask']:
                print(f"      预测掩膜 - IoU: {result['pred_iou']:.4f}, 覆盖率: {result['pred_coverage']:.2%}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()