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

from multi.seperate_height_width import GraphEnhancedCoarseInfoExtractor


warnings.filterwarnings('ignore')


# ==============================
# 1. 通用工具函数（保持不变）
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


def create_overlay_image(dsa_image: np.ndarray, attention_mask: np.ndarray, aneurysm_mask: np.ndarray, attention_color: str = 'red') -> np.ndarray:
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


def plot_comparison_grid(images: List[np.ndarray], titles: List[str], suptitle: str, save_path: Path, figsize: Tuple[int, int] = (25, 5)):
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
        'train_contrast_loss': history.get('train_contrast_loss', []),
        'train_orthogonal_loss': history.get('train_orthogonal_loss', []),
        'train_aneurysm_loss': history.get('train_aneurysm_loss', []),
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
    """Plot training curves including loss, accuracy, and learning rate"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Height loss
    axes[0, 1].plot(epochs, history['train_height_loss'], 'b-', label='Train Height Loss')
    axes[0, 1].plot(epochs, history['val_height_loss'], 'r-', label='Val Height Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Height Ratio Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Width regression loss
    axes[0, 2].plot(epochs, history.get('train_width_reg_loss', []), 'b-', label='Train Width Reg Loss')
    axes[0, 2].plot(epochs, history.get('val_width_reg_loss', []), 'r-', label='Val Width Reg Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Width Regression Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Width classification loss
    axes[0, 3].plot(epochs, history.get('train_width_cls_loss', []), 'b-', label='Train Width Cls Loss')
    axes[0, 3].plot(epochs, history.get('val_width_cls_loss', []), 'r-', label='Val Width Cls Loss')
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('Loss')
    axes[0, 3].set_title('Width Classification Loss')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # Contrastive learning loss
    axes[1, 0].plot(epochs, history.get('train_contrast_loss', []), 'g-', label='Train Contrast Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Contrastive Learning Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Orthogonality loss
    axes[1, 1].plot(epochs, history.get('train_orthogonal_loss', []), 'g-', label='Train Orthogonal Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Orthogonality Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Classification accuracy
    axes[1, 2].plot(epochs, history.get('train_accuracy', []), 'b-', label='Train Accuracy')
    axes[1, 2].plot(epochs, history.get('val_accuracy', []), 'r-', label='Val Accuracy')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Width Classification Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 3].plot(epochs, history['learning_rate'], 'g-', marker='o')
    axes[1, 3].set_xlabel('Epoch')
    axes[1, 3].set_ylabel('Learning Rate')
    axes[1, 3].set_title('Learning Rate Schedule')
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


# 2. Early Stopping
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

class IndependentEarlyStopping:
    """独立早停机制 - 分别监控两个路径"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta

        # 分别记录两个路径的状态
        self.height_counter = 0
        self.width_counter = 0
        self.height_best_loss = None
        self.width_best_loss = None
        self.height_stopped = False
        self.width_stopped = False

    def update_height(self, val_loss):
        """更新高度路径早停状态"""
        if self.height_stopped:
            return True

        if self.height_best_loss is None:
            self.height_best_loss = val_loss
        elif val_loss > self.height_best_loss - self.min_delta:
            self.height_counter += 1
            if self.height_counter >= self.patience:
                self.height_stopped = True
                print(f"\n🛑 Height path early stopping triggered! (patience={self.patience})")
        else:
            self.height_best_loss = val_loss
            self.height_counter = 0
        return self.height_stopped

    def update_width(self, val_loss):
        """更新宽度路径早停状态"""
        if self.width_stopped:
            return True

        if self.width_best_loss is None:
            self.width_best_loss = val_loss
        elif val_loss > self.width_best_loss - self.min_delta:
            self.width_counter += 1
            if self.width_counter >= self.patience:
                self.width_stopped = True
                print(f"\n🛑 Width path early stopping triggered! (patience={self.patience})")
        else:
            self.width_best_loss = val_loss
            self.width_counter = 0
        return self.width_stopped

    @property
    def all_stopped(self):
        """两条路径都停止"""
        return self.height_stopped and self.width_stopped

    @property
    def height_active(self):
        return not self.height_stopped

    @property
    def width_active(self):
        return not self.width_stopped

    def reset(self):
        """重置早停状态"""
        self.height_counter = 0
        self.width_counter = 0
        self.height_best_loss = None
        self.width_best_loss = None
        self.height_stopped = False
        self.width_stopped = False


class TwoStageEarlyStopping:
    """两阶段独立早停 - 位置路径和尺寸路径分别监控"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta

        # 位置路径早停
        self.pos_counter = 0
        self.pos_best_loss = None
        self.pos_stopped = False

        # 尺寸路径早停
        self.size_counter = 0
        self.size_best_loss = None
        self.size_stopped = False

        self._stage = 'position'  # 'position' 或 'size'

    def set_stage(self, stage: str):
        """设置当前训练阶段"""
        self._stage = stage

    def update(self, val_loss: float) -> bool:
        """根据当前阶段更新早停状态"""
        if self._stage == 'position':
            return self._update_position(val_loss)
        else:
            return self._update_size(val_loss)

    def _update_position(self, val_loss: float) -> bool:
        if self.pos_stopped:
            return True
        if self.pos_best_loss is None:
            self.pos_best_loss = val_loss
        elif val_loss > self.pos_best_loss - self.min_delta:
            self.pos_counter += 1
            if self.pos_counter >= self.patience:
                self.pos_stopped = True
                print(f"\n🛑 Position path early stopping triggered! (patience={self.patience})")
        else:
            self.pos_best_loss = val_loss
            self.pos_counter = 0
        return self.pos_stopped

    def _update_size(self, val_loss: float) -> bool:
        if self.size_stopped:
            return True
        if self.size_best_loss is None:
            self.size_best_loss = val_loss
        elif val_loss > self.size_best_loss - self.min_delta:
            self.size_counter += 1
            if self.size_counter >= self.patience:
                self.size_stopped = True
                print(f"\n🛑 Size path early stopping triggered! (patience={self.patience})")
        else:
            self.size_best_loss = val_loss
            self.size_counter = 0
        return self.size_stopped

    @property
    def position_active(self):
        return not self.pos_stopped

    @property
    def size_active(self):
        return not self.size_stopped

    @property
    def all_stopped(self):
        return self.pos_stopped and self.size_stopped

# 3. AttentionMaskGenerator
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

# 4. PositionInfoLoader
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
        print(f"警告: 未找到图像 {filename} 的类别信息，使用默认类别4")
        position_tensor = torch.zeros(6)
        position_tensor[2] = 1.0  # 默认使用类别0
        return position_tensor, 0, "Segment 1 (default)"

# 5. SizeRangeInfoLoader
class SizeRangeInfoLoader:
    """加载尺寸分类信息 - 新增宽度比例（X坐标）"""

    # 尺寸分类边界（真实值范围）
    SIZE_CATEGORIES = {
        'tiny': (0, 0.1),
        'small': (0.1, 0.2),
        'medium': (0.2, 0.475),
        'large': (0.475, 1.0)
    }

    CATEGORY_TO_CLASS = {
        'tiny': 0,
        'small': 1,
        'medium': 2,
        'large': 3
    }

    CLASS_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_CLASS.items()}

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}
        self.case_to_location = {}
        self.size_distribution = {category: 0 for category in self.CATEGORY_TO_CLASS.keys()}
        self._load_location_info()

    def _get_size_category(self, width_ratio: float) -> str:
        if width_ratio < 0.1:
            return 'tiny'
        elif width_ratio < 0.2:
            return 'small'
        elif width_ratio < 0.475:
            return 'medium'
        else:
            return 'large'

    def _get_size_class(self, width_ratio: float) -> int:
        if width_ratio < 0.1:
            return 0
        elif width_ratio < 0.2:
            return 1
        elif width_ratio < 0.475:
            return 2
        else:
            return 3

    def _parse_size_value(self, size_value: Any) -> float:
        try:
            if isinstance(size_value, (int, float)):
                return float(size_value)
            elif isinstance(size_value, str):
                try:
                    return float(size_value)
                except:
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
        """从Excel加载位置信息 - 新格式：第1列文件名，第2列高度，第3列宽度，第4列尺寸"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"位置信息Excel列名: {df.columns.tolist()}")
            print(f"前5行数据:\n{df.head()}")

            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()

                # 获取高度比例（第二列）
                try:
                    height_ratio = float(row.iloc[1])
                    height_ratio = max(0.0, min(1.0, height_ratio))
                except:
                    print(f"警告: 无法解析高度比例 '{row.iloc[1]}'，使用默认值0.5")
                    height_ratio = 0.5

                # 获取宽度比例/X坐标（第三列）
                try:
                    width_ratio_x = float(row.iloc[2])
                    width_ratio_x = max(0.0, min(1.0, width_ratio_x))
                except:
                    print(f"警告: 无法解析宽度比例/X坐标 '{row.iloc[2]}'，使用默认值0.5")
                    width_ratio_x = 0.5

                # 获取尺寸值（第四列）
                size_value = row.iloc[3] if len(row) > 3 else 0.19
                width_ratio_true = self._parse_size_value(size_value)
                width_ratio_true = max(0.0, min(0.4, width_ratio_true))

                size_category = self._get_size_category(width_ratio_true)
                size_class = self._get_size_class(width_ratio_true)

                self.size_distribution[size_category] += 1

                basename = os.path.splitext(filename)[0]

                location_info = {
                    'height_ratio': height_ratio,      # Y坐标
                    'width_ratio_x': width_ratio_x,    # X坐标（新增）
                    'width_ratio_true': width_ratio_true,
                    'width_ratio_scaled': width_ratio_true * 2.5,
                    'size_category': size_category,
                    'size_class': size_class,
                    'original_size_value': size_value
                }

                self.location_dict[basename] = location_info

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

        if basename in self.location_dict:
            return self.location_dict[basename]

        case_id = extract_case_id(filename)
        if case_id and case_id in self.case_to_location:
            return self.case_to_location[case_id]

        match = re.search(r'(\d+)', basename)
        if match:
            num_part = match.group(1)
            for stored_case_id in self.case_to_location.keys():
                if num_part in stored_case_id:
                    return self.case_to_location[stored_case_id]

        print(f"警告: 未找到图像 {filename} 的位置信息，使用默认值")
        return {
            'height_ratio': 0.5,
            'width_ratio_x': 0.5,
            'width_ratio_true': 0.1,
            'width_ratio_scaled': 0.25,
            'size_category': 'medium',
            'size_class': 2,
            'original_size_value': '0.19'
        }

# 6. StreamDataManager
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

# 7. StreamCoarseExtractorDataset
class StreamCoarseExtractorDataset(Dataset):
    """使用流式数据的CoarseExtractor数据集 - 支持X坐标"""

    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 transform=None,
                 augment_for_contrast: bool = False):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.augment_for_contrast = augment_for_contrast

        self.sample_files = list(self.data_dir.glob("*.npz"))
        self.sample_files.sort()

        print(f"加载{split}数据集: {len(self.sample_files)}个样本")
        if augment_for_contrast:
            print(f"  启用对比学习增强")

    def __len__(self):
        return len(self.sample_files)

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        img_np = image.numpy()
        aug_type = random.choice(['contrast', 'brightness', 'noise', 'blur', 'gamma'])

        if aug_type == 'contrast':
            factor = random.uniform(0.8, 1.2)
            mean = img_np.mean()
            img_np = mean + factor * (img_np - mean)
        elif aug_type == 'brightness':
            delta = random.uniform(-0.1, 0.1)
            img_np = img_np + delta
        elif aug_type == 'noise':
            noise_std = random.uniform(0, 0.02)
            noise = np.random.randn(*img_np.shape) * noise_std
            img_np = img_np + noise
        elif aug_type == 'blur':
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(0.3, 0.8)
            img_np[0] = gaussian_filter(img_np[0], sigma=sigma)
        elif aug_type == 'gamma':
            gamma = random.uniform(0.8, 1.2)
            img_np = np.power(np.clip(img_np, 0, 1), gamma)

        img_np = np.clip(img_np, 0, 1)
        return torch.from_numpy(img_np).float()

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        data = np.load(sample_path, allow_pickle=True)

        image = torch.from_numpy(data['image']).float()
        position = torch.from_numpy(data['position']).float()
        height_ratio = torch.tensor(data['height_ratio'].item(), dtype=torch.float32)
        width_ratio_x = torch.tensor(data['width_ratio_x'].item(), dtype=torch.float32)  # 新增X坐标
        width_ratio_true = torch.tensor(data['width_ratio_true'].item(), dtype=torch.float32)
        width_ratio_scaled = torch.tensor(data['width_ratio_scaled'].item(), dtype=torch.float32)
        size_class = torch.tensor(data['size_class'].item(), dtype=torch.long)

        filename = str(data['filename'].item()) if data['filename'].ndim == 0 else str(data['filename'][0])
        position_name = str(data['position_name'].item()) if data['position_name'].ndim == 0 else str(
            data['position_name'][0])
        position_num = int(data['position_num'].item()) if data['position_num'].ndim == 0 else int(
            data['position_num'][0])
        size_category = str(data['size_category'].item()) if 'size_category' in data.files else 'medium'

        result = {
            'image': image,
            'position': position,
            'height_ratio': height_ratio,
            'width_ratio_x': width_ratio_x,  # 新增
            'width_ratio_true': width_ratio_true,
            'width_ratio_scaled': width_ratio_scaled,
            'size_class': size_class,
            'filename': filename,
            'position_name': position_name,
            'position_num': position_num,
            'size_category': size_category
        }

        if self.augment_for_contrast:
            result['image_aug1'] = self._augment_image(image)
            result['image_aug2'] = self._augment_image(image)

        return result

# 8. DataPreprocessor
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
                    'width_ratio_x': np.array([location_info['width_ratio_x']]),  # 新增
                    'width_ratio_true': np.array([location_info['width_ratio_true']]),
                    'width_ratio_scaled': np.array([location_info['width_ratio_scaled']]),
                    'size_class': np.array([location_info['size_class']]),
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

# 9. 新模型的组合损失函数
class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - 重点惩罚大误差
    公式: loss = |pred - target|^gamma * (pred - target)^2
    """

    def __init__(self, gamma=2.0, epsilon=1e-8):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 计算绝对误差
        abs_error = torch.abs(pred - target)

        # MSE基础损失
        mse = (pred - target) ** 2

        # Focal权重：误差越大，权重越大
        focal_weight = (abs_error + self.epsilon) ** self.gamma

        # 组合损失
        focal_loss = focal_weight * mse

        return focal_loss.mean()


class CombinedNewModelLoss(nn.Module):
    """
    改进版组合损失函数 - 兼容原版调用方式
    - 支持单个增强版本（原版兼容）
    - 支持多个增强版本 + 增强方法权重（新功能）
    """

    def __init__(self,
                 # 主任务损失权重
                 height_weight: float = 1.2,
                 width_reg_weight: float = 0.3,
                 width_cls_weight: float = 1.5,
                 # 辅助任务损失权重
                 contrast_weight: float = 0.15,
                 orthogonal_weight: float = 0.08,
                 aneurysm_weight: float = 0.1,
                 spatial_weight: float = 0.05,
                 # Focal Loss参数
                 use_focal: bool = True,
                 focal_gamma: float = 2.0,
                 focal_mix: float = 0.3,
                 # L1损失权重
                 l1_weight: float = 0.5,
                 # 温度参数
                 contrast_temperature: float = 0.1,
                 # 不同增强类型的对比损失权重乘数（仅在多增强版本时生效）
                 aug_type_weights: Dict[str, float] = None,
                 # 训练阶段
                 current_epoch: int = 0,
                 total_epochs: int = 100):
        super().__init__()

        # 损失权重
        self.height_weight = height_weight
        self.width_reg_weight = width_reg_weight
        self.width_cls_weight = width_cls_weight
        self.contrast_weight = contrast_weight
        self.orthogonal_weight = orthogonal_weight
        self.aneurysm_weight = aneurysm_weight
        self.spatial_weight = spatial_weight
        self.l1_weight = l1_weight

        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_mix = focal_mix
        self.contrast_temperature = contrast_temperature
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        # 不同增强类型的对比损失权重乘数（固定，不随epoch变化）
        if aug_type_weights is None:
            self.aug_type_weights = {
                'contrast': 1.0,
                'brightness': 1.0,
                'noise': 1.2,
                'blur': 1.2,
                'gamma': 1.0,
                'none': 0.0
            }
        else:
            self.aug_type_weights = aug_type_weights

        # 基础损失函数
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.debug_count = 0

        print(f"\n{'=' * 60}")
        print("改进版组合损失函数初始化（兼容原版接口）")
        print(f"{'=' * 60}")
        print(f"主任务 - 高度: {height_weight:.2f}, 宽度回归: {width_reg_weight:.2f}, 宽度分类: {width_cls_weight:.2f}")
        print(f"辅助任务 - 对比: {contrast_weight:.2f}, 正交: {orthogonal_weight:.2f}")
        print(f"          动脉瘤: {aneurysm_weight:.2f}, 空间位置: {spatial_weight:.2f}")
        print(f"增强类型权重: {self.aug_type_weights}")
        print(f"{'=' * 60}\n")

    def update_epoch(self, epoch: int):
        """更新当前epoch，用于渐进式训练"""
        self.current_epoch = epoch

        if epoch < 6:
            # 早期：主要关注主任务
            self.contrast_weight = 0.05
            self.orthogonal_weight = 0.03
            self.spatial_weight = 0.05
            self.aneurysm_weight = 0.05
            self.l1_weight = 2
        elif epoch < 12:
            # 中期：逐渐增加辅助任务
            self.contrast_weight = 0.10
            self.orthogonal_weight = 0.08
            self.spatial_weight = 0.10
            self.aneurysm_weight = 0.10
            self.l1_weight = 5
        else:
            # 后期：使用配置值
            self.contrast_weight = 0.20
            self.orthogonal_weight = 0.15
            self.spatial_weight = 0.15
            self.aneurysm_weight = 0.15
            self.l1_weight = 10

    def _compute_height_loss(self, pred, target):
        """计算高度损失"""
        if self.use_focal:
            abs_error = torch.abs(pred - target)
            mse = (pred - target) ** 2
            focal_weight = (abs_error + 1e-8) ** self.focal_gamma
            height_focal_loss = (focal_weight * mse).mean()
            height_l1_loss = self.l1_loss(pred, target)
            L1_w = self.focal_mix * height_l1_loss
            return (1 - self.focal_mix) * height_focal_loss + L1_w
        else:
            height_mse_loss = self.mse_loss(pred, target)
            height_l1_loss = self.l1_loss(pred, target)
            return height_mse_loss + self.l1_weight * height_l1_loss

    def _compute_contrast_loss(self, outputs_orig, outputs_aug):
        """计算单个增强版本的对比损失"""
        if 'disentangled_features' not in outputs_orig or 'disentangled_features' not in outputs_aug:
            return torch.tensor(0.0, device=outputs_orig['height_ratio'].device)

        n_features = len(outputs_orig['disentangled_features'])
        if n_features == 0:
            return torch.tensor(0.0, device=outputs_orig['height_ratio'].device)

        contrast_loss = 0
        for i in range(n_features):
            f1 = F.normalize(outputs_orig['disentangled_features'][i], dim=1)
            f2 = F.normalize(outputs_aug['disentangled_features'][i], dim=1)
            sim_matrix = torch.mm(f1, f2.T) / self.contrast_temperature
            labels = torch.arange(sim_matrix.size(0)).to(f1.device)
            contrast_loss += (F.cross_entropy(sim_matrix, labels) +
                              F.cross_entropy(sim_matrix.T, labels)) / 2
        return contrast_loss / n_features

    def _compute_orthogonal_loss(self, outputs):
        """计算正交损失"""
        if 'disentangled_features' not in outputs:
            return torch.tensor(0.0, device=outputs['height_ratio'].device)

        n = len(outputs['disentangled_features'])
        if n <= 1:
            return torch.tensor(0.0, device=outputs['height_ratio'].device)

        orth_loss = 0
        for i in range(n):
            for j in range(i + 1, n):
                f1 = F.normalize(outputs['disentangled_features'][i], dim=1)
                f2 = F.normalize(outputs['disentangled_features'][j], dim=1)
                corr = torch.mm(f1, f2.T)
                orth_loss += corr.abs().mean()
        return orth_loss / (n * (n - 1) / 2)

    def _compute_aneurysm_loss(self, outputs_orig, outputs_aug):
        """计算动脉瘤特征一致性损失"""
        if ('aneurysm_features' not in outputs_orig or
                'aneurysm_features' not in outputs_aug):
            return torch.tensor(0.0, device=outputs_orig['height_ratio'].device)

        f1 = F.normalize(outputs_orig['aneurysm_features'], dim=1)
        f2 = F.normalize(outputs_aug['aneurysm_features'], dim=1)
        sim = (f1 * f2).sum(dim=1).mean()
        return 1 - sim

    def forward(self, outputs, targets, aug_outputs=None):
        """
        前向传播 - 兼容原版接口

        Args:
            outputs: 模型输出（可以是原始图像或验证时的输出）
            targets: 目标值
            aug_outputs: 可以是以下三种格式：
                - None: 验证模式，不计算对比损失
                - Dict: 单个增强版本的输出（原版兼容）
                - List[Tuple[Dict, List[str]]]: 多个增强版本的输出及方法名列表（新功能）

        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量字典
        """
        loss_dict = {}
        device = outputs['height_ratio'].device

        # ========== 1. 计算主任务损失（基于 outputs）==========
        height_loss = self._compute_height_loss(outputs['height_ratio'], targets['height_ratio'])
        width_reg_loss = self.mse_loss(outputs['width_value'], targets['width_ratio_scaled'])

        if 'width_logits' in outputs:
            width_cls_loss = self.ce_loss(outputs['width_logits'], targets['size_class'])
        else:
            pred_width_true = outputs['width_value'] / 2.5
            centers = torch.tensor([0.05, 0.15, 0.3375, 0.7375], device=device)
            distances = torch.abs(pred_width_true.unsqueeze(1) - centers.unsqueeze(0))
            logits = -distances / 0.1
            width_cls_loss = self.ce_loss(logits, targets['size_class'])

        total_loss = (self.height_weight * height_loss +
                      self.width_reg_weight * width_reg_loss +
                      self.width_cls_weight * width_cls_loss)

        loss_dict['height_raw'] = height_loss.item()
        loss_dict['width_reg_raw'] = width_reg_loss.item()
        loss_dict['width_cls_raw'] = width_cls_loss.item()

        # ========== 2. 辅助任务损失 ==========

        # 2.1 正交损失
        orth_loss = self._compute_orthogonal_loss(outputs)
        total_loss += self.orthogonal_weight * orth_loss
        loss_dict['orthogonal_raw'] = orth_loss.item()

        # 2.2 对比损失和动脉瘤损失
        contrast_loss_total = 0
        aneurysm_loss_total = 0

        if aug_outputs is not None:
            if isinstance(aug_outputs, dict):
                # 原版兼容：单个增强版本
                contrast_loss = self._compute_contrast_loss(outputs, aug_outputs)
                contrast_loss_total += self.contrast_weight * contrast_loss
                loss_dict['contrast_raw'] = contrast_loss.item()

                if self.aneurysm_weight > 0:
                    aneurysm_loss = self._compute_aneurysm_loss(outputs, aug_outputs)
                    aneurysm_loss_total += self.aneurysm_weight * aneurysm_loss
                    loss_dict['aneurysm_raw'] = aneurysm_loss.item()
                num_views = 1

            elif isinstance(aug_outputs, list):
                # 新功能：多个增强版本，每个带方法名列表
                num_views = len(aug_outputs)
                batch_size = outputs['height_ratio'].shape[0]

                for aug_out, aug_methods in aug_outputs:
                    # 对比损失
                    contrast_loss = self._compute_contrast_loss(outputs, aug_out)

                    # 计算该增强版本的平均类型权重
                    type_weight_sum = 0
                    for method in aug_methods:
                        type_weight_sum += self.aug_type_weights.get(method, 1.0)
                    avg_type_weight = type_weight_sum / len(aug_methods) if aug_methods else 1.0

                    contrast_loss_total += self.contrast_weight * avg_type_weight * contrast_loss

                    # 动脉瘤损失
                    if self.aneurysm_weight > 0:
                        aneurysm_loss = self._compute_aneurysm_loss(outputs, aug_out)
                        aneurysm_loss_total += self.aneurysm_weight * aneurysm_loss

                loss_dict['contrast_raw'] = (contrast_loss_total / num_views).item() if num_views > 0 else 0
                loss_dict['aneurysm_raw'] = (aneurysm_loss_total / num_views).item() if num_views > 0 else 0
            else:
                num_views = 0
                loss_dict['contrast_raw'] = 0
                loss_dict['aneurysm_raw'] = 0
        else:
            num_views = 0
            loss_dict['contrast_raw'] = 0
            loss_dict['aneurysm_raw'] = 0

        if num_views > 0:
            total_loss += contrast_loss_total
            total_loss += aneurysm_loss_total

        # 空间损失
        if self.spatial_weight > 0 and 'attention_map' in outputs:
            spatial_loss = torch.tensor(0.0, device=device)
            total_loss += self.spatial_weight * spatial_loss
            loss_dict['spatial_raw'] = 0.0

        loss_dict['total'] = total_loss.item()

        # ========== 3. 计算准确率 ==========
        if 'width_logits' in outputs:
            pred_classes = outputs['width_logits'].argmax(dim=1)
        else:
            pred_width_true = outputs['width_value'] / 2.5
            pred_classes = torch.zeros_like(pred_width_true, dtype=torch.long)
            pred_classes[pred_width_true < 0.1] = 0
            pred_classes[(pred_width_true >= 0.1) & (pred_width_true < 0.2)] = 1
            pred_classes[(pred_width_true >= 0.2) & (pred_width_true < 0.475)] = 2
            pred_classes[pred_width_true >= 0.475] = 3

        accuracy = (pred_classes == targets['size_class']).float().mean().item()
        loss_dict['accuracy'] = accuracy

        self.debug_count += 1

        return total_loss, loss_dict

# 10. 数据增强模块
class ContrastiveAugmentation:
    """对比学习数据增强 - 只改变整体，不改变血管结构，同时记录增强方法"""

    def __init__(self, device='cpu'):
        self.device = device
        # 可用的增强方法列表
        self.aug_methods = ['contrast', 'brightness', 'noise', 'blur', 'gamma', 'none']

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, Union[str, List[str]]]:
        """
        应用随机增强

        Args:
            image: [B, C, H, W] 或 [C, H, W]

        Returns:
            augmented_image: 增强后的图像
            aug_method: 如果输入是单张图像返回字符串，如果是batch返回列表
        """
        # 确保是4D张量
        single_image = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            single_image = True

        batch_size = image.shape[0]
        augmented = []
        methods = []

        for i in range(batch_size):
            img = image[i]  # [C, H, W]

            # 随机选择增强类型
            aug_type = random.choice(self.aug_methods)
            methods.append(aug_type)

            if aug_type == 'contrast':
                factor = random.uniform(0.7, 1.3)
                mean = img.mean()
                img_aug = mean + factor * (img - mean)

            elif aug_type == 'brightness':
                delta = random.uniform(-0.15, 0.15)
                img_aug = img + delta

            elif aug_type == 'noise':
                noise_std = random.uniform(0, 0.03)
                noise = torch.randn_like(img) * noise_std
                img_aug = img + noise

            elif aug_type == 'blur':
                kernel_size = random.choice([3, 5])
                padding = kernel_size // 2
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size * kernel_size)
                img_aug = F.conv2d(img.unsqueeze(0), kernel, padding=padding).squeeze(0)

            elif aug_type == 'gamma':
                gamma = random.uniform(0.7, 1.3)
                img_aug = torch.pow(torch.clamp(img, 0, 1), gamma)

            else:  # 'none'
                img_aug = img.clone()

            img_aug = torch.clamp(img_aug, 0, 1)
            augmented.append(img_aug)

        result = torch.stack(augmented)

        if single_image:
            return result.squeeze(0), methods[0]
        return result, methods

# 11. 训练器
class PositionDecompositionTrainer:
    """位置分解模型训练器 - 两阶段训练：先位置后尺寸"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 导入新模型
        from multi.GCNREG_subimg import PositionDecompositionModel, PositionDecompositionLoss
        self.PositionDecompositionModel = PositionDecompositionModel

        # 创建输出目录
        self.output_dir = setup_output_directory(
            config.get('model_save_root', "D:/med_data/ai/position_models"),
            prefix="pos_size"
        )

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.final_model_dir = self.output_dir / "final_models"

        for dir_path in [self.checkpoint_dir, self.plots_dir, self.final_model_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"Model save directory: {self.output_dir}")

        # 训练历史
        self.history = {
            'train_pos_loss': [],
            'val_pos_loss': [],
            'train_size_loss': [],
            'val_size_loss': [],
            'train_conf_loss': [],
            'val_conf_loss': [],
            'learning_rate_pos': [],
            'learning_rate_size': []
        }

        self.criterion = PositionDecompositionLoss(
            pos_weight=config.get('pos_weight', 1.0),
            size_weight=config.get('size_weight', 1.0),
            conf_weight=config.get('conf_weight', 0.5)
        )

        # 两阶段早停
        self.early_stopping = TwoStageEarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )

    def _create_model(self):
        """创建位置分解模型"""
        from multi.GCNREG_subimg import PositionDecompositionModel

        model = PositionDecompositionModel(
            image_size=self.config.get('image_size', 512),
            patch_size=self.config.get('patch_size', 16),
            encoder_base_channels=self.config.get('encoder_base_channels', 32),
            encoder_blocks=self.config.get('encoder_blocks', 2),
            encoder_output_dim=self.config.get('encoder_output_dim', 256),
            localization_hidden=self.config.get('localization_hidden', 128),
            confidence_hidden=self.config.get('confidence_hidden', 64),
            size_hidden=self.config.get('size_hidden', 128),
            local_window_size=self.config.get('local_window_size', 3)
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数总量: {total_params / 1e6:.2f}M")

        return model

    def _create_dataloaders(self, cache_session: str):
        """从缓存创建数据加载器"""
        train_dataset = StreamCoarseExtractorDataset(
            data_dir=cache_session,
            split="train",
            augment_for_contrast=True
        )

        val_dataset = StreamCoarseExtractorDataset(
            data_dir=cache_session,
            split="test",
            augment_for_contrast=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        print(f"Training set: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"Validation set: {len(val_dataset)} samples ({len(val_loader)} batches)")

        return train_loader, val_loader

    def _build_targets(self, batch):
        """构建训练目标"""
        height_ratio = batch['height_ratio'].to(self.device)  # Y坐标
        width_ratio_x = batch['width_ratio_x'].to(self.device)  # X坐标
        width_scaled = batch['width_ratio_scaled'].to(self.device)  # 尺寸

        return {
            'position': torch.stack([width_ratio_x, height_ratio], dim=1),  # [B, 2]
            'size': width_scaled.unsqueeze(1),  # [B, 1]
            'has_aneurysm': torch.ones_like(height_ratio).unsqueeze(1)  # [B, 1]
        }

    def train_position_epoch(self, model, train_loader, optimizer, epoch):
        """训练位置路径（只更新位置和置信度）"""
        model.train()

        total_pos_loss = 0
        total_conf_loss = 0

        train_bar = tqdm(train_loader, desc=f'[Position] Epoch {epoch + 1}')

        for batch in train_bar:
            images = batch['image'].to(self.device)
            targets = self._build_targets(batch)

            optimizer.zero_grad()
            outputs = model(images)

            # 只计算位置和置信度损失
            pos_loss = F.mse_loss(outputs['position'], targets['position'])
            conf_loss = F.binary_cross_entropy(outputs['confidence'], targets['has_aneurysm'])
            loss = pos_loss + conf_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_pos_loss += pos_loss.item()
            total_conf_loss += conf_loss.item()

            train_bar.set_postfix({'pos_loss': f'{pos_loss.item():.4f}', 'conf_loss': f'{conf_loss.item():.4f}'})

        num_batches = len(train_loader)
        return {
            'pos_loss': total_pos_loss / num_batches,
            'conf_loss': total_conf_loss / num_batches
        }

    def train_size_epoch(self, model, train_loader, optimizer, epoch):
        """训练尺寸路径（冻结位置相关参数）"""
        model.train()

        # 冻结位置头和置信度头
        for param in model.localization_head.parameters():
            param.requires_grad = False
        for param in model.confidence_head.parameters():
            param.requires_grad = False

        total_size_loss = 0

        train_bar = tqdm(train_loader, desc=f'[Size] Epoch {epoch + 1}')

        for batch in train_bar:
            images = batch['image'].to(self.device)
            targets = self._build_targets(batch)

            optimizer.zero_grad()
            outputs = model(images)

            # 只计算尺寸损失
            size_loss = F.mse_loss(outputs['size'], targets['size'])

            size_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_size_loss += size_loss.item()
            train_bar.set_postfix({'size_loss': f'{size_loss.item():.4f}'})

        num_batches = len(train_loader)
        return {'size_loss': total_size_loss / num_batches}

    def validate_epoch(self, model, val_loader):
        """验证"""
        model.eval()

        total_pos_loss = 0
        total_conf_loss = 0
        total_size_loss = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='[Validation]')
            for batch in val_bar:
                images = batch['image'].to(self.device)
                targets = self._build_targets(batch)

                outputs = model(images)

                pos_loss = F.mse_loss(outputs['position'], targets['position'])
                conf_loss = F.binary_cross_entropy(outputs['confidence'], targets['has_aneurysm'])
                size_loss = F.mse_loss(outputs['size'], targets['size'])

                total_pos_loss += pos_loss.item()
                total_conf_loss += conf_loss.item()
                total_size_loss += size_loss.item()

                val_bar.set_postfix({
                    'pos': f'{pos_loss.item():.4f}',
                    'size': f'{size_loss.item():.4f}'
                })

        num_batches = len(val_loader)
        return {
            'pos_loss': total_pos_loss / num_batches,
            'conf_loss': total_conf_loss / num_batches,
            'size_loss': total_size_loss / num_batches
        }

    def train(self, cache_session: str):
        """两阶段训练"""
        print("\n" + "=" * 60)
        print("Starting Position Decomposition Model Training")
        print("=" * 60)

        model = self._create_model()
        train_loader, val_loader = self._create_dataloaders(cache_session)

        # 第一阶段：训练位置路径
        print("\n" + "=" * 40)
        print("STAGE 1: Training Position Path")
        print("=" * 40)

        self.early_stopping.set_stage('position')

        # 位置路径优化器（只更新位置相关参数）
        pos_params = list(model.sub_extractor.parameters()) + \
                     list(model.encoder.parameters()) + \
                     list(model.localization_head.parameters()) + \
                     list(model.confidence_head.parameters())

        optimizer_pos = torch.optim.AdamW(pos_params, lr=self.config.get('pos_lr', 1e-4))
        scheduler_pos = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_pos, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        best_pos_loss = float('inf')

        for epoch in range(self.config.get('pos_epochs', 30)):
            train_metrics = self.train_position_epoch(model, train_loader, optimizer_pos, epoch)
            val_metrics = self.validate_epoch(model, val_loader)

            scheduler_pos.step(val_metrics['pos_loss'])

            self.history['train_pos_loss'].append(train_metrics['pos_loss'])
            self.history['val_pos_loss'].append(val_metrics['pos_loss'])
            self.history['train_conf_loss'].append(train_metrics['conf_loss'])
            self.history['val_conf_loss'].append(val_metrics['conf_loss'])
            self.history['learning_rate_pos'].append(optimizer_pos.param_groups[0]['lr'])

            print(f"\n  Train Pos Loss: {train_metrics['pos_loss']:.6f}")
            print(f"  Train Conf Loss: {train_metrics['conf_loss']:.6f}")
            print(f"  Val Pos Loss: {val_metrics['pos_loss']:.6f}")
            print(f"  Val Conf Loss: {val_metrics['conf_loss']:.6f}")

            if val_metrics['pos_loss'] < best_pos_loss:
                best_pos_loss = val_metrics['pos_loss']
                torch.save(model.state_dict(), self.checkpoint_dir / 'best_position_model.pth')
                print(f"  ✓ Saved best position model")

            if self.early_stopping.update(val_metrics['pos_loss']):
                print(f"\n🚨 Position path early stopping triggered!")
                break

        # 第二阶段：训练尺寸路径（冻结位置参数）
        print("\n" + "=" * 40)
        print("STAGE 2: Training Size Path")
        print("=" * 40)

        # 加载最佳位置模型
        model.load_state_dict(torch.load(self.checkpoint_dir / 'best_position_model.pth'))

        # 冻结位置相关参数
        for param in model.localization_head.parameters():
            param.requires_grad = False
        for param in model.confidence_head.parameters():
            param.requires_grad = False

        self.early_stopping.set_stage('size')
        self.early_stopping.pos_stopped = True  # 位置路径已停止

        # 尺寸路径优化器（所有参数，但位置参数已冻结）
        optimizer_size = torch.optim.AdamW(model.parameters(), lr=self.config.get('size_lr', 1e-4))
        scheduler_size = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_size, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        best_size_loss = float('inf')

        for epoch in range(self.config.get('size_epochs', 30)):
            train_metrics = self.train_size_epoch(model, train_loader, optimizer_size, epoch)
            val_metrics = self.validate_epoch(model, val_loader)

            scheduler_size.step(val_metrics['size_loss'])

            self.history['train_size_loss'].append(train_metrics['size_loss'])
            self.history['val_size_loss'].append(val_metrics['size_loss'])
            self.history['learning_rate_size'].append(optimizer_size.param_groups[0]['lr'])

            print(f"\n  Train Size Loss: {train_metrics['size_loss']:.6f}")
            print(f"  Val Size Loss: {val_metrics['size_loss']:.6f}")

            if val_metrics['size_loss'] < best_size_loss:
                best_size_loss = val_metrics['size_loss']
                torch.save(model.state_dict(), self.checkpoint_dir / 'best_model.pth')
                print(f"  ✓ Saved best model")

            if self.early_stopping.update(val_metrics['size_loss']):
                print(f"\n🚨 Size path early stopping triggered!")
                break

        # 保存最终模型
        final_model_path = self.final_model_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_pos_loss': best_pos_loss,
            'best_size_loss': best_size_loss
        }, final_model_path)

        print(f"\nTraining completed!")
        print(f"Best position loss: {best_pos_loss:.6f}")
        print(f"Best size loss: {best_size_loss:.6f}")

        return final_model_path

# 12. 测试器
# 12. 测试器（精简版 - 只输出comparison、focused_images、attention_maps、test_report、test_results）
class PositionDecompositionTester:
    """位置分解模型测试器"""

    def __init__(self, model_path, config, cache_session=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_session = cache_session

        self.model = self._load_model(model_path)
        self.model.eval()

        self.output_dir = setup_output_directory(
            "D:/med_data/ai/position_test",
            prefix="pos_test"
        )
        self.output_dir.mkdir(exist_ok=True)

        print(f"Test results will be saved to: {self.output_dir}")

    def _load_model(self, model_path):
        from multi.GCNREG_subimg import PositionDecompositionModel

        checkpoint = torch.load(model_path, map_location=self.device)

        model = PositionDecompositionModel(
            image_size=self.config.get('image_size', 512),
            patch_size=self.config.get('patch_size', 16),
            encoder_base_channels=self.config.get('encoder_base_channels', 32),
            encoder_blocks=self.config.get('encoder_blocks', 2),
            encoder_output_dim=self.config.get('encoder_output_dim', 256)
        ).to(self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"✅ Model loaded successfully")
        return model

    def test(self):
        """测试模型"""
        val_dataset = StreamCoarseExtractorDataset(
            data_dir=self.cache_session,
            split="test",
            augment_for_contrast=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        results = []
        total_pos_error = 0
        total_size_error = 0

        with torch.no_grad():
            test_bar = tqdm(val_loader, desc='Testing')
            for batch in test_bar:
                images = batch['image'].to(self.device)
                outputs = self.model(images)

                positions_pred = outputs['position'].cpu()  # [B, 2]
                sizes_pred = outputs['size'].cpu()  # [B, 1]

                for i in range(len(batch['filename'])):
                    filename = batch['filename'][i] if isinstance(batch['filename'], list) else batch['filename']

                    x_true = batch['width_ratio_x'][i].item()
                    y_true = batch['height_ratio'][i].item()
                    x_pred = positions_pred[i, 0].item()
                    y_pred = positions_pred[i, 1].item()

                    size_true = batch['width_ratio_scaled'][i].item()
                    size_pred = sizes_pred[i].item()

                    results.append({
                        'filename': filename,
                        'x_true': x_true,
                        'x_pred': x_pred,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'size_true': size_true,
                        'size_pred': size_pred,
                        'pos_error': abs(x_pred - x_true) + abs(y_pred - y_true),
                        'size_error': abs(size_pred - size_true)
                    })

                    total_pos_error += abs(x_pred - x_true) + abs(y_pred - y_true)
                    total_size_error += abs(size_pred - size_true)

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        total = len(results)
        print(f"\n{'=' * 60}")
        print("Test Results Summary")
        print(f"{'=' * 60}")
        print(f"Average Position Error: {total_pos_error / total:.4f}")
        print(f"Average Size Error: {total_size_error / total:.4f}")

        return results

# 13. 主函数（修改版）
# 13. 主函数（修改版 - 支持多增强版本对比学习）
def main():
    """主函数"""
    print("=" * 60)
    print("Position Decomposition Model Training Program")
    print("=" * 60)

    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/translate/reverse/train",
        'test_image_dir': "D:/med_data/ai/translate/reverse/test",
        'location_excel_path': "D:/med_data/ai/translate/reverse/location_contrast_size.xlsx",
        'position_excel_path': "D:/med_data/ai/translate/reverse/classify_all_trans_updated.xlsx",

        # 模型参数
        'image_size': 512,
        'patch_size': 16,
        'encoder_base_channels': 32,
        'encoder_blocks': 2,
        'encoder_output_dim': 256,
        'localization_hidden': 128,
        'confidence_hidden': 64,
        'size_hidden': 128,
        'local_window_size': 3,

        # 训练参数
        'batch_size': 8,
        'pos_epochs': 30,
        'size_epochs': 30,
        'pos_lr': 1e-4,
        'size_lr': 1e-4,

        # 早停参数
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,

        # 损失权重
        'pos_weight': 1.0,
        'size_weight': 1.0,
        'conf_weight': 0.5,

        # 其他
        'num_workers': 2,
        'cache_root': "D:/med_data/ai/stream_cache",
        'model_save_root': "D:/med_data/ai/position_models",
    }

    mode = input("选择模式 (1: 训练, 2: 测试, 3: 预处理数据): ").strip()

    if mode == '3':
        # 预处理数据
        preprocessor = DataPreprocessor(
            image_dir=config['train_image_dir'],
            location_excel_path=config['location_excel_path'],
            position_excel_path=config['position_excel_path'],
            image_size=(config['image_size'], config['image_size'])
        )
        cache_session = preprocessor.process_and_save(force_new=True, train_ratio=0.8)
        print(f"数据预处理完成！缓存会话: {cache_session}")

    elif mode == '1':
        # 训练
        cache_session = input("请输入缓存会话路径: ").strip()
        trainer = PositionDecompositionTrainer(config)
        model_path = trainer.train(cache_session)
        print(f"训练完成！模型保存至: {model_path}")

    elif mode == '2':
        # 测试
        model_path = input("请输入模型路径: ").strip()
        cache_session = input("请输入缓存会话路径: ").strip()
        tester = PositionDecompositionTester(model_path, config, cache_session)
        results = tester.test()
        print(f"测试完成！共测试 {len(results)} 个样本")


if __name__ == "__main__":
    main()