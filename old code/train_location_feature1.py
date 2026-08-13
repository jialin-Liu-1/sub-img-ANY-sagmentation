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
from multi.location_net_feature1 import ImprovedCoarseInfoExtractor  # 导入新模型
from multi.GCN_location_net import GraphEnhancedCoarseInfoExtractor

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
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

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
    axes[0, 3].plot(epochs, history.get('train_width_cls_loss', []), 'b-', label='训练宽度分类损失')
    axes[0, 3].plot(epochs, history.get('val_width_cls_loss', []), 'r-', label='验证宽度分类损失')
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('损失')
    axes[0, 3].set_title('宽度分类损失')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # 对比学习损失
    axes[1, 0].plot(epochs, history.get('train_contrast_loss', []), 'g-', label='训练对比损失')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].set_title('对比学习损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 正交性损失
    axes[1, 1].plot(epochs, history.get('train_orthogonal_loss', []), 'g-', label='训练正交损失')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('损失')
    axes[1, 1].set_title('正交性损失')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 分类准确率
    axes[1, 2].plot(epochs, history.get('train_accuracy', []), 'b-', label='训练准确率')
    axes[1, 2].plot(epochs, history.get('val_accuracy', []), 'r-', label='验证准确率')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('准确率')
    axes[1, 2].set_title('宽度分类准确率')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # 学习率
    axes[1, 3].plot(epochs, history['learning_rate'], 'g-', marker='o')
    axes[1, 3].set_xlabel('Epoch')
    axes[1, 3].set_ylabel('学习率')
    axes[1, 3].set_title('学习率调度')
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
        print(f"警告: 未找到图像 {filename} 的类别信息，使用默认类别0")
        position_tensor = torch.zeros(6)
        position_tensor[0] = 1.0  # 默认使用类别0
        return position_tensor, 0, "Segment 1 (default)"

# 5. SizeRangeInfoLoader
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
    """使用流式数据的CoarseExtractor数据集"""

    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 transform=None,
                 augment_for_contrast: bool = False):  # 新增：是否为对比学习生成增强版本
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.augment_for_contrast = augment_for_contrast

        # 获取所有.npz文件
        self.sample_files = list(self.data_dir.glob("*.npz"))
        self.sample_files.sort()

        print(f"加载{split}数据集: {len(self.sample_files)}个样本")
        if augment_for_contrast:
            print(f"  启用对比学习增强")

    def __len__(self):
        return len(self.sample_files)

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        图像增强（只改变整体，不改变血管结构）
        用于对比学习
        """
        # 转换为numpy进行操作
        img_np = image.numpy()  # [1, H, W]

        # 随机选择增强方式
        aug_type = random.choice(['contrast', 'brightness', 'noise', 'blur', 'gamma'])

        if aug_type == 'contrast':
            # 对比度调整
            factor = random.uniform(0.8, 1.2)
            mean = img_np.mean()
            img_np = mean + factor * (img_np - mean)

        elif aug_type == 'brightness':
            # 亮度调整
            delta = random.uniform(-0.1, 0.1)
            img_np = img_np + delta

        elif aug_type == 'noise':
            # 高斯噪声
            noise_std = random.uniform(0, 0.02)
            noise = np.random.randn(*img_np.shape) * noise_std
            img_np = img_np + noise

        elif aug_type == 'blur':
            # 高斯模糊
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(0.3, 0.8)
            img_np[0] = gaussian_filter(img_np[0], sigma=sigma)

        elif aug_type == 'gamma':
            # Gamma校正
            gamma = random.uniform(0.8, 1.2)
            img_np = np.power(np.clip(img_np, 0, 1), gamma)

        # 确保值在0-1范围内
        img_np = np.clip(img_np, 0, 1)

        return torch.from_numpy(img_np).float()

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

        result = {
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

        # 如果需要对比学习的增强版本
        if self.augment_for_contrast:
            # 生成两个不同的增强版本
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
    彻底修复版组合损失函数 - 加入L1损失提高高度重要性
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
                 # L1损失权重（新增）
                 l1_weight: float = 0.5,  # L1损失的权重
                 # 温度参数
                 contrast_temperature: float = 0.1,
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
        self.l1_weight = l1_weight  # 新增L1权重

        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_mix = focal_mix
        self.contrast_temperature = contrast_temperature
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        # 基础损失函数
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')  # 新增L1损失
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        # 用于调试
        self.debug_count = 0

        print(f"\n{'=' * 60}")
        print("彻底修复版组合损失函数初始化:")
        print(f"{'=' * 60}")
        print(f"主任务 - 高度: {height_weight:.2f}, 宽度回归: {width_reg_weight:.2f}, 宽度分类: {width_cls_weight:.2f}")
        print(f"辅助任务 - 对比: {contrast_weight:.2f}, 正交: {orthogonal_weight:.2f}")
        print(f"          动脉瘤: {aneurysm_weight:.2f}, 空间位置: {spatial_weight:.2f}")
        print(f"新增 - L1损失权重: {l1_weight:.2f}")
        print(f"Focal Loss - γ={focal_gamma}, mix={focal_mix}")
        print(f"{'=' * 60}\n")

    def update_epoch(self, epoch: int):
        """更新当前epoch，用于渐进式训练"""
        self.current_epoch = epoch

        # 渐进式调整权重
        progress = epoch / self.total_epochs

        if epoch < 10:
            # 早期：主要关注主任务
            self.contrast_weight = 0.1
            self.orthogonal_weight = 0.05
            self.spatial_weight = 0.1
            self.aneurysm_weight = 0.1
            self.l1_weight = 1  # 早期L1权重较低
        elif epoch < 18:
            # 中期：逐渐增加辅助任务
            self.contrast_weight = 0.3
            self.orthogonal_weight = 0.1
            self.spatial_weight = 0.2
            self.aneurysm_weight = 0.15
            self.l1_weight = 2  # 中期L1权重增加
        else:
            # 后期：使用配置值
            self.contrast_weight = 0.5
            self.orthogonal_weight = 0.15
            self.spatial_weight = 0.3
            self.aneurysm_weight = 0.2
            self.l1_weight = 3.5  # 后期L1权重最大

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                aug_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播 - 加入L1损失
        """
        loss_dict = {}
        device = outputs['height_ratio'].device

        # ========== 1. 计算各分量损失（都是标量）==========

        # 高度损失 - 组合MSE/Focal和L1
        if self.use_focal:
            # Focal MSE部分
            abs_error = torch.abs(outputs['height_ratio'] - targets['height_ratio'])
            mse = (outputs['height_ratio'] - targets['height_ratio']) ** 2
            focal_weight = (abs_error + 1e-8) ** self.focal_gamma
            height_focal_loss = (focal_weight * mse).mean()

            # L1损失部分
            height_l1_loss = self.l1_loss(outputs['height_ratio'], targets['height_ratio'])

            L1_w = self.focal_mix * height_l1_loss

            # 组合高度损失
            height_loss = (1 - self.focal_mix) * height_focal_loss + L1_w
        else:
            # 不使用Focal时，组合MSE和L1
            height_mse_loss = self.mse_loss(outputs['height_ratio'], targets['height_ratio'])
            height_l1_loss = self.l1_loss(outputs['height_ratio'], targets['height_ratio'])

            # 简单加权组合（可以调整比例）
            L1_w = self.l1_weight * height_l1_loss
            height_loss = height_mse_loss + L1_w

        # 宽度回归损失（保持不变）
        width_reg_loss = self.mse_loss(outputs['width_value'], targets['width_ratio_scaled'])

        # 宽度分类损失
        if 'width_logits' in outputs:
            width_cls_loss = self.ce_loss(outputs['width_logits'], targets['size_class'])
        else:
            pred_width_true = outputs['width_value'] / 2.5
            centers = torch.tensor([0.05, 0.15, 0.3375, 0.7375], device=device)
            distances = torch.abs(pred_width_true.unsqueeze(1) - centers.unsqueeze(0))
            logits = -distances / 0.1
            width_cls_loss = self.ce_loss(logits, targets['size_class'])

        # ========== 2. 加权求和 ==========
        total_loss = (self.height_weight * height_loss +
                      self.width_reg_weight * width_reg_loss +
                      self.width_cls_weight * width_cls_loss)

        # 记录主任务损失分量（未加权）
        loss_dict['height_raw'] = height_loss.item()
        loss_dict['width_reg_raw'] = width_reg_loss.item()
        loss_dict['width_cls_raw'] = width_cls_loss.item()
        loss_dict['main_weighted'] = total_loss.item()
        loss_dict['L1'] = L1_w.item()

        # ========== 3. 辅助任务损失 ==========

        # 对比损失
        if aug_outputs is not None and 'disentangled_features' in outputs:
            contrast_loss = 0
            n_features = len(outputs['disentangled_features'])
            for i in range(n_features):
                f1 = F.normalize(outputs['disentangled_features'][i], dim=1)
                f2 = F.normalize(aug_outputs['disentangled_features'][i], dim=1)
                sim_matrix = torch.mm(f1, f2.T) / self.contrast_temperature
                labels = torch.arange(sim_matrix.size(0)).to(device)
                contrast_loss += (F.cross_entropy(sim_matrix, labels) +
                                  F.cross_entropy(sim_matrix.T, labels)) / 2
            contrast_loss = contrast_loss / n_features

            total_loss += self.contrast_weight * contrast_loss
            loss_dict['contrast_raw'] = contrast_loss.item()

        # 正交损失
        if 'disentangled_features' in outputs:
            orth_loss = 0
            n = len(outputs['disentangled_features'])
            if n > 1:
                for i in range(n):
                    for j in range(i + 1, n):
                        f1 = F.normalize(outputs['disentangled_features'][i], dim=1)
                        f2 = F.normalize(outputs['disentangled_features'][j], dim=1)
                        corr = torch.mm(f1, f2.T)
                        orth_loss += corr.abs().mean()
                orth_loss = orth_loss / (n * (n - 1) / 2)

            total_loss += self.orthogonal_weight * orth_loss
            loss_dict['orthogonal_raw'] = orth_loss.item()

        # 空间损失（如果有）
        if self.spatial_weight > 0 and 'attention_map' in outputs:
            # 简化的空间损失
            spatial_loss = torch.tensor(0.0, device=device)
            total_loss += self.spatial_weight * spatial_loss
            loss_dict['spatial_raw'] = 0.0

        # 动脉瘤损失
        if (self.aneurysm_weight > 0 and aug_outputs is not None and
                'aneurysm_features' in outputs and 'aneurysm_features' in aug_outputs):
            f1 = F.normalize(outputs['aneurysm_features'], dim=1)
            f2 = F.normalize(aug_outputs['aneurysm_features'], dim=1)
            sim = (f1 * f2).sum(dim=1).mean()
            aneurysm_loss = 1 - sim
            total_loss += self.aneurysm_weight * aneurysm_loss
            loss_dict['aneurysm_raw'] = aneurysm_loss.item()

        # 记录总损失
        loss_dict['total'] = total_loss.item()

        # ========== 4. 计算准确率 ==========
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

        # ========== 5. 调试输出 ==========
        self.debug_count += 1

        return total_loss, loss_dict

# 10. 数据增强模块
class ContrastiveAugmentation:
    """对比学习数据增强 - 只改变整体，不改变血管结构"""

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        应用随机增强
        image: [B, C, H, W] 或 [C, H, W]
        """
        # 确保是4D张量
        if image.dim() == 3:
            image = image.unsqueeze(0)
            return_squeeze = True
        else:
            return_squeeze = False

        batch_size = image.shape[0]
        augmented = []

        for i in range(batch_size):
            img = image[i]  # [C, H, W]

            # 随机选择增强类型
            aug_type = random.choice(['contrast', 'brightness', 'noise', 'blur', 'gamma', 'none'])

            if aug_type == 'contrast':
                # 对比度调整
                factor = random.uniform(0.7, 1.3)
                mean = img.mean()
                img_aug = mean + factor * (img - mean)

            elif aug_type == 'brightness':
                # 亮度调整
                delta = random.uniform(-0.15, 0.15)
                img_aug = img + delta

            elif aug_type == 'noise':
                # 高斯噪声
                noise_std = random.uniform(0, 0.03)
                noise = torch.randn_like(img) * noise_std
                img_aug = img + noise

            elif aug_type == 'blur':
                # 高斯模糊 (使用简单平均)
                kernel_size = random.choice([3, 5])
                padding = kernel_size // 2
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size * kernel_size)
                img_aug = F.conv2d(img.unsqueeze(0), kernel, padding=padding).squeeze(0)

            elif aug_type == 'gamma':
                # Gamma校正
                gamma = random.uniform(0.7, 1.3)
                img_aug = torch.pow(torch.clamp(img, 0, 1), gamma)

            else:  # 'none'
                img_aug = img.clone()

            # 确保值在合理范围
            img_aug = torch.clamp(img_aug, 0, 1)
            augmented.append(img_aug)

        result = torch.stack(augmented)

        if return_squeeze:
            result = result.squeeze(0)

        return result

# 11. 训练器
class CoarseExtractorTrainer:
    """ImprovedCoarseInfoExtractor训练器 - 修复版"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


        # 创建输出目录
        self.output_dir = setup_output_directory(
            config.get('model_save_root', "D:/med_data/ai/pre_loc"),
            prefix="train"
        )

        # 子目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        self.final_model_dir = self.output_dir / "final_models"

        for dir_path in [self.checkpoint_dir, self.results_dir, self.plots_dir, self.final_model_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"Model save directory: {self.output_dir}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_height_loss': [],
            'train_width_reg_loss': [],
            'train_width_cls_loss': [],
            'train_contrast_loss': [],
            'train_orthogonal_loss': [],
            'train_aneurysm_loss': [],
            'val_height_loss': [],
            'val_width_reg_loss': [],
            'val_width_cls_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # 数据增强模块
        self.augmentation = ContrastiveAugmentation(device=self.device)

        # 当前epoch
        self.current_epoch = 0

        # 只在这里创建一次损失函数！
        self.criterion = self._create_criterion()

        print(f"\n{'=' * 60}")
        print("Trainer initialization completed")
        print(f"{'=' * 60}")

    def _create_model(self):
        """创建新模型"""
        """model = ImprovedCoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config.get('base_channels', 32),
            num_position_classes=6,
            num_width_classes=4,
            num_disentangled_features=self.config.get('num_disentangled_features', 6),
            dropout_rate=self.config.get('dropout_rate', 0.2),
            use_soft_label=self.config.get('use_soft_label', True)
        ).to(self.device)"""

        model = GraphEnhancedCoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config.get('base_channels', 64),
            num_position_classes=6,
            num_width_classes=4,
            num_disentangled_features=6,
            dropout_rate=self.config.get('dropout_rate', 0.2),
            use_soft_label=True,
            use_gcn=True,
            gcn_hidden_dim=192,
            gcn_out_dim=192
        ).to(self.device)

        # 打印模型参数量分析
        total_params = 0
        print("\nModel parameter analysis:")
        print("-" * 50)
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            print(f"  {name}: {params / 1e6:.2f}M ({params:,})")
        print("-" * 50)
        print(f"  Total: {total_params / 1e6:.2f}M ({total_params:,})")

        return model

    def save_checkpoint(self, model, optimizer, scheduler, epoch, train_metrics, val_metrics, is_best=False):
        """保存检查点（只保存权重）"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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

    def save_final_model(self, model, best_val_loss, best_val_acc, epoch):
        """保存完整的最终模型（包括结构和权重）"""
        print(f"\nSaving complete final model...")

        # 将模型移至CPU再保存，确保跨平台兼容性
        model_cpu = model.to('cpu')

        final_model_path = self.final_model_dir / 'final_complete_model.pth'

        # 保存完整的模型对象
        torch.save({
            'model': model_cpu,  # 完整的模型对象
            'model_state_dict': model.state_dict(),  # 保留权重以便向后兼容
            'config': self.config,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'history': self.history
        }, final_model_path)

        # 将模型移回原设备
        model.to(self.device)

        print(f"✅ Complete model saved to: {final_model_path}")
        return final_model_path

    def _create_criterion(self):
        """创建损失函数 - 只调用一次"""
        return CombinedNewModelLoss(
            # 主任务权重
            height_weight=self.config.get('height_weight', 1.2),
            width_reg_weight=self.config.get('width_reg_weight', 0.3),
            width_cls_weight=self.config.get('width_cls_weight', 1.5),

            # 辅助任务权重
            contrast_weight=self.config.get('contrast_weight', 0.15),
            orthogonal_weight=self.config.get('orthogonal_weight', 0.08),
            aneurysm_weight=self.config.get('aneurysm_weight', 0.1),
            spatial_weight=self.config.get('spatial_weight', 0.05),

            # Focal Loss参数
            use_focal=self.config.get('use_focal', True),
            focal_gamma=self.config.get('focal_gamma', 2.0),
            focal_mix=self.config.get('focal_mix', 0.3),

            # 温度参数
            contrast_temperature=self.config.get('contrast_temperature', 0.1),

            # 训练参数
            total_epochs=self.config.get('num_epochs', 100)
        )

    def _create_dataloaders(self, cache_session: str):
        """从缓存创建数据加载器"""
        # 训练集（需要增强用于对比学习）
        train_dataset = StreamCoarseExtractorDataset(
            data_dir=cache_session,
            split="train",
            augment_for_contrast=True
        )

        # 验证集（不需要增强）
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

    def train_epoch(self, model, train_loader, optimizer, epoch):
        """训练一个epoch - 使用self.criterion"""
        model.train()
        self.criterion.update_epoch(epoch)  # 让损失函数自己处理渐进式

        train_loss_total = 0
        train_height_loss = 0
        train_width_reg_loss = 0
        train_width_cls_loss = 0
        train_contrast_loss = 0
        train_orthogonal_loss = 0
        train_aneurysm_loss = 0
        train_accuracy = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Train]')
        for batch_idx, batch in enumerate(train_bar):
            # 将数据移动到设备
            images = batch['image'].to(self.device)
            positions = batch['position'].to(self.device)

            # 生成增强版本
            #images_aug1 = self.augmentation(images)
            images_aug2 = self.augmentation(images)

            targets = {
                'height_ratio': batch['height_ratio'].to(self.device),
                'width_ratio_true': batch['width_ratio_true'].to(self.device),
                'width_ratio_scaled': batch['width_ratio_scaled'].to(self.device),
                'size_class': batch['size_class'].to(self.device)
            }

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, positions)
            aug_outputs2 = model(images_aug2, positions)

            # 计算损失
            loss, loss_dict = self.criterion(outputs, targets, aug_outputs2)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录损失（使用原始值，不是加权后的）
            train_loss_total += loss_dict['total']
            train_height_loss += loss_dict.get('height_raw', loss_dict.get('height', 0))
            train_width_reg_loss += loss_dict.get('width_reg_raw', loss_dict.get('width_reg', 0))
            train_width_cls_loss += loss_dict.get('width_cls_raw', loss_dict.get('width_cls', 0))
            train_contrast_loss += loss_dict.get('contrast_raw', 0)
            train_orthogonal_loss += loss_dict.get('orthogonal_raw', 0)
            train_aneurysm_loss += loss_dict.get('aneurysm_raw', 0)
            train_accuracy += loss_dict['accuracy']

            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{loss_dict["total"]:.2f}',
                'acc': f'{loss_dict["accuracy"]:.2%}',
                'contrast': f'{loss_dict.get("contrast_raw", 0): .2f}',
                'L1': f'{loss_dict["L1"]:.2f}'
            })

        # 计算平均损失
        num_batches = len(train_loader)
        avg_metrics = {
            'total': train_loss_total / num_batches,
            'height': train_height_loss / num_batches,
            'width_reg': train_width_reg_loss / num_batches,
            'width_cls': train_width_cls_loss / num_batches,
            'contrast': train_contrast_loss / num_batches,
            'orthogonal': train_orthogonal_loss / num_batches,
            'aneurysm': train_aneurysm_loss / num_batches,
            'accuracy': train_accuracy / num_batches
        }

        return avg_metrics

    def validate_epoch(self, model, val_loader):
        """验证一个epoch - 使用self.criterion但禁用辅助损失"""
        model.eval()

        # 临时保存辅助损失权重
        temp_weights = {
            'contrast': self.criterion.contrast_weight,
            'orthogonal': self.criterion.orthogonal_weight,
            'aneurysm': self.criterion.aneurysm_weight,
            'spatial': self.criterion.spatial_weight
        }

        # 验证时禁用辅助损失
        self.criterion.contrast_weight = 0
        self.criterion.orthogonal_weight = 0
        self.criterion.aneurysm_weight = 0
        self.criterion.spatial_weight = 0

        val_loss_total = 0
        val_height_loss = 0
        val_width_reg_loss = 0
        val_width_cls_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='[Validation]')
            for batch in val_bar:
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)

                targets = {
                    'height_ratio': batch['height_ratio'].to(self.device),
                    'width_ratio_true': batch['width_ratio_true'].to(self.device),
                    'width_ratio_scaled': batch['width_ratio_scaled'].to(self.device),
                    'size_class': batch['size_class'].to(self.device)
                }

                outputs = model(images, positions)
                loss, loss_dict = self.criterion(outputs, targets, aug_outputs=None)

                val_loss_total += loss_dict['total']
                val_height_loss += loss_dict.get('height_raw', loss_dict.get('height', 0))
                val_width_reg_loss += loss_dict.get('width_reg_raw', loss_dict.get('width_reg', 0))
                val_width_cls_loss += loss_dict.get('width_cls_raw', loss_dict.get('width_cls', 0))
                val_accuracy += loss_dict['accuracy']

                val_bar.set_postfix({
                    'loss': f'{loss_dict["total"]:.2f}',
                    'acc': f'{loss_dict["accuracy"]:.2%}'
                })

        # 恢复辅助损失权重
        self.criterion.contrast_weight = temp_weights['contrast']
        self.criterion.orthogonal_weight = temp_weights['orthogonal']
        self.criterion.aneurysm_weight = temp_weights['aneurysm']
        self.criterion.spatial_weight = temp_weights['spatial']

        num_batches = len(val_loader)
        avg_metrics = {
            'total': val_loss_total / num_batches,
            'height': val_height_loss / num_batches,
            'width_reg': val_width_reg_loss / num_batches,
            'width_cls': val_width_cls_loss / num_batches,
            'accuracy': val_accuracy / num_batches
        }

        return avg_metrics

    def train(self, cache_session: str):
        """训练模型 - 训练完成后直接测试"""
        print("\n" + "=" * 60)
        print("Starting ImprovedCoarseInfoExtractor Training")
        print("=" * 60)

        # 创建模型
        model = self._create_model()

        # 创建数据加载器
        train_loader, val_loader = self._create_dataloaders(cache_session)

        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # 早停机制
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=self.config.get('early_stopping_min_delta', 0.001)
        )

        # 训练循环
        best_val_loss = float('inf')
        best_val_acc = 0
        final_epoch = 0
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        best_model_state = None  # 保存最佳模型的状态

        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)

            # 验证
            val_metrics = self.validate_epoch(model, val_loader)

            # 更新学习率调度器
            scheduler.step(val_metrics['total'])
            current_lr = optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['total'])
            self.history['train_height_loss'].append(train_metrics['height'])
            self.history['train_width_reg_loss'].append(train_metrics['width_reg'])
            self.history['train_width_cls_loss'].append(train_metrics['width_cls'])
            self.history['train_contrast_loss'].append(train_metrics.get('contrast', 0))
            self.history['train_orthogonal_loss'].append(train_metrics.get('orthogonal', 0))
            self.history['train_aneurysm_loss'].append(train_metrics.get('aneurysm', 0))
            self.history['val_height_loss'].append(val_metrics['height'])
            self.history['val_width_reg_loss'].append(val_metrics['width_reg'])
            self.history['val_width_cls_loss'].append(val_metrics['width_cls'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)

            # 打印统计信息
            print(f"  Train Total Loss: {train_metrics['total']:.4f}")
            print(f"    Height Loss: {train_metrics['height']:.4f}")
            print(f"    Width Classification: {train_metrics['width_cls']:.4f}")
            print(f"    Width Regression: {train_metrics['width_reg']:.4f}")
            print(f"    Train Accuracy: {train_metrics['accuracy']:.2%}")
            print(f"  Validation Total Loss: {val_metrics['total']:.4f}")
            print(f"    Validation Accuracy: {val_metrics['accuracy']:.2%}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # 保存最佳模型（同时保存状态到内存）
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                best_val_acc = val_metrics['accuracy']
                final_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                # 保存到文件
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': self.config,
                }, best_model_path)
                print(f"  ✓ Updated best model (Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.2%})")

            # 每10个epoch保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': self.config,
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint to: {checkpoint_path}")

            # 早停检查
            if early_stopping(val_metrics['total']):
                print(f"\n🚨 Early stopping triggered! No improvement for {early_stopping.patience} epochs")
                break

        # ========== 使用内存中的最佳模型进行测试 ==========
        print("\n" + "=" * 60)
        print("Step 2: Testing best model directly from memory")
        print("=" * 60)

        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        # 设置为eval模式
        model.eval()

        # 直接在这里实现测试逻辑，而不是通过测试器
        self._test_model_directly(model, cache_session)

        # 保存完整的最终模型
        final_complete_model_path = self.save_final_model(model, best_val_loss, best_val_acc, final_epoch)

        # 保存最终模型（仅权重）
        final_model_path = self.checkpoint_dir / 'final_model.pth'
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }, final_model_path)

        print(f"\nTraining completed! Total epochs: {final_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        print(f"Best model saved to: {best_model_path}")
        print(f"Complete model saved to: {final_complete_model_path}")
        print(f"Weights-only model saved to: {final_model_path}")

        # 绘制训练曲线
        plot_training_curves(self.history, self.plots_dir)

        # 保存训练历史
        save_training_history(self.history, self.output_dir, self.config)

        return final_complete_model_path

    def _test_model_directly(self, model, cache_session):
        """直接在训练器中测试模型 - 生成对比图和聚焦图像"""
        print(f"\nStarting test on validation data from cache: {cache_session}")
        print("=" * 60)

        # 创建数据集
        val_dataset = StreamCoarseExtractorDataset(
            data_dir=cache_session,
            split="test",
            augment_for_contrast=False
        )

        print(f"Loaded {len(val_dataset)} validation samples from cache")

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        # 创建输出目录
        test_output_dir = self.output_dir / "direct_test_results"
        masks_dir = test_output_dir / "masks"
        focused_dir = test_output_dir / "focused_images"
        comparison_dir = test_output_dir / "comparison"
        overlay_dir = test_output_dir / "overlay"
        standard_masks_dir = test_output_dir / "standard_masks"

        for dir_path in [masks_dir, focused_dir, comparison_dir, overlay_dir, standard_masks_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        print(f"Test results will be saved to: {test_output_dir}")

        # 统计信息
        stats = {
            'total': 0,
            'size_correct': 0,
            'height_error_sum': 0,
            'width_error_sum': 0,
            'height_correct_threshold': 0,
            'width_correct_threshold': 0,
            'both_correct_threshold': 0,
            'class_0_correct': 0,
            'class_0_total': 0,
            'class_1_correct': 0,
            'class_1_total': 0,
            'class_2_correct': 0,
            'class_2_total': 0,
            'class_3_correct': 0,
            'class_3_total': 0,
        }

        # 创建注意力掩码生成器
        mask_generator = AttentionMaskGenerator(
            image_size=self.config['image_size']
        ).to(self.device)

        class_names = ['tiny', 'small', 'medium', 'large']
        error_threshold = 0.05

        # 存储结果用于后续分析
        results = []

        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Testing progress")):
            images = batch['image'].to(self.device)
            positions = batch['position'].to(self.device)

            # ========== 关键修复：处理batch中的每个样本 ==========
            batch_size = images.size(0)

            with torch.no_grad():
                outputs = model(images, positions)
                height_preds = outputs['height_ratio']  # [batch_size]
                width_preds_scaled = outputs['width_value']  # [batch_size]
                width_preds_true = width_preds_scaled / 2.5  # [batch_size]

            # 遍历batch中的每个样本
            for i in range(batch_size):
                # 获取单个样本的值
                height_true = batch['height_ratio'][i].item()
                width_true = batch['width_ratio_true'][i].item()
                size_class_true = batch['size_class'][i].item()

                # 获取文件名（需要处理列表情况）
                if isinstance(batch['filename'], list):
                    filename = batch['filename'][i]
                else:
                    filename = batch['filename'][i] if batch['filename'].dim() > 0 else batch['filename']

                basename = os.path.splitext(filename)[0]

                # 获取其他元数据（需要处理列表/张量情况）
                position_name = batch.get('position_name', ['Unknown'])[0]
                if isinstance(position_name, (list, tuple)):
                    position_name = position_name[i] if i < len(position_name) else position_name[0]

                size_category = batch.get('size_category', ['medium'])[0]
                if isinstance(size_category, (list, tuple)):
                    size_category = size_category[i] if i < len(size_category) else size_category[0]

                # 获取当前样本的预测值
                height_pred = height_preds[i].item()
                width_pred_scaled = width_preds_scaled[i].item()
                width_pred_true = width_preds_true[i].item()

                # 计算误差
                height_error = abs(height_pred - height_true)
                width_error = abs(width_pred_true - width_true)

                # 阈值判断
                height_correct = height_error < error_threshold
                width_correct = width_error < error_threshold
                both_correct = height_correct and width_correct

                # 分类判断
                if width_pred_true < 0.1:
                    pred_class = 0
                elif width_pred_true < 0.2:
                    pred_class = 1
                elif width_pred_true < 0.475:
                    pred_class = 2
                else:
                    pred_class = 3

                pred_class_name = class_names[pred_class]
                true_class_name = class_names[size_class_true]
                size_correct = (pred_class == size_class_true)

                # 生成注意力掩膜（使用单样本）
                image_np = images[i, 0].cpu().numpy()

                height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
                width_tensor_pred = torch.tensor([width_pred_true], dtype=torch.float32).to(self.device)
                attention_mask_pred = mask_generator(height_tensor_pred, width_tensor_pred)
                attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

                # 生成标准注意力掩膜
                height_tensor_gt = torch.tensor([height_true], dtype=torch.float32).to(self.device)
                width_tensor_gt = torch.tensor([width_true], dtype=torch.float32).to(self.device)
                attention_mask_gt = mask_generator(height_tensor_gt, width_tensor_gt)
                attention_mask_gt_np = attention_mask_gt.squeeze().cpu().numpy()

                # 生成聚焦图像
                pred_focused = image_np * attention_mask_pred_np
                gt_focused = image_np * attention_mask_gt_np

                # ========== 保存图像（每个样本单独保存）==========
                # 保存原始图像
                save_image(image_np, test_output_dir / f"{basename}_original.png")

                # 保存预测掩膜
                pred_mask_uint8 = (attention_mask_pred_np * 255).astype(np.uint8)
                save_image(pred_mask_uint8, masks_dir / f"{basename}_pred_mask.png", normalize=False)

                # 保存标准掩膜
                gt_mask_uint8 = (attention_mask_gt_np * 255).astype(np.uint8)
                save_image(gt_mask_uint8, standard_masks_dir / f"{basename}_standard_mask.png", normalize=False)

                # 保存聚焦图像
                save_image(pred_focused, focused_dir / f"{basename}_pred_focused.png")
                save_image(gt_focused, focused_dir / f"{basename}_standard_focused.png")

                # ========== 创建对比图 ==========
                self._create_direct_comparison_plot(
                    basename=basename,
                    image=image_np,
                    pred_mask=attention_mask_pred_np,
                    gt_mask=attention_mask_gt_np,
                    height_pred=height_pred,
                    width_pred=width_pred_true,
                    height_true=height_true,
                    width_true=width_true,
                    position_name=position_name,
                    size_category=size_category,
                    size_correct=size_correct,
                    pred_class_name=pred_class_name,
                    true_class_name=true_class_name,
                    height_correct=height_correct,
                    width_correct=width_correct,
                    error_threshold=error_threshold,
                    save_dir=comparison_dir
                )

                # 更新统计
                stats['total'] += 1
                stats['height_error_sum'] += height_error
                stats['width_error_sum'] += width_error
                stats[f'class_{size_class_true}_total'] += 1

                if size_correct:
                    stats['size_correct'] += 1
                    stats[f'class_{size_class_true}_correct'] += 1
                if height_correct:
                    stats['height_correct_threshold'] += 1
                if width_correct:
                    stats['width_correct_threshold'] += 1
                if both_correct:
                    stats['both_correct_threshold'] += 1

                # 保存结果
                results.append({
                    'filename': filename,
                    'height_pred': height_pred,
                    'width_pred': width_pred_true,
                    'height_true': height_true,
                    'width_true': width_true,
                    'height_error': height_error,
                    'width_error': width_error,
                    'pred_class': pred_class,
                    'true_class': size_class_true,
                    'size_correct': size_correct,
                })

        # 保存结果到CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(test_output_dir / 'test_results.csv', index=False)

        # 打印结果
        total = stats['total']
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        print(f"Threshold accuracy (error < {error_threshold}):")
        print(
            f"  Height: {stats['height_correct_threshold'] / total:.2%} ({stats['height_correct_threshold']}/{total})")
        print(f"  Width: {stats['width_correct_threshold'] / total:.2%} ({stats['width_correct_threshold']}/{total})")
        print(f"  Both: {stats['both_correct_threshold'] / total:.2%} ({stats['both_correct_threshold']}/{total})")
        print(f"\nSize classification accuracy: {stats['size_correct'] / total:.2%} ({stats['size_correct']}/{total})")

        # 各类别准确率
        print("\nPer-class accuracy:")
        for i, class_name in enumerate(class_names):
            if stats[f'class_{i}_total'] > 0:
                acc = stats[f'class_{i}_correct'] / stats[f'class_{i}_total']
                print(f"  {class_name}: {acc:.2%} ({stats[f'class_{i}_correct']}/{stats[f'class_{i}_total']})")

        print(f"\nAverage height error: {stats['height_error_sum'] / total:.4f}")
        print(f"Average width error: {stats['width_error_sum'] / total:.4f}")
        print(f"\nDetailed results saved to: {test_output_dir}")

    def _create_direct_comparison_plot(self, basename, image, pred_mask, gt_mask,
                                       height_pred, width_pred, height_true, width_true,
                                       position_name, size_category, size_correct,
                                       pred_class_name, true_class_name,
                                       height_correct, width_correct, error_threshold, save_dir):
        """创建对比图 - 与测试器保持一致的样式"""

        # 计算状态显示
        height_status = f"✓ <{error_threshold}" if height_correct else f"✗ >{error_threshold}"
        width_status = f"✓ <{error_threshold}" if width_correct else f"✗ >{error_threshold}"

        # 准备图像列表
        images_to_plot = [image, pred_mask, gt_mask]

        # 准备标题
        titles = [
            'Original Image',
            f'Predicted Mask\nHeight: {height_pred:.3f} ({height_status})\nWidth: {width_pred:.3f} ({width_status})',
            f'Ground Truth Mask\nHeight: {height_true:.3f}\nWidth: {width_true:.3f}'
        ]

        # 总标题
        suptitle = (f"{basename}\nAneurysm Type: {position_name}\n"
                    f"Size: {size_category}\n"
                    f"Classification: {pred_class_name} vs {true_class_name} - {'✓ Correct' if size_correct else '✗ Wrong'}")

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (img, title) in enumerate(zip(images_to_plot, titles)):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')

        plt.suptitle(suptitle, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / f"{basename}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

# 12. 测试器
class CoarseExtractorTester:
    """ImprovedCoarseInfoExtractor测试器 - 使用验证集缓存数据测试"""

    def __init__(self, model_path, config, cache_session=None, mask_dir=None):
        """
        初始化测试器

        Args:
            model_path: 模型路径（可以是完整模型或权重文件）
            config: 配置参数
            cache_session: 验证集缓存会话路径（如果为None，则从model_path中读取）
            mask_dir: 动脉瘤掩膜目录路径（可选）
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_session = cache_session
        self.mask_dir = Path(mask_dir) if mask_dir else None

        # 加载模型（兼容两种格式）
        self.model = self._load_model(model_path)
        self.model.eval()

        # 创建注意力掩码生成器
        self.mask_generator = AttentionMaskGenerator(
            image_size=config['image_size']
        ).to(self.device)

        # 分类边界和类别名称
        self.class_boundaries = [0.1, 0.2, 0.475, 1.0]
        self.class_names = ['tiny', 'small', 'medium', 'large']
        self.num_classes = len(self.class_names)

        # 误差阈值（用于计算准确率）
        self.error_threshold = 0.05

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

        print(f"Test results will be saved to: {self.output_dir}")
        if self.cache_session:
            print(f"Using cache session: {self.cache_session}")
        if self.mask_dir:
            print(f"Using aneurysm mask directory: {self.mask_dir}")
        print(f"Classification boundaries (true values): {self.class_boundaries}")
        print(f"Class names: {self.class_names}")
        print(f"Threshold accuracy: Error < {self.error_threshold} considered correct")

    def _load_model(self, model_path):
        """加载训练好的模型 - 兼容两种格式：完整模型或权重文件"""
        print(f"\nLoading model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)


        # 方法1：如果是完整模型（包含'model'键）
        if 'model' in checkpoint:
            model = checkpoint['model']
            model = model.to(self.device)
            print(f"✅ Loaded complete model from checkpoint")

            # 获取训练信息
            epoch = checkpoint.get('epoch', 'Unknown')
            val_loss = checkpoint.get('best_val_loss', None)
            val_acc = checkpoint.get('best_val_acc', None)

        # 方法2：如果是权重文件（只包含state_dict）
        elif 'model_state_dict' in checkpoint:
            print(f"Loading weights from checkpoint...")

            # 从checkpoint中获取训练时的配置
            if 'config' in checkpoint:
                train_config = checkpoint['config']
                print(f"✅ Loaded training configuration from checkpoint")
            else:
                train_config = self.config
                print(f"⚠️ Using current configuration")

            # 打印训练配置
            print(f"Training configuration:")
            print(f"  base_channels: {train_config.get('base_channels', 32)}")
            print(f"  num_disentangled_features: {train_config.get('num_disentangled_features', 6)}")
            print(f"  dropout_rate: {train_config.get('dropout_rate', 0.2)}")

            """# 使用训练时的配置创建模型
            model = ImprovedCoarseInfoExtractor(
                image_size=train_config.get('image_size', (512, 512)),
                base_channels=train_config.get('base_channels', 32),
                num_position_classes=6,
                num_width_classes=4,
                num_disentangled_features=train_config.get('num_disentangled_features', 6),
                dropout_rate=train_config.get('dropout_rate', 0.2),
                use_soft_label=train_config.get('use_soft_label', True)
            ).to(self.device)"""
            # 创建模型
            model = GraphEnhancedCoarseInfoExtractor(
                image_size=train_config.get('image_size', (512, 512)),
                base_channels=train_config.get('base_channels', 64),
                num_position_classes=6,
                num_width_classes=4,
                num_disentangled_features=6,
                dropout_rate=train_config.get('dropout_rate', 0.2),
                use_soft_label=True,
                use_gcn=True,
                gcn_hidden_dim=192,
                gcn_out_dim=192
            ).to(self.device)


            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])

            # 获取训练信息
            epoch = checkpoint.get('epoch', 'Unknown')
            val_loss = checkpoint.get('best_val_loss', None)
            val_acc = checkpoint.get('best_val_acc', None)

            print(f"✅ Loaded weights from checkpoint")

        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")

        print(f"\n✅ Model loaded successfully!")
        print(f"   Training epochs: {epoch}")
        if val_loss:
            print(f"   Best validation loss: {val_loss:.4f}")
        if val_acc:
            print(f"   Best validation accuracy: {val_acc:.2%}")

        return model

    def _load_aneurysm_mask(self, filename: str) -> Optional[np.ndarray]:
        """加载动脉瘤掩膜图像"""
        if self.mask_dir is None:
            return None

        try:
            basename = os.path.splitext(filename)[0]

            # 可能的掩膜文件名格式
            possible_names = [
                f"{basename}.tif",
                f"{basename}.tiff",
                f"{basename}.png",
                f"{basename}.jpg",
                f"{basename}_mask.tif",
                f"{basename}_mask.tiff",
                f"{basename}_mask.png"
            ]

            mask_path = None
            for mask_name in possible_names:
                test_path = self.mask_dir / mask_name
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
            print(f"Failed to load aneurysm mask for {filename}: {e}")
            return None

    def test_from_cache(self):
        """
        从缓存会话中读取验证集数据进行测试
        完全复用训练过程中的验证数据
        """
        if self.cache_session is None:
            raise ValueError("Cache session path must be provided")

        print(f"\nStarting test on validation data from cache: {self.cache_session}")
        print("=" * 60)


        # 创建验证集数据集（与训练时完全一致，不需要增强）
        val_dataset = StreamCoarseExtractorDataset(
            data_dir=self.cache_session,
            split="test",
            augment_for_contrast=False
        )

        print(f"Loaded {len(val_dataset)} validation samples from cache")

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        results = []
        stats = {
            'total': 0,
            'size_correct': 0,
            'height_error_sum': 0,
            'width_error_sum': 0,
            'height_correct_threshold': 0,
            'width_correct_threshold': 0,
            'both_correct_threshold': 0,
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

        # 创建保存所有对比图的目录
        all_comparisons_dir = self.comparison_dir / "all_samples"
        all_comparisons_dir.mkdir(exist_ok=True)
        # 在测试开始前，取第一个batch
        first_batch = next(iter(val_loader))
        images = first_batch['image'].to(self.device)
        positions = first_batch['position'].to(self.device)

        with torch.no_grad():
            outputs = self.model(images, positions)

        print(f"\nFirst batch predictions:")
        print(f"  height_pred: {outputs['height_ratio'][0].item():.4f}")
        print(f"  width_value: {outputs['width_value'][0].item():.4f}")
        print(f"  width_logits: {outputs['width_logits'][0].cpu().numpy()}")

        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Testing progress")):
            images = batch['image'].to(self.device)
            positions = batch['position'].to(self.device)

            # ========== 关键修复：处理batch中的每个样本 ==========
            batch_size = images.size(0)

            # 模型推理（整个batch一起推理，提高效率）
            with torch.no_grad():
                outputs = self.model(images, positions)
                height_preds = outputs['height_ratio']  # [batch_size]
                width_preds_scaled = outputs['width_value']  # [batch_size]
                width_preds_true = width_preds_scaled / 2.5  # [batch_size]

            # 遍历batch中的每个样本
            for i in range(batch_size):
                # 获取真实值
                height_true = batch['height_ratio'][i].item()
                width_true = batch['width_ratio_true'][i].item()
                width_scaled_true = batch['width_ratio_scaled'][i].item()
                size_class_true = batch['size_class'][i].item()

                # 获取文件名（需要处理列表情况）
                if isinstance(batch['filename'], list):
                    filename = batch['filename'][i]
                else:
                    filename = batch['filename'][i] if batch['filename'].dim() > 0 else batch['filename']

                # 获取其他元数据（需要处理列表/张量情况）
                position_name = batch.get('position_name', ['Unknown'])[0]
                if isinstance(position_name, (list, tuple)):
                    position_name = position_name[i] if i < len(position_name) else position_name[0]

                size_category = batch.get('size_category', ['medium'])[0]
                if isinstance(size_category, (list, tuple)):
                    size_category = size_category[i] if i < len(size_category) else size_category[0]

                # 获取当前样本的预测值
                height_pred = height_preds[i].item()
                width_pred_scaled = width_preds_scaled[i].item()
                width_pred_true = width_preds_true[i].item()

                # 计算误差和统计
                height_error = abs(height_pred - height_true)
                width_error = abs(width_pred_true - width_true)

                height_correct_threshold = height_error < self.error_threshold
                width_correct_threshold = width_error < self.error_threshold
                both_correct_threshold = height_correct_threshold and width_correct_threshold

                # 预测类别
                if width_pred_true < 0.1:
                    pred_class = 0
                elif width_pred_true < 0.2:
                    pred_class = 1
                elif width_pred_true < 0.475:
                    pred_class = 2
                else:
                    pred_class = 3

                pred_class_name = self.class_names[pred_class]
                true_class_name = self.class_names[size_class_true]
                size_correct = (pred_class == size_class_true)

                # 生成注意力掩膜（使用单样本）
                image_np = images[i, 0].cpu().numpy()
                height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
                width_tensor_pred = torch.tensor([width_pred_true], dtype=torch.float32).to(self.device)
                attention_mask_pred = self.mask_generator(height_tensor_pred, width_tensor_pred)
                attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

                # 生成标准注意力掩膜
                height_tensor_gt = torch.tensor([height_true], dtype=torch.float32).to(self.device)
                width_tensor_gt = torch.tensor([width_true], dtype=torch.float32).to(self.device)
                attention_mask_gt = self.mask_generator(height_tensor_gt, width_tensor_gt)
                attention_mask_gt_np = attention_mask_gt.squeeze().cpu().numpy()

                # 加载动脉瘤掩膜（如果有）
                aneurysm_mask = self._load_aneurysm_mask(filename)
                has_aneurysm_mask = aneurysm_mask is not None

                # 计算重叠指标（如果有掩膜）
                pred_overlap_metrics = None
                if has_aneurysm_mask:
                    pred_overlap_metrics = compute_overlap_metrics(attention_mask_pred_np, aneurysm_mask)
                    if pred_overlap_metrics['iou'] > 0.5:
                        stats['iou_above_05'] += 1
                    if pred_overlap_metrics['iou'] > 0.7:
                        stats['iou_above_07'] += 1
                    stats['iou_total'] += 1

                # 更新统计
                stats['total'] += 1
                stats['height_error_sum'] += height_error
                stats['width_error_sum'] += width_error

                if size_correct:
                    stats['size_correct'] += 1
                if height_correct_threshold:
                    stats['height_correct_threshold'] += 1
                if width_correct_threshold:
                    stats['width_correct_threshold'] += 1
                if both_correct_threshold:
                    stats['both_correct_threshold'] += 1

                stats[f'class_{size_class_true}_total'] += 1
                if size_correct:
                    stats[f'class_{size_class_true}_correct'] += 1

                # 保存结果
                result = {
                    'filename': filename,
                    'height_pred': height_pred,
                    'width_pred_scaled': width_pred_scaled,
                    'width_pred_true': width_pred_true,
                    'height_true': height_true,
                    'width_true': width_true,
                    'height_error': height_error,
                    'width_error': width_error,
                    'height_correct_threshold': height_correct_threshold,
                    'width_correct_threshold': width_correct_threshold,
                    'both_correct_threshold': both_correct_threshold,
                    'position_name': position_name,
                    'size_category': size_category,
                    'pred_class': pred_class,
                    'true_class': size_class_true,
                    'pred_class_name': pred_class_name,
                    'true_class_name': true_class_name,
                    'size_correct': size_correct,
                    'has_aneurysm_mask': has_aneurysm_mask,
                    'pred_iou': pred_overlap_metrics['iou'] if has_aneurysm_mask else None,
                    'pred_coverage': pred_overlap_metrics['coverage'] if has_aneurysm_mask else None
                }
                results.append(result)

                # 为每个样本保存对比图
                self._save_output_images(
                    os.path.splitext(filename)[0],
                    image_np,
                    attention_mask_pred_np,
                    attention_mask_gt_np,
                    image_np * attention_mask_pred_np,
                    image_np * attention_mask_gt_np,
                    aneurysm_mask,
                    pred_overlap_metrics,
                    None,
                    height_pred,
                    width_pred_true,
                    height_true,
                    width_true,
                    position_name,
                    size_category,
                    size_correct,
                    pred_class_name,
                    true_class_name
                )

        self._save_test_results(results, stats)
        return results

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
        """Create comparison plot"""
        images_to_plot = [image, pred_mask]

        # 计算是否在阈值内
        height_error = abs(height_pred - height_true) if height_true is not None else 1.0
        width_error = abs(width_pred - width_true) if width_true is not None else 1.0
        height_status = f"✓ <{self.error_threshold}" if height_error < self.error_threshold else f"✗ >{self.error_threshold}"
        width_status = f"✓ <{self.error_threshold}" if width_error < self.error_threshold else f"✗ >{self.error_threshold}"

        # Prediction mask title
        pred_title = f'Predicted Mask\nHeight: {height_pred:.3f} ({height_status})\nWidth: {width_pred:.3f} ({width_status})'
        titles_to_plot = ['Original Image', pred_title]

        # Ground truth mask
        if gt_mask is not None:
            gt_title = f'Ground Truth Mask\nHeight: {height_true:.3f}\nWidth: {width_true:.3f}'
            images_to_plot.append(gt_mask)
            titles_to_plot.append(gt_title)
        else:
            images_to_plot.append(np.zeros_like(image))
            titles_to_plot.append('No Ground Truth')

        # Prediction overlay
        if aneurysm_mask is not None:
            overlay_display = create_overlay_image(image, pred_mask, aneurysm_mask, 'red')
            images_to_plot.append(overlay_display)
            title = 'Prediction Overlay\n'
            if pred_metrics:
                title += f'IoU: {pred_metrics["iou"]:.3f}\n'
                title += f'Coverage: {pred_metrics["coverage"]:.1%}'
            titles_to_plot.append(title)
        else:
            images_to_plot.append(image)
            titles_to_plot.append('No Aneurysm Mask')

        # Ground truth overlay
        if aneurysm_mask is not None and gt_mask is not None:
            gt_overlay_display = create_overlay_image(image, gt_mask, aneurysm_mask, 'blue')
            images_to_plot.append(gt_overlay_display)
            title = 'Ground Truth Overlay\n'
            if gt_metrics:
                title += f'IoU: {gt_metrics["iou"]:.3f}\n'
                title += f'Coverage: {gt_metrics["coverage"]:.1%}'
            titles_to_plot.append(title)
        else:
            if len(images_to_plot) < 5:
                images_to_plot.append(image)
                titles_to_plot.append('No Overlay')

        # Main title
        suptitle = f"{basename}\nAneurysm Type: {position_name}"
        if height_true is not None:
            suptitle += f"\nPred: Height={height_pred:.3f} ({height_status}), Width={width_pred:.3f} ({width_status}) | "
            suptitle += f"GT: Height={height_true:.3f}, Width={width_true:.3f} | "
            suptitle += f"Size: {size_category}"
            suptitle += f"\nClassification: {pred_class_name} vs {true_class_name} - {'✓ Correct' if size_correct else '✗ Wrong'}"
        else:
            suptitle += f"\nPred: Height={height_pred:.3f}, Width={width_pred:.3f}"

        plot_comparison_grid(
            images_to_plot,
            titles_to_plot,
            suptitle,
            self.comparison_dir / f"{basename}_comparison.png",
            figsize=(25, 5)
        )

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

        height_threshold_accuracy = stats['height_correct_threshold'] / total
        width_threshold_accuracy = stats['width_correct_threshold'] / total
        both_threshold_accuracy = stats['both_correct_threshold'] / total

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
            f.write("Test Results Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total test images: {total}\n")
            f.write(f"Error threshold: {self.error_threshold}\n\n")

            f.write("1. Threshold Accuracy\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  Height accuracy: {height_threshold_accuracy:.2%} ({stats['height_correct_threshold']}/{total})\n")
            f.write(f"  Width accuracy: {width_threshold_accuracy:.2%} ({stats['width_correct_threshold']}/{total})\n")
            f.write(f"  Both accuracy: {both_threshold_accuracy:.2%} ({stats['both_correct_threshold']}/{total})\n\n")

            f.write("2. Size Classification Accuracy\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Overall accuracy: {size_accuracy:.2%} ({stats['size_correct']}/{total})\n\n")

            f.write("  Per-class accuracy:\n")
            for class_name in self.class_names:
                acc = class_accuracies[class_name]['accuracy']
                correct = class_accuracies[class_name]['correct']
                total_class = class_accuracies[class_name]['total']
                f.write(f"    {class_name.capitalize():6}: {acc:.2%} ({correct}/{total_class})\n")
            f.write("\n")

            f.write("3. Regression Errors\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Average height error: {avg_height_error:.4f}\n")
            f.write(f"  Average width error: {avg_width_error:.4f}\n\n")

            if stats['iou_total'] > 0:
                f.write("4. Mask Overlap Metrics\n")
                f.write("-" * 40 + "\n")
                f.write(f"  IoU > 0.5: {iou_above_05_rate:.2%} ({stats['iou_above_05']}/{stats['iou_total']})\n")
                f.write(f"  IoU > 0.7: {iou_above_07_rate:.2%} ({stats['iou_above_07']}/{stats['iou_total']})\n")

        # 打印简明统计
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        print(f"Threshold accuracy (error < {self.error_threshold}):")
        print(f"  Height: {height_threshold_accuracy:.2%} ({stats['height_correct_threshold']}/{total})")
        print(f"  Width: {width_threshold_accuracy:.2%} ({stats['width_correct_threshold']}/{total})")
        print(f"  Both: {both_threshold_accuracy:.2%} ({stats['both_correct_threshold']}/{total})")
        print(f"\nSize classification accuracy: {size_accuracy:.2%} ({stats['size_correct']}/{total})")
        print(f"Average height error: {avg_height_error:.4f}")
        print(f"Average width error: {avg_width_error:.4f}")
        if stats['iou_total'] > 0:
            print(f"IoU > 0.5: {iou_above_05_rate:.2%}")
        print(f"\nDetailed report saved to: {self.output_dir / 'test_report.txt'}")

# 13. 主函数（修改版）
def main():
    """主函数 - 支持新模型的训练和测试"""
    print("ImprovedCoarseInfoExtractor 训练与测试程序")
    print("=" * 60)

    # 配置参数
    config = {
        # ========== 数据路径 ==========
        'train_image_dir': "D:/med_data/ai/translate/contrast",
        'val_image_dir': "D:/med_data/ai/translate/test1",
        'test_image_dir': "D:/med_data/ai/translate/test1",
        'location_excel_path': "D:/med_data/ai/translate/location_contrast_size.xlsx",
        'position_excel_path': "D:/med_data/ai/translate/contrast/classify_all_trans_updated.xlsx",

        # ========== 模型参数 ==========
        'image_size': (512, 512),
        'base_channels': 64,  # 基础通道数
        'num_disentangled_features': 6,  # 解耦特征数量（每个8维，共48维）
        'dropout_rate': 0.1,
        'use_soft_label': True,

        # ========== 损失函数权重（推荐值）==========
        # 主任务
        'height_weight': 4.0,           # 增加高度权重
        'width_reg_weight': 0.6,
        'width_cls_weight': 1.0,

        # Focal Loss参数
        'use_focal': True,
        'focal_gamma': 2.0,
        'focal_mix': 0.5,                # Focal和MSE的混合比例

        # 空间位置监督
        'spatial_weight': 0.5,

        # 辅助任务
        'contrast_weight': 0.25,  # 对比学习损失权重
        'orthogonal_weight': 0.15,  # 正交性损失权重
        'aneurysm_weight': 0.25,  # 动脉瘤特征一致性损失权重
        'contrast_temperature': 0.3,  # 对比学习温度参数

        # ========== 训练参数 ==========
        'batch_size': 8,  # 新模型稍大，batch_size适当减小
        'num_epochs': 60,
        'learning_rate': 8e-5,  # 略小的学习率
        'weight_decay': 1e-4,

        # ========== 早停参数 ==========
        'early_stopping_patience': 8,
        'early_stopping_min_delta': 0.0003,

        # ========== 其他参数 ==========
        'num_workers': 2,
        'max_train_samples': None,
        'max_val_samples': None,

        # ========== 保存路径 ==========
        'cache_root': "D:/med_data/ai/stream_cache",
        'model_save_root': "D:/med_data/ai/pre_GCN_loc",
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
    print("\n步骤 1: 训练 ImprovedCoarseInfoExtractor 模型")
    trainer = CoarseExtractorTrainer(config)
    trained_model_path = trainer.train(cache_session)

    # ========== 测试模型（使用缓存数据，确保与验证集一致）==========
    print("\n" + "=" * 60)
    print("步骤 2: 测试训练好的模型（使用缓存数据）")

    if isinstance(trained_model_path, Path):
        trained_model_path = str(trained_model_path)

    # 使用缓存会话进行测试，确保预处理完全一致
    tester = CoarseExtractorTester(
        trained_model_path,
        config,
        cache_session=cache_session  # 传入缓存会话
    )

    # 从缓存读取验证集数据进行测试
    test_results = tester.test_from_cache()

    print("\n" + "=" * 60)
    print("程序完成!")
    print(f"训练输出目录: {trainer.output_dir}")
    print(f"测试输出目录: {tester.output_dir}")

    if test_results:
        print(f"\n测试了 {len(test_results)} 张图像")

        # 计算平均误差
        avg_height_error = sum(r['height_error'] for r in test_results) / len(test_results)
        avg_width_error = sum(r['width_error'] for r in test_results) / len(test_results)
        accuracy = sum(1 for r in test_results if r['size_correct']) / len(test_results)

        print(f"\n平均高度误差: {avg_height_error:.4f}")
        print(f"平均宽度误差: {avg_width_error:.4f}")
        print(f"分类准确率: {accuracy:.2%}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()