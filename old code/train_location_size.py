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
from multi.locate_net_locate_encoder_simple import CoarseInfoExtractor

warnings.filterwarnings('ignore')


# ==============================
# 1. 通用工具函数（提取公共功能）
# ==============================

def setup_output_directory(base_dir: str, prefix: str = "") -> Path:
    """
    创建带日期和随机数的输出目录

    Args:
        base_dir: 基础目录
        prefix: 前缀

    Returns:
        创建的目录路径
    """
    date_str = datetime.now().strftime('%Y%m%d')
    random_num = random.randint(100, 999)
    dir_name = f"{prefix}_{date_str}_{random_num}" if prefix else f"{date_str}_{random_num}"
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_image(file_path: Path, target_size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
    """
    通用图像加载函数

    Args:
        file_path: 图像文件路径
        target_size: 目标尺寸

    Returns:
        归一化后的图像数组
    """
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
    """
    通用图像保存函数

    Args:
        image: 图像数组
        path: 保存路径
        normalize: 是否归一化
    """
    if normalize and image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    cv2.imwrite(str(path), image)


def extract_case_id(filename: str) -> Optional[str]:
    """
    从文件名提取病历号

    Args:
        filename: 文件名

    Returns:
        病历号或None
    """
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
    """
    计算重叠指标

    Args:
        pred_mask: 预测掩膜
        gt_mask: 真实掩膜
        threshold: 二值化阈值

    Returns:
        包含IoU和覆盖率的字典
    """
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
    """
    创建叠加图像

    Args:
        dsa_image: 原始DSA图像
        attention_mask: 注意力掩膜
        aneurysm_mask: 动脉瘤掩膜
        attention_color: 注意力颜色

    Returns:
        叠加后的RGB图像
    """
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
    """
    绘制对比网格图

    Args:
        images: 图像列表
        titles: 标题列表
        suptitle: 总标题
        save_path: 保存路径
        figsize: 图像尺寸
    """
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
    """
    保存训练历史

    Args:
        history: 训练历史字典
        output_dir: 输出目录
        config: 配置参数
    """
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_height_loss': history['train_height_loss'],
        'train_width_loss': history['train_width_loss'],
        'val_height_loss': history['val_height_loss'],
        'val_width_loss': history['val_width_loss'],
        'learning_rate': history['learning_rate']
    })

    history_df.to_csv(output_dir / 'training_history.csv', index=False)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4, default=str)


def plot_training_curves(history: Dict[str, List[float]], save_dir: Path):
    """
    绘制训练曲线

    Args:
        history: 训练历史字典
        save_dir: 保存目录
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

    # 宽度损失
    axes[1, 0].plot(epochs, history['train_width_loss'], 'b-', label='训练宽度损失')
    axes[1, 0].plot(epochs, history['val_width_loss'], 'r-', label='验证宽度损失')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].set_title('宽度比例损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 学习率
    axes[1, 1].plot(epochs, history['learning_rate'], 'g-', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('学习率')
    axes[1, 1].set_title('学习率调度')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


# 2. 强化版混合损失函数
class HybridLoss(nn.Module):
    """
    混合损失函数基类
    """

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MSEL1Loss(HybridLoss):
    """
    MSE + L1 混合损失
    """

    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            weights = {'mse': 0.5, 'l1': 0.5}
        super().__init__(weights)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        return self.weights['mse'] * mse_loss + self.weights['l1'] * l1_loss


class SmoothL1MSELoss(HybridLoss):
    """
    Smooth L1 + MSE 混合损失
    Smooth L1对异常值更鲁棒
    """

    def __init__(self, weights: Dict[str, float] = None, beta: float = 1.0):
        if weights is None:
            weights = {'smooth_l1': 0.6, 'mse': 0.4}
        super().__init__(weights)
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth_l1_loss = self.smooth_l1(pred, target)
        mse_loss = self.mse(pred, target)
        return self.weights['smooth_l1'] * smooth_l1_loss + self.weights['mse'] * mse_loss


class HuberMSELoss(HybridLoss):
    """
    Huber + MSE 混合损失
    Huber结合了MSE和MAE的优点
    """

    def __init__(self, weights: Dict[str, float] = None, delta: float = 1.0):
        if weights is None:
            weights = {'huber': 0.8, 'mse': 0.2}
        super().__init__(weights)
        self.huber = nn.HuberLoss(delta=delta)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(pred, target)
        mse_loss = self.mse(pred, target)
        return self.weights['huber'] * huber_loss + self.weights['mse'] * mse_loss


class LogCoshMSELoss(HybridLoss):
    """
    Log-Cosh + MSE 混合损失
    Log-Cosh近似于Huber但处处二阶可导
    """

    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            weights = {'log_cosh': 0.5, 'mse': 0.5}
        super().__init__(weights)

    def log_cosh_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_cosh_loss = self.log_cosh_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        return self.weights['log_cosh'] * log_cosh_loss + self.weights['mse'] * mse_loss


class AdaptiveWeightedLoss(HybridLoss):
    """
    自适应加权损失（MSE + L1 + 边界约束）
    根据预测误差动态调整权重
    """

    def __init__(self, base_weights: Dict[str, float] = None, alpha: float = 0.1):
        if base_weights is None:
            base_weights = {'mse': 0.5, 'l1': 0.3, 'boundary': 0.2}
        super().__init__(base_weights)
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """边界约束：确保预测值在有效范围内"""
        lower_bound = torch.clamp(0.0 - pred, min=0)
        upper_bound = torch.clamp(pred - 1.0, min=0)
        return torch.mean(lower_bound + upper_bound)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        boundary_loss = self.boundary_loss(pred, target)

        # 根据误差大小动态调整权重
        with torch.no_grad():
            error = torch.abs(pred - target).mean()
            mse_weight = self.weights['mse'] * (1 + self.alpha * error)
            l1_weight = self.weights['l1'] * (1 - self.alpha * error)
            total_weight = mse_weight + l1_weight + self.weights['boundary']
            mse_weight = mse_weight / total_weight
            l1_weight = l1_weight / total_weight
            boundary_weight = self.weights['boundary'] / total_weight

        return mse_weight * mse_loss + l1_weight * l1_loss + boundary_weight * boundary_loss


class MultiTaskLoss(nn.Module):
    """
    多任务损失：高度回归 + 尺寸分类
    兼容现有模型输出格式（直接输出尺寸分类值）
    """

    def __init__(self,
                 height_weight=1.0,  # 高度回归损失权重
                 size_weight=1.0,  # 尺寸分类损失权重
                 reg_loss_type='mse_l1'):  # 回归损失类型
        super().__init__()

        # 高度回归损失
        if reg_loss_type == 'mse_l1':
            self.height_criterion = MSEL1Loss()
        elif reg_loss_type == 'huber_mse':
            self.height_criterion = HuberMSELoss()
        elif reg_loss_type == 'smooth_l1_mse':
            self.height_criterion = SmoothL1MSELoss()
        else:
            self.height_criterion = nn.MSELoss()

        # 尺寸分类损失 - 将连续的预测值转为分类
        self.size_criterion = nn.CrossEntropyLoss()

        self.height_weight = height_weight
        self.size_weight = size_weight

        # 尺寸分类的边界值
        self.size_boundaries = torch.tensor([0.04, 0.08, 0.19, 0.4])

    def _convert_width_to_class(self, width_values):
        """
        将连续的宽度值转换为类别索引
        用于计算分类损失
        """
        batch_size = width_values.shape[0]
        class_indices = torch.zeros(batch_size, dtype=torch.long, device=width_values.device)

        # 根据边界值分类
        for i, val in enumerate(width_values):
            if val < 0.04:
                class_indices[i] = 0  # '0.04' - tiny
            elif val < 0.08:
                class_indices[i] = 1  # '0.08' - small
            elif val < 0.19:
                class_indices[i] = 2  # '0.19' - medium
            else:
                class_indices[i] = 3  # '0.4' - large

        return class_indices

    def _convert_target_to_class(self, target_width):
        """
        将目标宽度值转换为类别索引
        处理从Excel读取的原始值（可能是字符串或数值）
        """
        # 如果目标是字符串（如 '0.04'），直接映射到类别
        if isinstance(target_width, str):
            mapping = {
                '0.04': 0,
                '0.08': 1,
                '0.19': 2,
                '0.4': 3
            }
            return mapping.get(target_width, 2)  # 默认用medium

        # 如果目标是数值，用边界分类
        if target_width < 0.04:
            return 0
        elif target_width < 0.08:
            return 1
        elif target_width < 0.19:
            return 2
        else:
            return 3

    def forward(self, pred_height, pred_width, target_height, target_width):
        """
        Args:
            pred_height: 模型预测的高度 [batch_size]
            pred_width: 模型预测的宽度 [batch_size] (连续值，但代表类别)
            target_height: 真实高度 [batch_size]
            target_width: 真实宽度 [batch_size] (可能是字符串或数值)

        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        # 1. 高度回归损失
        height_loss = self.height_criterion(pred_height, target_height)

        # 2. 尺寸分类损失
        # 将预测的宽度转为类别logits（这里用预测值本身作为类别置信度）
        # 注意：这里假设模型输出的pred_width已经包含了类别信息
        batch_size = pred_width.shape[0]

        # 创建类别logits（4类）
        # 使用预测宽度值构建简单的logits
        size_logits = torch.zeros(batch_size, 4, device=pred_width.device)

        # 根据预测值填充logits
        for i in range(batch_size):
            pred_val = pred_width[i].item()

            if pred_val < 0.04:
                size_logits[i, 0] = 2.0  # 增强对应类别的置信度
                size_logits[i, 1:] = -1.0
            elif pred_val < 0.08:
                size_logits[i, 1] = 2.0
                size_logits[i, [0, 2, 3]] = -1.0
            elif pred_val < 0.19:
                size_logits[i, 2] = 2.0
                size_logits[i, [0, 1, 3]] = -1.0
            else:
                size_logits[i, 3] = 2.0
                size_logits[i, :3] = -1.0

        # 获取目标类别
        target_classes = []
        for i in range(batch_size):
            target_val = target_width[i]
            if isinstance(target_val, torch.Tensor):
                target_val = target_val.item()
            target_class = self._convert_target_to_class(target_val)
            target_classes.append(target_class)

        target_classes = torch.tensor(target_classes, dtype=torch.long, device=pred_width.device)

        # 计算分类损失
        size_loss = self.size_criterion(size_logits, target_classes)

        # 总损失
        total_loss = self.height_weight * height_loss + self.size_weight * size_loss

        return total_loss, {
            'height_loss': height_loss.item(),
            'size_loss': size_loss.item(),
            'total_loss': total_loss.item()
        }


class HardClassificationLoss(nn.Module):
    """
    硬分类损失 - 直接将连续值映射到离散类别
    类别映射规则:
        width < 0.04  → 类别0 (tiny)
        0.04 ≤ width < 0.08 → 类别1 (small)
        0.08 ≤ width < 0.19 → 类别2 (medium)
        width ≥ 0.19 → 类别3 (large)
    """

    def __init__(self,
                 height_weight=1.0,  # 高度回归损失权重
                 width_weight=0.3,  # 宽度回归损失权重（辅助）
                 size_weight=1.0,  # 尺寸分类损失权重
                 boundaries=[0.04, 0.08, 0.19, 0.4]):  # 分类边界
        super().__init__()

        # 回归损失
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # 分类损失
        self.ce_loss = nn.CrossEntropyLoss()

        self.height_weight = height_weight
        self.width_weight = width_weight
        self.size_weight = size_weight

        # 分类边界（硬阈值）
        self.boundaries = boundaries
        self.num_classes = len(boundaries)

        print(f"硬分类损失初始化:")
        print(f"  分类边界: {boundaries}")
        print(f"  类别数: {self.num_classes}")
        print(f"  权重 - 高度: {height_weight}, 宽度: {width_weight}, 尺寸: {size_weight}")

    def convert_value_to_class(self, values):
        """
        将连续值转换为类别索引（硬分类）

        Args:
            values: [batch_size] 连续值张量

        Returns:
            class_indices: [batch_size] 类别索引 (0,1,2,3)
        """
        class_indices = torch.zeros_like(values, dtype=torch.long)

        # 根据边界值进行硬分类
        class_indices[values < self.boundaries[0]] = 0  # tiny
        class_indices[(values >= self.boundaries[0]) & (values < self.boundaries[1])] = 1  # small
        class_indices[(values >= self.boundaries[1]) & (values < self.boundaries[2])] = 2  # medium
        class_indices[values >= self.boundaries[2]] = 3  # large

        return class_indices

    def get_class_logits(self, values):
        """
        将连续值转换为类别logits（用于分类损失）
        使用one-hot形式，预测值属于哪个类别，就给那个类别高置信度

        Args:
            values: [batch_size] 连续值张量

        Returns:
            logits: [batch_size, num_classes] 类别logits
        """
        batch_size = values.shape[0]
        class_indices = self.convert_value_to_class(values)

        # 创建one-hot编码的logits
        logits = torch.zeros(batch_size, self.num_classes, device=values.device)

        # 给正确的类别高置信度，其他类别低置信度
        for i in range(batch_size):
            logits[i, class_indices[i]] = 10.0  # 正类高置信度
            # 其他类别保持为0（负类）

        return logits

    def forward(self, pred_height, pred_width, target_height, target_width):
        """
        前向传播 - 计算多任务损失

        Args:
            pred_height: [batch_size] 预测高度
            pred_width: [batch_size] 预测宽度（连续值）
            target_height: [batch_size] 真实高度
            target_width: [batch_size] 真实宽度（可能是字符串或数值）

        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        batch_size = pred_height.shape[0]

        # 1. 高度回归损失
        height_loss = self.mse_loss(pred_height, target_height)

        # 2. 宽度回归损失（辅助任务）- 保持原有能力
        # 将目标宽度转换为浮点数
        target_width_float = []
        for i in range(batch_size):
            if isinstance(target_width[i], torch.Tensor):
                val = target_width[i].item()
            elif isinstance(target_width[i], str):
                # 如果是字符串（如 '0.04'），转换为浮点数
                val = float(target_width[i])
            else:
                val = float(target_width[i])
            target_width_float.append(val)

        target_width_tensor = torch.tensor(target_width_float, device=pred_width.device)
        width_reg_loss = self.l1_loss(pred_width, target_width_tensor)

        # 3. 尺寸分类损失（硬分类）
        # 将预测宽度转为类别logits
        pred_class_logits = self.get_class_logits(pred_width)  # [batch, 4]

        # 获取目标类别
        target_classes = []
        for i in range(batch_size):
            target_val = target_width[i]
            if isinstance(target_val, torch.Tensor):
                target_val = target_val.item()

            if isinstance(target_val, str):
                # 如果是字符串（如 '0.04'），直接映射
                mapping = {'0.04': 0, '0.08': 1, '0.19': 2, '0.4': 3}
                target_class = mapping.get(target_val, 2)  # 默认medium
            else:
                # 如果是数值，用边界分类
                target_class = self.convert_value_to_class(torch.tensor([target_val]))[0].item()

            target_classes.append(target_class)

        target_classes = torch.tensor(target_classes, dtype=torch.long, device=pred_width.device)  # [batch]

        # 计算分类损失
        size_loss = self.ce_loss(pred_class_logits, target_classes)

        # 总损失
        total_loss = (self.height_weight * height_loss +
                      self.width_weight * width_reg_loss +
                      self.size_weight * size_loss)

        # 计算分类准确率（用于监控）
        pred_classes = self.convert_value_to_class(pred_width)
        accuracy = (pred_classes == target_classes).float().mean().item()

        return total_loss, {
            'height_loss': height_loss.item(),
            'width_reg_loss': width_reg_loss.item(),
            'size_loss': size_loss.item(),
            'total_loss': total_loss.item(),
            'classification_accuracy': accuracy
        }


# ==============================
# 简化的多任务损失（直接使用现有模型输出）
# ==============================

class SimpleMultiTaskLoss(nn.Module):
    """
    简化的多任务损失 - 直接使用现有模型输出
    不改变模型结构，只改变损失计算方式
    """

    def __init__(self,
                 height_weight=1.0,
                 width_weight=0.5,  # 宽度回归损失权重（辅助任务）
                 size_weight=1.0):  # 尺寸分类损失权重
        super().__init__()

        # 回归损失（用于高度和宽度）
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # 分类损失
        self.ce_loss = nn.CrossEntropyLoss()

        self.height_weight = height_weight
        self.width_weight = width_weight
        self.size_weight = size_weight

        # 尺寸分类边界
        self.size_classes = ['0.04', '0.08', '0.19', '0.4']
        self.size_boundaries = [0.04, 0.08, 0.19, 0.4]

    def width_to_class_logits(self, width_pred):
        """
        将宽度预测值转换为类别logits
        使用高斯分布建模每个类别的概率
        """
        batch_size = width_pred.shape[0]
        logits = torch.zeros(batch_size, 4, device=width_pred.device)

        # 为每个类别计算概率（基于距离）
        centers = torch.tensor([0.02, 0.06, 0.135, 0.295], device=width_pred.device)
        sigma = 0.03  # 标准差

        for i, center in enumerate(centers):
            # 计算高斯概率
            probs = torch.exp(-(width_pred - center) ** 2 / (2 * sigma ** 2))
            logits[:, i] = probs

        return logits

    def get_target_class(self, target_width):
        """获取目标类别"""
        if isinstance(target_width, torch.Tensor):
            target_width = target_width.item()

        if isinstance(target_width, str):
            # 如果是字符串，直接映射
            mapping = {cls: i for i, cls in enumerate(self.size_classes)}
            return mapping.get(target_width, 2)
        else:
            # 如果是数值，根据边界分类
            for i, boundary in enumerate(self.size_boundaries):
                if target_width < boundary:
                    return i
            return 3

    def forward(self, pred_height, pred_width, target_height, target_width):
        """
        前向传播
        """
        batch_size = pred_height.shape[0]

        # 1. 高度回归损失（主要任务）
        height_loss = self.mse_loss(pred_height, target_height)

        # 2. 宽度回归损失（辅助任务）- 保持原有能力
        width_reg_loss = self.l1_loss(pred_width, target_width.float())

        # 3. 尺寸分类损失（主要任务）
        # 将预测宽度转为类别logits
        size_logits = self.width_to_class_logits(pred_width)

        # 获取目标类别
        target_classes = []
        for i in range(batch_size):
            target_val = target_width[i]
            target_class = self.get_target_class(target_val)
            target_classes.append(target_class)

        target_classes = torch.tensor(target_classes, dtype=torch.long, device=pred_width.device)

        # 分类损失
        size_loss = self.ce_loss(size_logits, target_classes)

        # 总损失
        total_loss = (self.height_weight * height_loss +
                      self.width_weight * width_reg_loss +
                      self.size_weight * size_loss)

        return total_loss, {
            'height_loss': height_loss.item(),
            'width_reg_loss': width_reg_loss.item(),
            'size_loss': size_loss.item(),
            'total_loss': total_loss.item()
        }


# 3. 改进的SizeRange信息加载器
class SizeRangeInfoLoader:
    """加载尺寸分类信息（从第五列读取）"""

    SIZE_CATEGORIES = {
        '0.04': 'tiny',  # 小于0.04
        '0.08': 'small',  # 0.04-0.08
        '0.19': 'medium',  # 0.08-0.19
        '0.4': 'large'  # 大于0.19
    }

    SIZE_RANGES = {
        'tiny': (0, 0.04),
        'small': (0.04, 0.08),
        'medium': (0.08, 0.19),
        'large': (0.19, 1.0)
    }

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}  # 存储文件名到位置信息的映射
        self.case_to_location = {}  # 存储病历号到位置信息的映射
        self.size_distribution = {category: 0 for category in self.SIZE_CATEGORIES.values()}
        self._load_location_info()

    def _get_size_category(self, width_ratio: float) -> str:
        """根据宽度比例获取尺寸分类"""
        if width_ratio < 0.04:
            return 'tiny'
        elif width_ratio < 0.08:
            return 'small'
        elif width_ratio < 0.19:
            return 'medium'
        else:
            return 'large'

    def _parse_size_range(self, size_value: str) -> float:
        """
        解析第五列的尺寸范围值
        返回实际的宽度比例值（使用范围的中值或边界值）
        """
        try:
            # 如果直接是数值，可能是分类标识
            if size_value in self.SIZE_CATEGORIES:
                category = self.SIZE_CATEGORIES[size_value]
                range_min, range_max = self.SIZE_RANGES[category]
                # 返回范围的中值作为代表值
                return (range_min + range_max) / 2
            else:
                # 尝试直接转换为浮点数
                return float(size_value)
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

                # 获取尺寸分类（第五列）
                size_value = str(row.iloc[4]).strip() if len(row) > 4 else '0.19'
                width_ratio = self._parse_size_range(size_value)

                # 确保宽度比例在0-1范围内
                width_ratio = max(0.0, min(1.0, width_ratio))

                # 获取尺寸分类名称
                size_category = self._get_size_category(width_ratio)
                self.size_distribution[size_category] += 1

                # 去除可能的扩展名
                basename = os.path.splitext(filename)[0]

                location_info = {
                    'height_ratio': height_ratio,
                    'width_ratio': width_ratio,
                    'size_category': size_category,
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
                    range_min, range_max = self.SIZE_RANGES[category]
                    print(f"  {category}: {count}个样本 (范围: {range_min}-{range_max})")

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
            'width_ratio': 0.1,
            'size_category': 'medium',
            'original_size_value': '0.19'
        }


# 4. Early Stopping (保持不变)
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

# 5. AttentionMaskGenerator (保持不变)
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
            original_w_ratio = w_ratio
            if w_ratio < self.min_radius_ratio:
                print(
                    f"警告: 样本 {b} 的宽度比例 {w_ratio:.4f} 小于最小值 {self.min_radius_ratio}，将使用 {self.min_radius_ratio}")
                w_ratio = self.min_radius_ratio

            # Calculate center height position
            y_center = int(h_ratio * (H - 1))
            y_center = max(0, min(y_center, H - 1))

            # Calculate window width
            window_height = int(w_ratio * H)
            if window_height < 1:
                window_height = 1
            elif window_height > H:
                window_height = H

            # 记录原始和调整后的窗口高度
            original_window_height = int(original_w_ratio * H)
            if original_window_height < 1:
                original_window_height = 1
            elif original_window_height > H:
                original_window_height = H

            if original_window_height != window_height:
                print(
                    f"样本 {b}: 窗口高度从 {original_window_height} 调整为 {window_height} (原始比例: {original_w_ratio:.4f}, 调整后比例: {w_ratio:.4f})")

            # Calculate rectangle boundaries
            half_height = window_height // 2
            y_min = max(0, y_center - half_height)
            y_max = min(H - 1, y_center + half_height)

            # If window width is odd, adjust boundaries
            if window_height % 2 == 1:
                if y_min > 0:
                    y_min -= 1
                elif y_max < H - 1:
                    y_max += 1

            # Create rectangular mask
            attention_mask = torch.zeros(H, W, device=self.device)
            attention_mask[y_min:y_max + 1, :] = 1.0

            batch_masks.append(attention_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(batch_masks, dim=0)


# 6. PositionInfoLoader (保持不变)
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

# 7. StreamDataManager (保持不变)
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
            elif isinstance(value, (int, float, str)):
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

# 8. StreamCoarseExtractorDataset (保持不变)
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
        width_ratio = torch.tensor(data['width_ratio'].item(), dtype=torch.float32)

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
            'width_ratio': width_ratio,
            'filename': filename,
            'position_name': position_name,
            'position_num': position_num,
            'size_category': size_category
        }


# 9. DataPreprocessor (更新版)
class DataPreprocessor:
    """数据预处理器 - 负责将原始数据转换为流式存储格式"""

    def __init__(self,
                 image_dir: str,
                 location_excel_path: str,
                 position_excel_path: str = None,
                 image_size: Tuple[int, int] = (512, 512)):

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # 加载位置信息（使用更新后的SizeRangeInfoLoader）
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

        Args:
            output_session: 输出会话路径，如果为None则自动创建
            train_ratio: 训练集比例
            max_samples: 最大样本数
            force_new: 是否强制新建（如果存在则覆盖）

        Returns:
            会话路径
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

                # 获取位置信息（使用新的SizeRangeInfoLoader）
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
                    'width_ratio': np.array([location_info['width_ratio']]),
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

# 10. 训练器（简化版）
class CoarseExtractorTrainer:
    """CoarseInfoExtractor训练器"""

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
            'train_width_loss': [],
            'val_height_loss': [],
            'val_width_loss': [],
            'learning_rate': []
        }

        # 选择损失函数
        #self.criterion = self._create_loss_function()
        self.criterion = self._create_hard_classification_loss()

    def _create_hard_classification_loss(self):
        """创建硬分类损失函数"""
        loss_type = self.config.get('loss_type', 'hard_classification')

        print(f"正在创建损失函数: {loss_type}")

        if loss_type == 'hard_classification':
            print("使用 HardClassificationLoss (硬分类)")
            return HardClassificationLoss(
                height_weight=self.config.get('height_weight', 1.0),
                width_weight=self.config.get('width_weight', 0.3),
                size_weight=self.config.get('size_weight', 1.0),
                boundaries=self.config.get('class_boundaries', [0.04, 0.08, 0.19, 0.4])
            )
        else:
            # 兼容其他损失类型
            return self._create_multi_task_loss()

    def _create_multi_task_loss(self):
        """创建多任务损失函数"""
        loss_type = self.config.get('loss_type', 'multi_task_simple')

        print(f"正在创建多任务损失函数: {loss_type}")

        if loss_type == 'multi_task_simple':
            print("使用 SimpleMultiTaskLoss")
            return SimpleMultiTaskLoss(
                height_weight=self.config.get('height_weight', 1.0),
                width_weight=self.config.get('width_weight', 0.3),
                size_weight=self.config.get('size_weight', 1.0)
            )
        elif loss_type == 'multi_task':
            print("使用 MultiTaskLoss")
            return MultiTaskLoss(
                height_weight=self.config.get('height_weight', 1.0),
                size_weight=self.config.get('size_weight', 1.0),
                reg_loss_type=self.config.get('reg_loss_type', 'mse_l1')
            )
        else:
            print(f"未知损失类型 '{loss_type}'，使用默认MSE损失")
            return nn.MSELoss()

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        loss_type = self.config.get('loss_type', 'mse_l1')

        if loss_type == 'mse_l1':
            weights = self.config.get('loss_weights', {'mse': 0.7, 'l1': 0.3})
            return MSEL1Loss(weights)
        elif loss_type == 'smooth_l1_mse':
            weights = self.config.get('loss_weights', {'smooth_l1': 0.6, 'mse': 0.4})
            beta = self.config.get('smooth_l1_beta', 1.0)
            return SmoothL1MSELoss(weights, beta)
        elif loss_type == 'huber_mse':
            weights = self.config.get('loss_weights', {'huber': 0.8, 'mse': 0.2})
            delta = self.config.get('huber_delta', 1.0)
            return HuberMSELoss(weights, delta)
        elif loss_type == 'log_cosh_mse':
            weights = self.config.get('loss_weights', {'log_cosh': 0.5, 'mse': 0.5})
            return LogCoshMSELoss(weights)
        elif loss_type == 'adaptive':
            weights = self.config.get('loss_weights', {'mse': 0.5, 'l1': 0.3, 'boundary': 0.2})
            alpha = self.config.get('adaptive_alpha', 0.1)
            return AdaptiveWeightedLoss(weights, alpha)
        else:
            print(f"未知损失类型 '{loss_type}'，使用默认MSE损失")
            return nn.MSELoss()

    def _create_model(self):
        """创建模型（6类）"""
        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=6,  # 固定为6类
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
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
        """训练一个epoch（支持硬分类损失）"""
        model.train()
        train_loss_total = 0
        train_height_loss = 0
        train_width_loss = 0
        train_size_loss = 0
        train_accuracy = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [训练]')
        for batch in train_bar:
            images = batch['image'].to(self.device)
            positions = batch['position'].to(self.device)
            height_targets = batch['height_ratio'].to(self.device)
            width_targets = batch['width_ratio']  # 可能是字符串或数值，先不移动到GPU

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, positions)
            height_pred = outputs['height_ratio']
            width_pred = outputs['width_ratio']

            # 根据损失函数类型调用
            if isinstance(criterion, HardClassificationLoss):
                # 硬分类损失需要完整的参数
                loss, loss_dict = criterion(
                    height_pred, width_pred,
                    height_targets, width_targets
                )
                train_height_loss += loss_dict['height_loss']
                train_size_loss += loss_dict['size_loss']
                train_width_loss += loss_dict['width_reg_loss']
                train_accuracy += loss_dict['classification_accuracy']
            elif isinstance(criterion, (MultiTaskLoss, SimpleMultiTaskLoss)):
                # 其他多任务损失
                loss, loss_dict = criterion(
                    height_pred, width_pred,
                    height_targets, width_targets
                )
                train_height_loss += loss_dict['height_loss']
                train_size_loss += loss_dict.get('size_loss', 0)
                train_width_loss += loss_dict.get('width_reg_loss', 0)
            else:
                # 兼容原有损失函数
                height_loss = criterion(height_pred, height_targets)
                width_loss = criterion(width_pred, height_targets)
                loss = height_loss + width_loss
                train_height_loss += height_loss.item()
                train_width_loss += width_loss.item()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录总损失
            train_loss_total += loss.item()

            # 更新进度条
            postfix = {'loss': f'{loss.item():.4f}'}
            if isinstance(criterion, HardClassificationLoss):
                postfix['acc'] = f'{loss_dict["classification_accuracy"]:.2%}'
            train_bar.set_postfix(postfix)

        # 计算平均损失和准确率
        num_batches = len(train_loader)
        avg_metrics = {
            'total': train_loss_total / num_batches,
            'height': train_height_loss / num_batches,
            'width': train_width_loss / num_batches,
            'size': train_size_loss / num_batches,
            'accuracy': train_accuracy / num_batches if train_accuracy > 0 else 0
        }

        return avg_metrics['total'], avg_metrics['height'], avg_metrics['width'], avg_metrics['size'], avg_metrics[
            'accuracy']

    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch（支持多任务损失）"""
        model.eval()
        val_loss_total = 0
        val_height_loss = 0
        val_width_loss = 0
        val_size_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='[验证]')
            for batch in val_bar:
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)
                height_targets = batch['height_ratio'].to(self.device)
                width_targets = batch['width_ratio']  # 可能是字符串，所以先不移动到GPU

                # 前向传播
                outputs = model(images, positions)
                height_pred = outputs['height_ratio']
                width_pred = outputs['width_ratio']

                # 根据损失函数类型调用
                if isinstance(criterion, HardClassificationLoss):
                    # 硬分类损失需要完整的参数
                    loss, loss_dict = criterion(
                        height_pred, width_pred,
                        height_targets, width_targets  # width_targets可能是字符串，HardClassificationLoss内部会处理
                    )
                    val_height_loss += loss_dict['height_loss']
                    val_size_loss += loss_dict['size_loss']
                    val_width_loss += loss_dict['width_reg_loss']
                    val_accuracy += loss_dict.get('classification_accuracy', 0)
                elif isinstance(criterion, (MultiTaskLoss, SimpleMultiTaskLoss)):
                    # 其他多任务损失
                    loss, loss_dict = criterion(
                        height_pred, width_pred,
                        height_targets, width_targets
                    )
                    val_height_loss += loss_dict['height_loss']
                    val_size_loss += loss_dict.get('size_loss', 0)
                    val_width_loss += loss_dict.get('width_reg_loss', 0)
                else:
                    # 兼容原有损失函数（单任务）
                    height_loss = criterion(height_pred, height_targets)
                    width_loss = criterion(width_pred, height_targets)  # 注意：这里可能需要调整
                    loss = height_loss + width_loss
                    val_height_loss += height_loss.item()
                    val_width_loss += width_loss.item()

                val_loss_total += loss.item()

                # 更新进度条
                postfix = {'loss': f'{loss.item():.4f}'}
                if isinstance(criterion, HardClassificationLoss) and 'classification_accuracy' in loss_dict:
                    postfix['acc'] = f'{loss_dict["classification_accuracy"]:.2%}'
                val_bar.set_postfix(postfix)

        # 计算平均损失
        num_batches = len(val_loader)
        avg_metrics = {
            'total': val_loss_total / num_batches,
            'height': val_height_loss / num_batches,
            'width': val_width_loss / num_batches,
            'size': val_size_loss / num_batches,
            'accuracy': val_accuracy / num_batches if val_accuracy > 0 else 0
        }

        return avg_metrics['total'], avg_metrics['height'], avg_metrics['width'], avg_metrics['size'], avg_metrics[
            'accuracy']

    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
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
        """训练模型（完整训练循环）"""
        print("\n" + "=" * 60)
        print(f"开始 CoarseInfoExtractor 训练 (多任务学习)")
        print("=" * 60)

        # 创建模型
        model = self._create_model()

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
            train_loss_avg, train_height_avg, train_width_avg, train_size_avg, train_acc = self.train_epoch(
                model, train_loader, optimizer, self.criterion, epoch
            )

            # 验证 - 接收5个返回值
            val_loss_avg, val_height_avg, val_width_avg, val_size_avg, val_acc = self.validate_epoch(
                model, val_loader, self.criterion
            )

            # 更新学习率
            scheduler.step(val_loss_avg)

            # 记录历史
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss_avg)
            self.history['train_height_loss'].append(train_height_avg)
            self.history['train_width_loss'].append(train_width_avg)
            self.history['val_height_loss'].append(val_height_avg)
            self.history['val_width_loss'].append(val_width_avg)

            # 新增尺寸损失和准确率记录
            if 'train_size_loss' not in self.history:
                self.history['train_size_loss'] = []
                self.history['val_size_loss'] = []
                self.history['train_accuracy'] = []
                self.history['val_accuracy'] = []
            self.history['train_size_loss'].append(train_size_avg)
            self.history['val_size_loss'].append(val_size_avg)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)

            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 打印统计信息
            print(f"\nEpoch {epoch + 1} 统计:")
            print(f"  训练总损失: {train_loss_avg:.4f}")
            print(f"    高度损失: {train_height_avg:.4f}")
            print(f"    尺寸损失: {train_size_avg:.4f}")
            print(f"    训练准确率: {train_acc:.2%}")
            print(f"  验证总损失: {val_loss_avg:.4f}")
            print(f"    高度损失: {val_height_avg:.4f}")
            print(f"    尺寸损失: {val_size_avg:.4f}")
            print(f"    验证准确率: {val_acc:.2%}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_path = self.save_checkpoint(
                    model, optimizer, epoch, train_loss_avg, val_loss_avg, is_best=True
                )
                print(f"  ✓ 保存最佳模型到: {best_model_path}")

            # 每5个epoch保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.save_checkpoint(
                    model, optimizer, epoch, train_loss_avg, val_loss_avg, is_best=False
                )
                print(f"  ✓ 保存检查点到: {checkpoint_path}")

            # 早停检查
            if early_stopping(val_loss_avg):
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
        self._plot_training_history()

        # 保存训练历史
        save_training_history(self.history, self.output_dir, self.config)

        return final_model_path

    def _plot_training_history(self):
        """绘制训练历史曲线（增加准确率）- 英文版"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Total Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Height Loss
        axes[0, 1].plot(epochs, self.history['train_height_loss'], 'b-', label='Train Height Loss')
        axes[0, 1].plot(epochs, self.history['val_height_loss'], 'r-', label='Validation Height Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Height Ratio Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Size Loss
        axes[0, 2].plot(epochs, self.history.get('train_size_loss', []), 'b-', label='Train Size Loss')
        axes[0, 2].plot(epochs, self.history.get('val_size_loss', []), 'r-', label='Validation Size Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Size Classification Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Width Loss (Auxiliary Task)
        axes[1, 0].plot(epochs, self.history['train_width_loss'], 'b-', label='Train Width Loss')
        axes[1, 0].plot(epochs, self.history['val_width_loss'], 'r-', label='Validation Width Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Width Regression Loss (Auxiliary)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[1, 1].plot(epochs, self.history.get('train_accuracy', []), 'b-', label='Train Accuracy')
        axes[1, 1].plot(epochs, self.history.get('val_accuracy', []), 'r-', label='Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Size Classification Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 2].plot(epochs, self.history['learning_rate'], 'g-', marker='o')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

# 11. 测试器（简化版）
class CoarseExtractorTester:
    """CoarseInfoExtractor测试器 - 包含详细准确率统计"""

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
        self.class_boundaries = [0.04, 0.08, 0.19, 0.4]
        self.class_names = ['tiny', 'small', 'medium', 'large']
        self.num_classes = len(self.class_names)

        # 创建输出目录
        self.output_dir = setup_output_directory(
            "D:/med_data/ai/test_results",
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
        print(f"分类边界: {self.class_boundaries}")
        print(f"类别名称: {self.class_names}")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=6,
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss')

        print(f"模型加载成功，训练epochs: {epoch}")
        if isinstance(val_loss, (int, float)):
            print(f"验证损失: {val_loss:.4f}")
        else:
            print(f"验证损失: Unknown")

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

    def _convert_width_to_class(self, width_value):
        """
        将宽度值转换为类别索引（硬分类）

        Args:
            width_value: 浮点数，宽度比例

        Returns:
            class_index: 0=tiny, 1=small, 2=medium, 3=large
        """
        if width_value < self.class_boundaries[0]:
            return 0  # tiny
        elif width_value < self.class_boundaries[1]:
            return 1  # small
        elif width_value < self.class_boundaries[2]:
            return 2  # medium
        else:
            return 3  # large

    def _get_class_name(self, class_index):
        """获取类别名称"""
        return self.class_names[class_index] if 0 <= class_index < len(self.class_names) else 'unknown'

    def test_single_image(self, image_path):
        """测试单张图像 - 包含详细准确率统计"""
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
            width_pred = outputs['width_ratio'].item()

        print(f"  预测 - 高度: {height_pred:.4f}, 宽度: {width_pred:.4f}")

        # 获取真实值
        true_info = self.location_loader.get_location_for_image(filename)
        height_true = true_info['height_ratio']
        width_true = true_info['width_ratio']
        size_category = true_info['size_category']

        print(f"  真实值 - 高度: {height_true:.4f}, 宽度: {width_true:.4f}, 尺寸分类: {size_category}")
        print(f"  误差 - 高度: {abs(height_pred - height_true):.4f}, 宽度: {abs(width_pred - width_true):.4f}")

        # ========== 准确率计算 ==========
        # 1. 尺寸分类准确率
        pred_class = self._convert_width_to_class(width_pred)
        true_class = self._convert_width_to_class(width_true)
        size_correct = (pred_class == true_class)

        pred_class_name = self._get_class_name(pred_class)
        true_class_name = self._get_class_name(true_class)

        print(f"  尺寸分类 - 预测: {pred_class_name} ({pred_class}), 真实: {true_class_name} ({true_class})")
        print(f"  分类正确: {'✓' if size_correct else '✗'}")

        # 2. 高度误差阈值准确率
        height_error = abs(height_pred - height_true)
        height_error_thresholds = [0.02, 0.05, 0.1]
        height_accuracy = {}
        for thresh in height_error_thresholds:
            height_accuracy[f'height_acc_{thresh:.2f}'.replace('.', '_')] = (height_error <= thresh)

        # 3. 宽度误差阈值准确率
        width_error = abs(width_pred - width_true)
        width_accuracy = {}
        for thresh in height_error_thresholds:
            width_accuracy[f'width_acc_{thresh:.2f}'.replace('.', '_')] = (width_error <= thresh)

        # 生成注意力掩膜
        height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
        width_tensor_pred = torch.tensor([width_pred], dtype=torch.float32).to(self.device)

        attention_mask_pred = self.mask_generator(height_tensor_pred, width_tensor_pred)
        attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

        focused_image_pred = image * attention_mask_pred_np

        # 生成标准注意力掩膜（如果有真实值）
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

        # 保存当前图像的信息用于对比图（修复：在这里设置实例变量）
        self._current_height_pred = height_pred
        self._current_width_pred = width_pred
        self._current_height_true = height_true
        self._current_width_true = width_true
        self._current_position_name = position_name
        self._current_size_category = size_category
        self._current_size_correct = size_correct
        self._current_pred_class_name = pred_class_name
        self._current_true_class_name = true_class_name

        # 保存图像
        self._save_output_images(basename, image, attention_mask_pred_np, attention_mask_gt_np,
                                 focused_image_pred, focused_image_gt, aneurysm_mask,
                                 pred_overlap_metrics, gt_overlap_metrics,
                                 height_pred, width_pred, height_true, width_true,
                                 position_name, size_category, size_correct,
                                 pred_class_name, true_class_name)

        # 准备结果字典
        result = {
            'filename': filename,
            'height_pred': height_pred,
            'width_pred': width_pred,
            'height_true': height_true,
            'width_true': width_true,
            'height_error': height_error,
            'width_error': width_error,
            'position_name': position_name,
            'position_num': position_num,
            'size_category': size_category,

            # 准确率相关字段
            'pred_class': pred_class,
            'true_class': true_class,
            'pred_class_name': pred_class_name,
            'true_class_name': true_class_name,
            'size_correct': size_correct,

            # 不同阈值下的准确率
            'height_acc_0_02': height_accuracy['height_acc_0_02'],
            'height_acc_0_05': height_accuracy['height_acc_0_05'],
            'height_acc_0_10': height_accuracy['height_acc_0_10'],
            'width_acc_0_02': width_accuracy['width_acc_0_02'],
            'width_acc_0_05': width_accuracy['width_acc_0_05'],
            'width_acc_0_10': width_accuracy['width_acc_0_10'],

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

        # 创建对比图（直接传递参数，不使用实例变量）
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
        """创建对比图（使用传递的参数，不依赖实例变量）"""
        images_to_plot = [image, pred_mask]

        # 预测掩膜标题
        pred_title = f'Predicted Mask\nHeight: {height_pred:.3f}\nWidth: {width_pred:.3f}'
        if hasattr(self, '_min_radius_ratio') and width_pred < self.mask_generator.min_radius_ratio:
            pred_title += f'\n(Adjusted to min: {self.mask_generator.min_radius_ratio:.3f})'
        titles_to_plot = ['Original Image', pred_title]

        # 标准掩膜
        if gt_mask is not None:
            gt_title = f'Ground Truth Mask\nHeight: {height_true:.3f}\nWidth: {width_true:.3f}'
            if hasattr(self, '_min_radius_ratio') and width_true < self.mask_generator.min_radius_ratio:
                gt_title += f'\n(Adjusted to min: {self.mask_generator.min_radius_ratio:.3f})'
            images_to_plot.append(gt_mask)
            titles_to_plot.append(gt_title)
        else:
            images_to_plot.append(np.zeros_like(image))
            titles_to_plot.append('No Ground Truth')

        # 预测叠加
        if aneurysm_mask is not None:
            overlay_display = create_overlay_image(image, pred_mask, aneurysm_mask, 'red')
            images_to_plot.append(overlay_display)
            title = 'Predicted Overlay\n'
            if pred_metrics:
                title += f'IoU: {pred_metrics["iou"]:.3f}\n'
                title += f'Coverage: {pred_metrics["coverage"]:.1%}'
            titles_to_plot.append(title)
        else:
            images_to_plot.append(image)
            titles_to_plot.append('No Aneurysm Mask')

        # 标准叠加
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

        # 总标题
        suptitle = f"{basename}\nAneurysm Type: {position_name}"
        if height_true is not None:
            suptitle += f"\nPred: Height={height_pred:.3f}, Width={width_pred:.3f} | "
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

    def test_all_images(self, test_image_dir):
        """测试所有图像 - 包含详细准确率统计"""
        test_dir = Path(test_image_dir)

        image_files = []
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']:
                    image_files.append(file_path)

        print(f"\n开始测试 {len(image_files)} 张图像...")
        print("=" * 60)

        results = []

        # 初始化统计计数器
        stats = {
            'total': 0,
            'size_correct': 0,

            'height_correct_002': 0,
            'height_correct_005': 0,
            'height_correct_010': 0,

            'width_correct_002': 0,
            'width_correct_005': 0,
            'width_correct_010': 0,

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

                # 按类别统计
                true_class = result['true_class']
                stats[f'class_{true_class}_total'] += 1
                if result['size_correct']:
                    stats[f'class_{true_class}_correct'] += 1

                # 高度误差阈值统计
                if result['height_acc_0_02']:
                    stats['height_correct_002'] += 1
                if result['height_acc_0_05']:
                    stats['height_correct_005'] += 1
                if result['height_acc_0_10']:
                    stats['height_correct_010'] += 1

                # 宽度误差阈值统计
                if result['width_acc_0_02']:
                    stats['width_correct_002'] += 1
                if result['width_acc_0_05']:
                    stats['width_correct_005'] += 1
                if result['width_acc_0_10']:
                    stats['width_correct_010'] += 1

                # IoU统计
                if result.get('pred_iou') is not None:
                    stats['iou_total'] += 1
                    if result['pred_iou'] > 0.5:
                        stats['iou_above_05'] += 1
                    if result['pred_iou'] > 0.7:
                        stats['iou_above_07'] += 1

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(image_files)} 张图像")

        if results:
            self._save_test_results_with_accuracy(results, stats)
            self._plot_confusion_matrix(results)

        return results

    def _save_test_results_with_accuracy(self, results, stats):
        """保存测试结果（包含准确率统计）"""
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        # 计算各种准确率
        total = stats['total']

        # 尺寸分类准确率
        size_accuracy = stats['size_correct'] / total if total > 0 else 0

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

        # 高度误差阈值准确率
        height_acc_002 = stats['height_correct_002'] / total if total > 0 else 0
        height_acc_005 = stats['height_correct_005'] / total if total > 0 else 0
        height_acc_010 = stats['height_correct_010'] / total if total > 0 else 0

        # 宽度误差阈值准确率
        width_acc_002 = stats['width_correct_002'] / total if total > 0 else 0
        width_acc_005 = stats['width_correct_005'] / total if total > 0 else 0
        width_acc_010 = stats['width_correct_010'] / total if total > 0 else 0

        # IoU统计
        iou_above_05_rate = stats['iou_above_05'] / stats['iou_total'] if stats['iou_total'] > 0 else 0
        iou_above_07_rate = stats['iou_above_07'] / stats['iou_total'] if stats['iou_total'] > 0 else 0

        # 创建详细统计报告
        accuracy_stats = {
            'total_images': total,

            # 尺寸分类准确率
            'size_classification_accuracy': size_accuracy,
            'size_correct_count': stats['size_correct'],

            # 各类别准确率
            'tiny_accuracy': class_accuracies['tiny']['accuracy'],
            'tiny_count': class_accuracies['tiny']['total'],
            'tiny_correct': class_accuracies['tiny']['correct'],

            'small_accuracy': class_accuracies['small']['accuracy'],
            'small_count': class_accuracies['small']['total'],
            'small_correct': class_accuracies['small']['correct'],

            'medium_accuracy': class_accuracies['medium']['accuracy'],
            'medium_count': class_accuracies['medium']['total'],
            'medium_correct': class_accuracies['medium']['correct'],

            'large_accuracy': class_accuracies['large']['accuracy'],
            'large_count': class_accuracies['large']['total'],
            'large_correct': class_accuracies['large']['correct'],

            # 高度误差阈值准确率
            'height_accuracy_0.02': height_acc_002,
            'height_correct_0.02': stats['height_correct_002'],
            'height_accuracy_0.05': height_acc_005,
            'height_correct_0.05': stats['height_correct_005'],
            'height_accuracy_0.10': height_acc_010,
            'height_correct_0.10': stats['height_correct_010'],

            # 宽度误差阈值准确率
            'width_accuracy_0.02': width_acc_002,
            'width_correct_0.02': stats['width_correct_002'],
            'width_accuracy_0.05': width_acc_005,
            'width_correct_0.05': stats['width_correct_005'],
            'width_accuracy_0.10': width_acc_010,
            'width_correct_0.10': stats['width_correct_010'],

            # IoU统计
            'iou_above_0.5_rate': iou_above_05_rate,
            'iou_above_0.5_count': stats['iou_above_05'],
            'iou_above_0.7_rate': iou_above_07_rate,
            'iou_above_0.7_count': stats['iou_above_07'],
            'iou_total_samples': stats['iou_total'],
        }

        # 保存统计报告
        stats_df = pd.DataFrame([accuracy_stats])
        stats_df.to_csv(self.output_dir / 'test_accuracy_statistics.csv', index=False)

        # 生成详细的文本报告
        with open(self.output_dir / 'test_accuracy_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("测试结果准确率报告\n")
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

            f.write("2. 高度比例误差准确率\n")
            f.write("-" * 40 + "\n")
            f.write(f"  误差 ≤ 0.02: {height_acc_002:.2%} ({stats['height_correct_002']}/{total})\n")
            f.write(f"  误差 ≤ 0.05: {height_acc_005:.2%} ({stats['height_correct_005']}/{total})\n")
            f.write(f"  误差 ≤ 0.10: {height_acc_010:.2%} ({stats['height_correct_010']}/{total})\n\n")

            f.write("3. 宽度比例误差准确率\n")
            f.write("-" * 40 + "\n")
            f.write(f"  误差 ≤ 0.02: {width_acc_002:.2%} ({stats['width_correct_002']}/{total})\n")
            f.write(f"  误差 ≤ 0.05: {width_acc_005:.2%} ({stats['width_correct_005']}/{total})\n")
            f.write(f"  误差 ≤ 0.10: {width_acc_010:.2%} ({stats['width_correct_010']}/{total})\n\n")

            if stats['iou_total'] > 0:
                f.write("4. 掩膜重叠准确率\n")
                f.write("-" * 40 + "\n")
                f.write(f"  IoU > 0.5: {iou_above_05_rate:.2%} ({stats['iou_above_05']}/{stats['iou_total']})\n")
                f.write(f"  IoU > 0.7: {iou_above_07_rate:.2%} ({stats['iou_above_07']}/{stats['iou_total']})\n")

        # 打印简明统计
        print("\n" + "=" * 60)
        print("测试准确率统计")
        print("=" * 60)
        print(f"尺寸分类准确率: {size_accuracy:.2%} ({stats['size_correct']}/{total})")
        print(f"高度误差 ≤0.05: {height_acc_005:.2%}")
        print(f"宽度误差 ≤0.05: {width_acc_005:.2%}")
        if stats['iou_total'] > 0:
            print(f"IoU > 0.5: {iou_above_05_rate:.2%}")

        print(f"\n详细报告已保存到: {self.output_dir / 'test_accuracy_report.txt'}")

    def _plot_confusion_matrix(self, results):
        """绘制尺寸分类的混淆矩阵"""
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            # 提取真实类别和预测类别
            y_true = [r['true_class'] for r in results]
            y_pred = [r['pred_class'] for r in results]

            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)

            # 绘制
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.title('尺寸分类混淆矩阵')

            # 添加准确率标注
            total = len(y_true)
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            accuracy = correct / total if total > 0 else 0
            plt.suptitle(f'总体准确率: {accuracy:.2%}', y=1.02)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
            plt.close()

            print(f"混淆矩阵已保存到: {self.output_dir / 'confusion_matrix.png'}")

        except ImportError:
            print("警告: 无法导入sklearn或seaborn，跳过混淆矩阵绘制")
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")


def main():
    """主函数"""
    print("CoarseInfoExtractor 训练与测试程序 (硬分类版本)")
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
        'dropout_rate': 0.2,

        # 硬分类损失参数
        'loss_type': 'hard_classification',  # 使用硬分类损失
        'height_weight': 1.0,  # 高度回归权重
        'width_weight': 0.3,   # 宽度回归权重（辅助任务）
        'size_weight': 1.0,     # 尺寸分类权重
        'class_boundaries': [0.04, 0.08, 0.19, 0.4],  # 分类边界

        # 训练参数
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,

        # 早停参数
        'early_stopping_patience': 6,
        'early_stopping_min_delta': 0.0003,

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
    print("\n步骤 1: 训练 CoarseInfoExtractor 模型 (6类)")
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
            print(f"      预测 - 高度: {result['height_pred']:.4f}, 宽度: {result['width_pred']:.4f}")
            if result['height_true']:
                print(f"      标准 - 高度: {result['height_true']:.4f}, 宽度: {result['width_true']:.4f}")
                print(f"      误差 - 高度: {result['height_error']:.4f}, 宽度: {result['width_error']:.4f}")
            if result['has_aneurysm_mask']:
                print(f"      预测掩膜 - IoU: {result['pred_iou']:.4f}, 覆盖率: {result['pred_coverage']:.2%}")
                if result['gt_iou']:
                    print(f"      标准掩膜 - IoU: {result['gt_iou']:.4f}, 覆盖率: {result['gt_coverage']:.2%}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()