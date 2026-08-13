"""
双通道U-Net分割模型训练代码
输入：DSA图像 + 注意力图（双通道）
输出：动脉瘤分割mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
import warnings
import json
from datetime import datetime
import random
import re
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

# 导入U-Net模型
from Jmodel.unet import LightweightUNet4


# ==============================
# 1. 工具函数
# ==============================

def load_dicom_image(dicom_path: Path, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """加载DICOM图像并归一化"""
    try:
        dicom_data = pydicom.dcmread(str(dicom_path), force=True)
        image = dicom_data.pixel_array.astype(np.float32)

        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            image = np.zeros_like(image)

        if image.shape != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        return image
    except Exception as e:
        print(f"加载DICOM失败 {dicom_path}: {e}")
        return np.zeros(target_size, dtype=np.float32)


def load_attention_map(att_path: Path, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """加载注意力图（PNG格式）并归一化"""
    try:
        att = cv2.imread(str(att_path), cv2.IMREAD_GRAYSCALE)
        if att is None:
            return np.zeros(target_size, dtype=np.float32)

        att = att.astype(np.float32) / 255.0

        if att.shape != target_size:
            att = cv2.resize(att, target_size, interpolation=cv2.INTER_LINEAR)

        return att
    except Exception as e:
        print(f"加载注意力图失败 {att_path}: {e}")
        return np.zeros(target_size, dtype=np.float32)


def load_mask(mask_path: Path, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """加载mask图像（.tif格式）并二值化"""
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros(target_size, dtype=np.float32)

        if mask.max() > 1.0:
            mask = mask / 255.0

        mask = (mask > 0.5).astype(np.float32)

        if mask.shape != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        return mask
    except Exception as e:
        print(f"加载mask失败 {mask_path}: {e}")
        return np.zeros(target_size, dtype=np.float32)


# ==============================
# 2. 权重图生成
# ==============================

def create_weight_map_scheme1(mask: torch.Tensor, high_weight: float = 10.0, threshold: float = 0.5) -> torch.Tensor:
    """基于GT mask生成权重图"""
    weight_map = torch.ones_like(mask)
    high_weight_region = (mask > threshold).float()
    weight_map = weight_map + (high_weight_region * (high_weight - 1.0))
    return weight_map


def create_weight_map_scheme2(mask: torch.Tensor, pred: torch.Tensor, high_weight: float = 10.0,
                              threshold: float = 0.5) -> torch.Tensor:
    """基于GT和预测的并集生成权重图"""
    weight_map = torch.ones_like(mask)
    mask_region = (mask > threshold).float()
    pred_region = (pred > threshold).float()
    high_weight_region = ((mask_region + pred_region) > 0.5).float()
    weight_map = weight_map + (high_weight_region * (high_weight - 1.0))
    return weight_map


# ==============================
# 3. 损失函数
# ==============================

class WeightedCombinedLoss(nn.Module):
    """加权组合损失：Dice Loss + L1 Loss"""

    def __init__(self,
                 l1_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 smooth: float = 1e-6,
                 weight_scheme: str = 'scheme1',
                 high_weight: float = 10.0,
                 threshold: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.weight_scheme = weight_scheme
        self.high_weight = high_weight
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.weight_scheme == 'scheme1':
            weight_map = create_weight_map_scheme1(target, self.high_weight, self.threshold)
        else:
            weight_map = create_weight_map_scheme2(target, pred, self.high_weight, self.threshold)

        weight_map = weight_map.to(pred.device)

        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        weight_flat = weight_map.contiguous().view(-1)

        weighted_intersection = (pred_flat * target_flat * weight_flat).sum()
        weighted_union = (pred_flat * weight_flat).sum() + (target_flat * weight_flat).sum()

        dice_loss = 1 - (2. * weighted_intersection + self.smooth) / (weighted_union + self.smooth)

        l1_loss = F.l1_loss(pred, target, reduction='none')
        weighted_l1 = (l1_loss * weight_map).mean()

        total_loss = self.dice_weight * dice_loss + self.l1_weight * weighted_l1

        loss_components = {
            'dice_loss': dice_loss.item(),
            'l1_loss': weighted_l1.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_components


# ==============================
# 4. 评估指标
# ==============================

class SegmentationMetrics:
    """分割评估指标"""

    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return float(dice)

    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)

    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        precision = (tp + smooth) / (tp + fp + smooth)
        return float(precision)

    @staticmethod
    def recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        recall = (tp + smooth) / (tp + fn + smooth)
        return float(recall)

    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        specificity = (tn + smooth) / (tn + fp + smooth)
        return float(specificity)

    @staticmethod
    def get_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        pred_binary = (pred > 0.5).astype(np.float32)
        target_binary = (target > 0.5).astype(np.float32)

        return {
            'dice': SegmentationMetrics.dice_coefficient(pred_binary, target_binary),
            'iou': SegmentationMetrics.iou_score(pred_binary, target_binary),
            'precision': SegmentationMetrics.precision(pred_binary, target_binary),
            'recall': SegmentationMetrics.recall(pred_binary, target_binary),
            'specificity': SegmentationMetrics.specificity(pred_binary, target_binary)
        }


# ==============================
# 5. 数据匹配器（双通道版本）
# ==============================

class DualChannelDataMatcher:
    """匹配DICOM、注意力图和mask"""

    def __init__(self, dicom_dir: str, attention_dir: str, mask_dir: str):
        self.dicom_dir = Path(dicom_dir)
        self.attention_dir = Path(attention_dir)
        self.mask_dir = Path(mask_dir)

    def find_paired_data(self) -> List[Tuple[Path, Path, Path]]:
        """找到匹配的DICOM、注意力图和mask三元组"""
        paired_data = []

        # 获取所有DICOM文件
        dicom_files = []
        for file_path in self.dicom_dir.iterdir():
            if file_path.is_file() and not file_path.suffix:
                dicom_files.append(file_path)
        print(f"找到 {len(dicom_files)} 个DICOM文件")

        # 获取所有注意力图（PNG格式，以_att结尾）
        attention_files = list(self.attention_dir.glob("*_att.png"))
        print(f"找到 {len(attention_files)} 个注意力图文件")

        # 获取所有mask文件（.tif格式）
        mask_files = list(self.mask_dir.glob("*.tif"))
        print(f"找到 {len(mask_files)} 个mask文件")

        # 建立索引
        attention_index = {}
        for att_path in attention_files:
            # 从 "ANY_001_0_att.png" 提取 "ANY_001_0"
            stem = att_path.stem.replace('_att', '')
            attention_index[stem] = att_path

        mask_index = {mask_path.stem: mask_path for mask_path in mask_files}

        # 匹配
        for dicom_path in dicom_files:
            dicom_stem = dicom_path.name
            if dicom_stem in attention_index and dicom_stem in mask_index:
                paired_data.append((
                    dicom_path,
                    attention_index[dicom_stem],
                    mask_index[dicom_stem]
                ))

        print(f"成功匹配 {len(paired_data)} 组数据")

        if paired_data:
            print("\n匹配示例:")
            for i, (d, a, m) in enumerate(paired_data[:5]):
                print(f"  {i+1}. {d.name} + {a.name} -> {m.name}")

        return paired_data


# ==============================
# 6. 双通道数据集
# ==============================

class DualChannelSegmentationDataset(Dataset):
    """双通道分割数据集"""

    def __init__(self, paired_data: List[Tuple[Path, Path, Path]],
                 image_size: Tuple[int, int] = (512, 512)):
        self.paired_data = paired_data
        self.image_size = image_size

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        dicom_path, att_path, mask_path = self.paired_data[idx]

        # 加载DICOM图像
        dicom_img = load_dicom_image(dicom_path, self.image_size)

        # 加载注意力图
        att_img = load_attention_map(att_path, self.image_size)

        # 加载mask
        mask = load_mask(mask_path, self.image_size)

        # 堆叠为双通道输入 [2, H, W]
        image = np.stack([dicom_img, att_img], axis=0)

        # 添加mask通道维度 [1, H, W]
        mask = np.expand_dims(mask, axis=0)

        return {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).float(),
            'dicom_name': dicom_path.name,
            'att_name': att_path.name,
            'mask_name': mask_path.name
        }


# ==============================
# 7. 流式数据管理器
# ==============================

class StreamDataManager:
    """流式数据管理器"""

    def __init__(self, cache_root: str = "stream_cache_seg_dual"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def create_cache_session(self, prefix: str = "seg_dual") -> str:
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        session_name = f"{date_str}_{random_num}"
        session_path = self.cache_root / session_name
        session_path.mkdir(parents=True, exist_ok=True)

        (session_path / "train").mkdir(exist_ok=True)
        (session_path / "val").mkdir(exist_ok=True)
        (session_path / "test").mkdir(exist_ok=True)

        print(f"缓存会话创建: {session_path}")
        return str(session_path)

    def get_existing_sessions(self) -> List[str]:
        sessions = []
        for item in self.cache_root.iterdir():
            if item.is_dir() and re.match(r'\d{8}_\d{3}', item.name):
                sessions.append(str(item))
        return sorted(sessions)

    def save_sample(self, data: Dict, save_path: str):
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
        np.savez(save_path, **save_dict)


class StreamDualChannelDataset(Dataset):
    """流式双通道数据集"""

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir) / split
        self.sample_files = list(self.data_dir.glob("*.npz"))
        self.sample_files.sort()
        print(f"加载{split}数据集: {len(self.sample_files)}个样本")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        data = np.load(sample_path, allow_pickle=True)

        image = torch.from_numpy(data['image']).float()
        mask = torch.from_numpy(data['mask']).float()
        dicom_name = str(data['dicom_name'].item()) if data['dicom_name'].ndim == 0 else str(data['dicom_name'][0])
        att_name = str(data['att_name'].item()) if data['att_name'].ndim == 0 else str(data['att_name'][0])
        mask_name = str(data['mask_name'].item()) if data['mask_name'].ndim == 0 else str(data['mask_name'][0])

        return {
            'image': image,
            'mask': mask,
            'dicom_name': dicom_name,
            'att_name': att_name,
            'mask_name': mask_name
        }


# ==============================
# 8. 数据预处理器（双通道版本）
# ==============================

class DualChannelDataPreprocessor:
    """双通道数据预处理器"""

    def __init__(self, dicom_dir: str, attention_dir: str, mask_dir: str,
                 image_size: Tuple[int, int] = (512, 512)):
        self.dicom_dir = Path(dicom_dir)
        self.attention_dir = Path(attention_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.matcher = DualChannelDataMatcher(dicom_dir, attention_dir, mask_dir)
        self.stream_manager = StreamDataManager()

    def process_and_save(self, output_session: str = None,
                         val_ratio: float = 0.1,
                         test_ratio: float = 0.1,
                         force_new: bool = True) -> str:
        """处理并保存数据到缓存"""
        paired_data = self.matcher.find_paired_data()

        if not paired_data:
            raise ValueError("没有找到匹配的数据")

        random.shuffle(paired_data)

        n_total = len(paired_data)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        train_data = paired_data[:n_train]
        val_data = paired_data[n_train:n_train + n_val]
        test_data = paired_data[n_train + n_val:]

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_data)} 样本 ({len(train_data)/n_total*100:.1f}%)")
        print(f"  验证集: {len(val_data)} 样本 ({len(val_data)/n_total*100:.1f}%)")
        print(f"  测试集: {len(test_data)} 样本 ({len(test_data)/n_total*100:.1f}%)")

        if output_session is None:
            output_session = self.stream_manager.create_cache_session()

        output_path = Path(output_session)

        # 保存各数据集
        self._save_split(train_data, output_path / "train", "训练集")
        self._save_split(val_data, output_path / "val", "验证集")
        self._save_split(test_data, output_path / "test", "测试集")

        # 保存元数据
        metadata = {
            'total_samples': n_total,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'image_size': self.image_size,
            'input_channels': 2,
            'created_at': datetime.now().isoformat()
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n预处理完成！保存至: {output_session}")
        return output_session

    def _save_split(self, data_list: List, save_dir: Path, desc: str):
        """保存数据划分"""
        print(f"\n保存{desc} ({len(data_list)}个样本)...")
        for dicom_path, att_path, mask_path in tqdm(data_list, desc=desc):
            dicom_img = load_dicom_image(dicom_path, self.image_size)
            att_img = load_attention_map(att_path, self.image_size)
            mask = load_mask(mask_path, self.image_size)

            image = np.stack([dicom_img, att_img], axis=0)
            mask = np.expand_dims(mask, axis=0)

            sample_data = {
                'image': image,
                'mask': mask,
                'dicom_name': np.array([dicom_path.name]),
                'att_name': np.array([att_path.name]),
                'mask_name': np.array([mask_path.name])
            }

            save_path = save_dir / f"{dicom_path.name}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path))


# ==============================
# 9. 早停机制
# ==============================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=15, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        return self.early_stop


# ==============================
# 10. 双通道U-Net包装器
# ==============================

class DualChannelUNet(nn.Module):
    """双通道输入的U-Net包装器"""

    def __init__(self, base_channels=32, dropout_rate=0.0, negative_slope=0.01):
        super().__init__()
        self.unet = LightweightUNet4(
            in_channels=2,  # 双通道输入
            out_channels=1,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
            negative_slope=negative_slope
        )

    def forward(self, x):
        return self.unet(x)


# ==============================
# 11. 训练器
# ==============================

class DualChannelSegmentationTrainer:
    """双通道分割模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        model_dir_name = f"dual_channel_{date_str}_{random_num}"

        self.output_dir = Path(config['model_save_root']) / model_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.test_results_dir = self.output_dir / "test_results"

        for dir_path in [self.checkpoint_dir, self.results_dir, self.test_results_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"模型保存目录: {self.output_dir}")

        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_dice': [], 'val_iou': [],
            'learning_rate': [],
            'train_dice_loss': [], 'train_l1_loss': []
        }

        # 损失函数
        self.criterion = WeightedCombinedLoss(
            l1_weight=config.get('l1_weight', 1.0),
            dice_weight=config.get('dice_weight', 1.0),
            weight_scheme=config.get('weight_scheme', 'scheme1'),
            high_weight=config.get('high_weight', 10.0),
            threshold=config.get('threshold', 0.5)
        )

        self.metrics = SegmentationMetrics()

    def _create_model(self):
        """创建双通道U-Net模型"""
        model = DualChannelUNet(
            base_channels=self.config['base_channels'],
            dropout_rate=self.config['dropout_rate'],
            negative_slope=self.config.get('negative_slope', 0.01)
        ).to(self.device)

        return model

    def _create_dataloaders(self, cache_session: str):
        """创建数据加载器"""
        train_dataset = StreamDualChannelDataset(cache_session, split="train")
        val_dataset = StreamDualChannelDataset(cache_session, split="val")

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

        print(f"训练集: {len(train_dataset)} 样本 ({len(train_loader)} batches)")
        print(f"验证集: {len(val_dataset)} 样本 ({len(val_loader)} batches)")

        return train_loader, val_loader

    def train_epoch(self, model, train_loader, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_dice_loss = 0
        total_l1_loss = 0

        for batch in tqdm(train_loader, desc="训练", leave=False):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_components = self.criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_dice_loss += loss_components['dice_loss']
            total_l1_loss += loss_components['l1_loss']

        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'dice_loss': total_dice_loss / n_batches,
            'l1_loss': total_l1_loss / n_batches
        }

    def validate(self, model, val_loader):
        """验证"""
        model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_dice_loss = 0
        total_l1_loss = 0
        valid_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证", leave=False):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = model(images)
                loss, loss_components = self.criterion(outputs, masks)

                outputs_np = outputs.cpu().numpy()
                masks_np = masks.cpu().numpy()

                for i in range(outputs_np.shape[0]):
                    pred = (outputs_np[i, 0] > 0.5).astype(np.float32)
                    target = (masks_np[i, 0] > 0.5).astype(np.float32)

                    if target.sum() > 0:
                        dice = self.metrics.dice_coefficient(pred, target)
                        iou = self.metrics.iou_score(pred, target)
                        total_dice += dice
                        total_iou += iou
                        valid_samples += 1

                total_loss += loss.item()
                total_dice_loss += loss_components['dice_loss']
                total_l1_loss += loss_components['l1_loss']

        n_batches = len(val_loader)
        return {
            'loss': total_loss / n_batches,
            'dice_loss': total_dice_loss / n_batches,
            'l1_loss': total_l1_loss / n_batches,
            'dice': total_dice / valid_samples if valid_samples > 0 else 0,
            'iou': total_iou / valid_samples if valid_samples > 0 else 0
        }

    def train(self, cache_session: str):
        """执行完整训练流程"""
        print("\n" + "=" * 60)
        print("开始双通道U-Net训练")
        print(f"输入通道: 2 (DSA + 注意力图)")
        print(f"损失配置: {self.config.get('weight_scheme', 'scheme1')}, 高权重={self.config.get('high_weight', 10.0)}")
        print("=" * 60)

        model = self._create_model()
        print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

        train_loader, val_loader = self._create_dataloaders(cache_session)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )

        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 20),
            min_delta=self.config.get('early_stopping_min_delta', 0.001),
            mode='min'
        )

        best_val_loss = float('inf')

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            train_metrics = self.train_epoch(model, train_loader, optimizer)
            val_metrics = self.validate(model, val_loader)

            scheduler.step(val_metrics['loss'])

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice_loss'].append(train_metrics['dice_loss'])
            self.history['train_l1_loss'].append(train_metrics['l1_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            print(f"  训练 Loss: {train_metrics['loss']:.4f} (Dice: {train_metrics['dice_loss']:.4f}, L1: {train_metrics['l1_loss']:.4f})")
            print(f"  验证 Loss: {val_metrics['loss']:.4f} (Dice: {val_metrics['dice_loss']:.4f}, L1: {val_metrics['l1_loss']:.4f})")
            print(f"  验证 Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = self.checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'config': self.config,
                    'history': self.history
                }, best_model_path)
                print(f"  ✓ 保存最佳模型至: {best_model_path}")

            if early_stopping(val_metrics['loss']):
                print(f"\n🚨 早停触发! {early_stopping.patience}个epoch无改善")
                break

        final_model_path = self.checkpoint_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history
        }, final_model_path)

        self._plot_training_history()
        self._save_training_history()
        return final_model_path

    def _plot_training_history(self):
        """绘制训练历史"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(epochs, self.history['train_loss'], label='训练总损失')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='验证总损失')
        axes[0, 0].set_title('组合损失')
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.history['train_dice_loss'], label='训练Dice损失')
        axes[0, 1].set_title('Dice损失')

        axes[0, 2].plot(epochs, self.history['train_l1_loss'], label='训练L1损失', color='orange')
        axes[0, 2].set_title('L1损失')

        axes[1, 0].plot(epochs, self.history['val_dice'], color='purple')
        axes[1, 0].set_title('验证Dice')

        axes[1, 1].plot(epochs, self.history['val_iou'], color='brown')
        axes[1, 1].set_title('验证IoU')

        axes[1, 2].plot(epochs, self.history['learning_rate'], 'm-')
        axes[1, 2].set_title('学习率')
        axes[1, 2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=300)
        plt.close()

    def _save_training_history(self):
        """保存训练历史"""
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, default=str)


# ==============================
# 12. 测试器
# ==============================

class DualChannelSegmentationTester:
    """双通道分割模型测试器"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._load_model(model_path)
        self.model.eval()

        self.metrics = SegmentationMetrics()

        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        self.test_output_dir = Path(config['model_save_root']) / f"test_dual_{date_str}_{random_num}"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        self.comparison_dir = self.test_output_dir / "comparison"
        self.comparison_dir.mkdir(exist_ok=True)

    def _load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = DualChannelUNet(
            base_channels=self.config['base_channels'],
            dropout_rate=self.config['dropout_rate'],
            negative_slope=self.config.get('negative_slope', 0.01)
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        val_dice = checkpoint.get('val_dice', 'Unknown')

        print(f"模型加载成功")
        print(f"  训练轮数: {epoch}")
        if isinstance(val_loss, float):
            print(f"  验证损失: {val_loss:.4f}")
        if isinstance(val_dice, float):
            print(f"  验证Dice: {val_dice:.4f}")

        return model

    def test(self, cache_session: str):
        """测试模型"""
        print("\n" + "=" * 60)
        print("开始模型测试")
        print("=" * 60)

        test_dataset = StreamDualChannelDataset(cache_session, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('test_batch_size', 4),
            shuffle=False,
            num_workers=2
        )

        print(f"测试集: {len(test_dataset)} 样本")

        all_metrics = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试")):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                dicom_names = batch['dicom_name']

                outputs = self.model(images)

                images_np = images.cpu().numpy()
                masks_np = masks.cpu().numpy()
                outputs_np = outputs.cpu().numpy()

                for i in range(images_np.shape[0]):
                    dsa_img = images_np[i, 0]
                    att_img = images_np[i, 1]
                    true_mask = masks_np[i, 0]
                    pred_mask = outputs_np[i, 0]

                    pred_binary = (pred_mask > 0.5).astype(np.float32)
                    true_binary = (true_mask > 0.5).astype(np.float32)

                    if true_binary.sum() > 0:
                        metrics = self.metrics.get_all_metrics(pred_binary, true_binary)
                    else:
                        metrics = {
                            'dice': 1.0 if pred_binary.sum() == 0 else 0.0,
                            'iou': 1.0 if pred_binary.sum() == 0 else 0.0,
                            'precision': 1.0 if pred_binary.sum() == 0 else 0.0,
                            'recall': 1.0 if true_binary.sum() == 0 else 0.0,
                            'specificity': 1.0
                        }

                    metrics['dicom_name'] = dicom_names[i]
                    all_metrics.append(metrics)

                    if batch_idx * test_loader.batch_size + i < 100:
                        self.save_comparison_image(
                            dsa_img, att_img, true_mask, pred_mask,
                            pred_binary, true_binary,
                            dicom_names[i], metrics
                        )

        avg_metrics = {}
        for key in ['dice', 'iou', 'precision', 'recall', 'specificity']:
            values = [m[key] for m in all_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)

        self.save_test_results(all_metrics, avg_metrics)

        return all_metrics, avg_metrics

    def save_comparison_image(self, dsa_img, att_img, true_mask, pred_mask,
                              pred_binary, true_binary, dicom_name, metrics):
        """保存对比图"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            # DSA图像
            axes[0, 0].imshow(dsa_img, cmap='gray')
            axes[0, 0].set_title('DSA图像')
            axes[0, 0].axis('off')

            # 注意力图
            axes[0, 1].imshow(att_img, cmap='hot')
            axes[0, 1].set_title('注意力图')
            axes[0, 1].axis('off')

            # 真实mask
            axes[0, 2].imshow(true_mask, cmap='gray')
            axes[0, 2].set_title('真实Mask')
            axes[0, 2].axis('off')

            # 预测mask（概率）
            axes[0, 3].imshow(pred_mask, cmap='viridis')
            axes[0, 3].set_title('预测Mask（概率）')
            axes[0, 3].axis('off')

            # 二值化预测
            axes[1, 0].imshow(pred_binary, cmap='gray')
            axes[1, 0].set_title('二值化预测')
            axes[1, 0].axis('off')

            # DSA+注意力叠加
            axes[1, 1].imshow(dsa_img, cmap='gray')
            axes[1, 1].imshow(att_img, cmap='hot', alpha=0.5)
            axes[1, 1].set_title('DSA+注意力叠加')
            axes[1, 1].axis('off')

            # 预测与真实叠加
            overlay = np.zeros((*dsa_img.shape, 3), dtype=np.float32)
            overlay[..., 0] = pred_binary * 0.7
            overlay[..., 1] = true_binary * 0.7
            overlap = np.logical_and(pred_binary > 0.5, true_binary > 0.5)
            overlay[overlap, 0] = 1.0
            overlay[overlap, 1] = 1.0
            overlay[overlap, 2] = 0.0

            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('叠加（红:预测, 绿:真实, 黄:重叠）')
            axes[1, 2].axis('off')

            # 指标信息
            axes[1, 3].axis('off')
            info_text = f"Dice: {metrics['dice']:.4f}\n"
            info_text += f"IoU: {metrics['iou']:.4f}\n"
            info_text += f"Precision: {metrics['precision']:.4f}\n"
            info_text += f"Recall: {metrics['recall']:.4f}\n"
            info_text += f"Specificity: {metrics['specificity']:.4f}"

            axes[1, 3].text(0.1, 0.5, info_text, transform=axes[1, 3].transAxes,
                            fontsize=12, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f'分割结果 - {dicom_name}', fontsize=14)
            plt.tight_layout()

            save_path = self.comparison_dir / f"{dicom_name}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"保存对比图失败: {e}")

    def save_test_results(self, all_metrics, avg_metrics):
        """保存测试结果"""
        df = pd.DataFrame(all_metrics)
        df.to_csv(self.test_output_dir / 'test_results.csv', index=False)

        with open(self.test_output_dir / 'test_summary.txt', 'w', encoding='utf-8') as f:
            f.write("双通道U-Net分割模型测试结果\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试样本数: {len(all_metrics)}\n\n")
            f.write("平均指标:\n")
            f.write("-" * 40 + "\n")
            for key in ['dice', 'iou', 'precision', 'recall', 'specificity']:
                f.write(f"  {key}: {avg_metrics[f'avg_{key}']:.4f} (±{avg_metrics[f'std_{key}']:.4f})\n")

        print(f"\n测试结果保存至: {self.test_output_dir}")
        print(f"平均Dice: {avg_metrics['avg_dice']:.4f} (±{avg_metrics['std_dice']:.4f})")
        print(f"平均IoU: {avg_metrics['avg_iou']:.4f} (±{avg_metrics['std_iou']:.4f})")


# ==============================
# 13. 主函数
# ==============================

def main():
    """主函数"""
    print("双通道U-Net分割训练")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'dicom_dir': "D:/med_data/ai/translate/reverse/train",
        'attention_dir': "D:/med_data/multi/ATT",
        'mask_dir': "D:/med_data/ai/translate/all_mask",
        'cache_root': "D:/med_data/multi/stream_cache_seg_dual",
        'model_save_root': "D:/med_data/multi/dual_channel_models",

        # 模型参数
        'image_size': (512, 512),
        'base_channels': 32,
        'dropout_rate': 0.1,
        'negative_slope': 0.01,

        # 损失参数
        'l1_weight': 10.0,
        'dice_weight': 1.0,
        'weight_scheme': 'scheme1',
        'high_weight': 10.0,
        'threshold': 0.5,

        # 训练参数
        'batch_size': 8,
        'test_batch_size': 8,
        'num_epochs': 75,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0005,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'num_workers': 2,
        'use_existing_cache': False
    }

    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 步骤1: 数据预处理
    print("\n" + "=" * 60)
    print("步骤1: 数据预处理（双通道）")
    print("=" * 60)

    preprocessor = DualChannelDataPreprocessor(
        config['dicom_dir'],
        config['attention_dir'],
        config['mask_dir'],
        config['image_size']
    )

    if config['use_existing_cache']:
        stream_manager = StreamDataManager(cache_root=config['cache_root'])
        existing_sessions = stream_manager.get_existing_sessions()
        cache_session = existing_sessions[-1] if existing_sessions else preprocessor.process_and_save(
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio']
        )
    else:
        cache_session = preprocessor.process_and_save(
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio']
        )

    print(f"缓存会话: {cache_session}")

    # 步骤2: 训练
    print("\n" + "=" * 60)
    print("步骤2: 训练双通道U-Net")
    print("=" * 60)

    trainer = DualChannelSegmentationTrainer(config)
    trained_model_path = trainer.train(cache_session)
    print(f"模型保存至: {trained_model_path}")

    # 步骤3: 测试
    print("\n" + "=" * 60)
    print("步骤3: 测试模型")
    print("=" * 60)

    tester = DualChannelSegmentationTester(str(trained_model_path), config)
    test_results, avg_metrics = tester.test(cache_session)

    print("\n" + "=" * 60)
    print("流程完成!")
    print(f"平均Dice: {avg_metrics['avg_dice']:.4f} (±{avg_metrics['std_dice']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()