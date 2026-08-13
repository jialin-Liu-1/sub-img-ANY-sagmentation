import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
from skimage.measure import label, regionprops
import pandas as pd

warnings.filterwarnings('ignore')


# ==============================
# 轻量级U-Net模型
# ==============================

class LightweightUNet(nn.Module):
    """
    轻量级U-Net分割网络
    只在注意力区域内进行精细分割
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 dropout_rate: float = 0.1):
        super().__init__()

        # ========== 编码器部分 ==========
        self.encoder1 = self._make_encoder_block(in_channels, base_channels, dropout_rate)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels * 2, dropout_rate)
        self.encoder3 = self._make_encoder_block(base_channels * 2, base_channels * 4, dropout_rate)

        # ========== 瓶颈层 ==========
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # ========== 解码器部分 ==========
        self.decoder3 = self._make_decoder_block(base_channels * 8, base_channels * 2, dropout_rate)
        self.decoder2 = self._make_decoder_block(base_channels * 4, base_channels, dropout_rate)
        self.decoder1 = self._make_decoder_block(base_channels * 2, base_channels, dropout_rate)

        # ========== 输出层 ==========
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _make_encoder_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

    def _make_decoder_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout_rate)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: [B, 1, H, W] 经过注意力mask处理的DSA图像
        返回:
            segmentation: [B, 1, H, W] 动脉瘤分割结果
        """
        # ========== 编码器路径 ==========
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # ========== 瓶颈层 ==========
        bottleneck = self.bottleneck(enc3)

        # ========== 解码器路径 ==========
        dec3 = self.decoder3(torch.cat([bottleneck, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        # ========== 输出分割 ==========
        segmentation = self.output_conv(dec1)

        # 确保输出尺寸与输入一致
        if segmentation.shape[2:] != x.shape[2:]:
            segmentation = F.interpolate(segmentation, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)

        return segmentation

# ==============================
# Dice损失函数
# ==============================

class DiceLoss(nn.Module):
    """Dice损失函数"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失
        参数:
            pred: 预测分割图 [B, 1, H, W]
            target: 真实分割图 [B, 1, H, W]
        """
        # 展平
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 返回Dice损失
        return 1 - dice


# ==============================
# 评估指标
# ==============================

class SegmentationMetrics:
    """分割评估指标"""

    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """计算Dice系数"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return float(dice)

    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """计算IoU分数"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        return float(iou)

    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """计算精确率"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        true_positive = (pred_flat * target_flat).sum()
        false_positive = (pred_flat * (1 - target_flat)).sum()

        precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
        return float(precision)

    @staticmethod
    def recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """计算召回率"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        true_positive = (pred_flat * target_flat).sum()
        false_negative = ((1 - pred_flat) * target_flat).sum()

        recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
        return float(recall)

    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """计算特异性"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        true_negative = ((1 - pred_flat) * (1 - target_flat)).sum()
        false_positive = (pred_flat * (1 - target_flat)).sum()

        specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
        return float(specificity)

    @staticmethod
    def get_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """计算所有指标"""
        # 二值化
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
# 数据加载和预处理
# ==============================

class DataMatcher:
    """数据匹配器：匹配DICOM图像和mask图像"""

    def __init__(self, dicom_dir: str, mask_dir: str):
        self.dicom_dir = Path(dicom_dir)
        self.mask_dir = Path(mask_dir)

    def find_paired_data(self) -> List[Tuple[Path, Path]]:
        """查找配对的DICOM图像和mask图像"""
        paired_data = []

        # 获取所有DICOM文件（无后缀）
        dicom_files = []
        for file_path in self.dicom_dir.iterdir():
            if file_path.is_file() and not file_path.suffix:  # 无后缀文件
                dicom_files.append(file_path)

        print(f"找到 {len(dicom_files)} 个DICOM文件")

        # 获取所有mask文件（.tif）
        mask_files = list(self.mask_dir.glob("*.tif"))
        print(f"找到 {len(mask_files)} 个mask文件")

        # 建立mask文件索引
        mask_index = {mask_path.stem: mask_path for mask_path in mask_files}

        # 匹配
        for dicom_path in dicom_files:
            dicom_stem = dicom_path.name  # 无后缀，直接使用文件名
            if dicom_stem in mask_index:
                paired_data.append((dicom_path, mask_index[dicom_stem]))

        print(f"成功匹配 {len(paired_data)} 对数据")

        # 显示一些示例
        if paired_data:
            print("\n匹配示例:")
            for i, (dicom_path, mask_path) in enumerate(paired_data[:5]):
                print(f"  {i + 1}. {dicom_path.name} <-> {mask_path.name}")

        return paired_data


class SegmentationDataset(Dataset):
    """分割数据集"""

    def __init__(self, paired_data: List[Tuple[Path, Path]],
                 image_size: Tuple[int, int] = (512, 512),
                 transform=None):
        self.paired_data = paired_data
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.paired_data)

    def _load_dicom(self, dicom_path: Path) -> np.ndarray:
        """加载DICOM图像"""
        try:
            # 读取DICOM文件
            dicom_data = pydicom.dcmread(str(dicom_path), force=True)
            image = dicom_data.pixel_array.astype(np.float32)

            # 归一化
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # 调整大小
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载DICOM失败 {dicom_path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """加载mask图像"""
        try:
            # 读取mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"警告: 无法读取mask {mask_path.name}")
                return np.zeros(self.image_size, dtype=np.float32)

            # 转换为0-1范围
            if mask.max() > 1.0:
                mask = mask / 255.0

            # 二值化
            mask = (mask > 0.5).astype(np.float32)

            # 调整大小
            if mask.shape != self.image_size:
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            return mask

        except Exception as e:
            print(f"加载mask失败 {mask_path}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)

    def __getitem__(self, idx):
        dicom_path, mask_path = self.paired_data[idx]

        # 加载数据
        image = self._load_dicom(dicom_path)
        mask = self._load_mask(mask_path)

        # 添加通道维度
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # 转换为tensor
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'dicom_name': dicom_path.name,
            'mask_name': mask_path.name
        }


# ==============================
# 流式数据管理器
# ==============================

class StreamDataManager:
    """流式数据管理器"""

    def __init__(self, cache_root: str = "D:/med_data/ai/stream_cache_seg"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def create_cache_session(self, prefix: str = "seg_dataset") -> str:
        """创建新的缓存会话"""
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        session_name = f"{date_str}_{random_num}"
        session_path = self.cache_root / session_name
        session_path.mkdir(parents=True, exist_ok=True)

        # 创建训练、验证和测试子文件夹
        (session_path / "train").mkdir(exist_ok=True)
        (session_path / "val").mkdir(exist_ok=True)
        (session_path / "test").mkdir(exist_ok=True)

        print(f"创建缓存会话: {session_path}")
        return str(session_path)

    def get_existing_sessions(self) -> List[str]:
        """获取所有现有缓存会话"""
        sessions = []
        for item in self.cache_root.iterdir():
            if item.is_dir() and re.match(r'\d{8}_\d{3}', item.name):
                sessions.append(str(item))
        return sorted(sessions)

    def save_sample(self, data: Dict, save_path: str):
        """保存样本"""
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

    def load_sample(self, npz_path: str) -> Dict:
        """加载样本"""
        data = np.load(npz_path, allow_pickle=True)

        result = {}
        for key in data.files:
            arr = data[key]
            if arr.dtype == np.object_ and len(arr) == 1:
                try:
                    result[key] = str(arr[0])
                except:
                    result[key] = arr[0]
            elif arr.ndim == 0:
                result[key] = arr.item()
            else:
                result[key] = torch.from_numpy(arr).float()

        return result


class StreamSegmentationDataset(Dataset):
    """使用流式数据的分割数据集"""

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
        mask_name = str(data['mask_name'].item()) if data['mask_name'].ndim == 0 else str(data['mask_name'][0])

        return {
            'image': image,
            'mask': mask,
            'dicom_name': dicom_name,
            'mask_name': mask_name
        }


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, dicom_dir: str, mask_dir: str, image_size: Tuple[int, int] = (512, 512)):
        self.dicom_dir = Path(dicom_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.matcher = DataMatcher(dicom_dir, mask_dir)
        self.stream_manager = StreamDataManager()

    def _load_dicom(self, dicom_path: Path) -> np.ndarray:
        """加载DICOM图像"""
        try:
            dicom_data = pydicom.dcmread(str(dicom_path), force=True)
            image = dicom_data.pixel_array.astype(np.float32)

            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            return np.expand_dims(image, axis=0)

        except Exception as e:
            print(f"加载DICOM失败 {dicom_path}: {e}")
            return np.zeros((1, *self.image_size), dtype=np.float32)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """加载mask图像"""
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                return np.zeros((1, *self.image_size), dtype=np.float32)

            if mask.max() > 1.0:
                mask = mask / 255.0

            mask = (mask > 0.5).astype(np.float32)

            if mask.shape != self.image_size:
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            return np.expand_dims(mask, axis=0)

        except Exception as e:
            print(f"加载mask失败 {mask_path}: {e}")
            return np.zeros((1, *self.image_size), dtype=np.float32)

    def process_and_save(self, output_session: str = None,
                         val_ratio: float = 0.07,  # 7%用于验证
                         test_ratio: float = 0.07,  # 7%用于测试
                         force_new: bool = True) -> str:
        """处理数据并保存"""
        # 获取配对数据
        paired_data = self.matcher.find_paired_data()

        if not paired_data:
            raise ValueError("没有找到配对的数据")

        # 随机打乱
        random.shuffle(paired_data)

        # 划分数据集
        n_total = len(paired_data)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        train_data = paired_data[:n_train]
        val_data = paired_data[n_train:n_train + n_val]
        test_data = paired_data[n_train + n_val:]

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_data)} 样本 ({len(train_data) / n_total * 100:.1f}%)")
        print(f"  验证集: {len(val_data)} 样本 ({len(val_data) / n_total * 100:.1f}%)")
        print(f"  测试集: {len(test_data)} 样本 ({len(test_data) / n_total * 100:.1f}%)")

        # 创建输出会话
        if output_session is None:
            output_session = self.stream_manager.create_cache_session()

        output_path = Path(output_session)

        # 保存训练集
        print(f"\n保存训练集 ({len(train_data)}个样本)...")
        for i, (dicom_path, mask_path) in enumerate(tqdm(train_data, desc="训练集")):
            image = self._load_dicom(dicom_path)
            mask = self._load_mask(mask_path)

            sample_data = {
                'image': image,
                'mask': mask,
                'dicom_name': np.array([dicom_path.name]),
                'mask_name': np.array([mask_path.name])
            }

            save_path = output_path / "train" / f"{dicom_path.name}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path))

        # 保存验证集
        print(f"保存验证集 ({len(val_data)}个样本)...")
        for dicom_path, mask_path in tqdm(val_data, desc="验证集"):
            image = self._load_dicom(dicom_path)
            mask = self._load_mask(mask_path)

            sample_data = {
                'image': image,
                'mask': mask,
                'dicom_name': np.array([dicom_path.name]),
                'mask_name': np.array([mask_path.name])
            }

            save_path = output_path / "val" / f"{dicom_path.name}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path))

        # 保存测试集
        print(f"保存测试集 ({len(test_data)}个样本)...")
        for dicom_path, mask_path in tqdm(test_data, desc="测试集"):
            image = self._load_dicom(dicom_path)
            mask = self._load_mask(mask_path)

            sample_data = {
                'image': image,
                'mask': mask,
                'dicom_name': np.array([dicom_path.name]),
                'mask_name': np.array([mask_path.name])
            }

            save_path = output_path / "test" / f"{dicom_path.name}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path))

        # 保存元数据
        metadata = {
            'total_samples': n_total,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_ratio': 1 - val_ratio - test_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'image_size': self.image_size,
            'created_at': datetime.now().isoformat()
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n数据预处理完成! 保存到: {output_session}")
        return output_session


# ==============================
# 早停机制
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
        else:  # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        return self.early_stop


# ==============================
# 训练器
# ==============================

class SegmentationTrainer:
    """分割模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        model_dir_name = f"{date_str}_{random_num}"

        self.output_dir = Path(config['model_save_root']) / model_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.test_results_dir = self.output_dir / "test_results"

        for dir_path in [self.checkpoint_dir, self.results_dir, self.test_results_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"模型保存目录: {self.output_dir}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'learning_rate': []
        }

        # 损失函数
        self.criterion = DiceLoss()
        self.metrics = SegmentationMetrics()

    def _create_model(self):
        """创建模型"""
        model = LightweightUNet(
            in_channels=1,
            out_channels=1,
            base_channels=self.config['base_channels'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        return model

    def _create_dataloaders(self, cache_session: str):
        """创建数据加载器"""
        train_dataset = StreamSegmentationDataset(cache_session, split="train")
        val_dataset = StreamSegmentationDataset(cache_session, split="val")

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

        print(f"训练集: {len(train_dataset)}个样本 ({len(train_loader)}个batch)")
        print(f"验证集: {len(val_dataset)}个样本 ({len(val_loader)}个batch)")

        return train_loader, val_loader

    def train_epoch(self, model, train_loader, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="训练", leave=False):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = self.criterion(outputs, masks)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, model, val_loader):
        """验证"""
        model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证", leave=False):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # 前向传播
                outputs = model(images)
                loss = self.criterion(outputs, masks)

                # 计算指标
                outputs_np = outputs.cpu().numpy()
                masks_np = masks.cpu().numpy()

                for i in range(outputs_np.shape[0]):
                    pred = (outputs_np[i, 0] > 0.5).astype(np.float32)
                    target = (masks_np[i, 0] > 0.5).astype(np.float32)

                    if target.sum() > 0:  # 只计算有目标的样本
                        dice = self.metrics.dice_coefficient(pred, target)
                        iou = self.metrics.iou_score(pred, target)

                        total_dice += dice
                        total_iou += iou

                total_loss += loss.item()

        n_batches = len(val_loader)
        n_samples_with_target = total_dice / n_batches if total_dice > 0 else 0

        return {
            'loss': total_loss / n_batches,
            'dice': total_dice / n_batches if total_dice > 0 else 0,
            'iou': total_iou / n_batches if total_iou > 0 else 0,
            'samples_with_target': n_samples_with_target
        }

    def train(self, cache_session: str):
        """训练模型"""
        print("\n" + "=" * 60)
        print("开始 LightweightUNet 训练")
        print("=" * 60)

        # 创建模型
        model = self._create_model()
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

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
            optimizer, mode='min', factor=0.5, patience=8
        )

        # 早停机制
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 20),
            min_delta=self.config.get('early_stopping_min_delta', 0.001),
            mode='min'
        )

        # 训练循环
        best_val_loss = float('inf')
        best_model_path = None

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer)

            # 验证
            val_metrics = self.validate(model, val_loader)

            # 更新学习率
            scheduler.step(val_metrics['loss'])

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 打印统计信息
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_metrics['loss']:.4f}")
            print(f"  验证Dice: {val_metrics['dice']:.4f}")
            print(f"  验证IoU: {val_metrics['iou']:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = self.checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'config': self.config,
                    'history': self.history
                }, best_model_path)

                print(f"  ✓ 保存最佳模型到: {best_model_path}")

            # 每4个epoch保存检查点
            if (epoch + 1) % 4 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss']
                }, checkpoint_path)
                print(f"  ✓ 保存检查点到: {checkpoint_path}")

            # 早停检查
            if early_stopping(val_metrics['loss']):
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
        print(f"最终模型保存到: {final_model_path}")

        # 绘制训练曲线
        self._plot_training_history()

        # 保存训练历史
        self._save_training_history()

        return final_model_path

    def _plot_training_history(self):
        """绘制训练历史曲线"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 训练和验证损失
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('Dice损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Dice系数
        axes[0, 1].plot(epochs, self.history['val_dice'], 'g-', label='验证Dice')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice系数')
        axes[0, 1].set_title('验证Dice系数')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # IoU分数
        axes[1, 0].plot(epochs, self.history['val_iou'], 'orange', label='验证IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU分数')
        axes[1, 0].set_title('验证IoU分数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 学习率
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'm-', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('学习率')
        axes[1, 1].set_title('学习率调度')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练曲线保存到: {self.results_dir / 'training_history.png'}")

    def _save_training_history(self):
        """保存训练历史"""
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_dice': self.history['val_dice'],
            'val_iou': self.history['val_iou'],
            'learning_rate': self.history['learning_rate']
        })

        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)

        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, default=str)

        print(f"训练历史保存到: {self.output_dir / 'training_history.csv'}")


# ==============================
# 测试器
# ==============================

class SegmentationTester:
    """分割模型测试器"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        self.metrics = SegmentationMetrics()

        # 创建测试输出目录
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        self.test_output_dir = Path(config['model_save_root']) / f"test_{date_str}_{random_num}"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        self.comparison_dir = self.test_output_dir / "comparison"
        self.comparison_dir.mkdir(exist_ok=True)

    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = LightweightUNet(
            in_channels=1,
            out_channels=1,
            base_channels=self.config['base_channels'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        val_dice = checkpoint.get('val_dice', 'Unknown')

        print(f"模型加载成功")
        print(f"  训练epochs: {epoch}")
        print(f"  验证损失: {val_loss:.4f}" if isinstance(val_loss, float) else f"  验证损失: {val_loss}")
        print(f"  验证Dice: {val_dice:.4f}" if isinstance(val_dice, float) else f"  验证Dice: {val_dice}")

        return model

    def test(self, cache_session: str):
        """测试模型"""
        print("\n" + "=" * 60)
        print("开始模型测试")
        print("=" * 60)

        # 加载测试集
        test_dataset = StreamSegmentationDataset(cache_session, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('test_batch_size', 4),
            shuffle=False,
            num_workers=2
        )

        print(f"测试集: {len(test_dataset)}个样本")

        # 测试
        all_metrics = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试")):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                dicom_names = batch['dicom_name']

                # 预测
                outputs = self.model(images)

                # 转换为numpy
                images_np = images.cpu().numpy()
                masks_np = masks.cpu().numpy()
                outputs_np = outputs.cpu().numpy()

                # 对每个样本计算指标并保存对比图
                for i in range(images_np.shape[0]):
                    image = images_np[i, 0]
                    true_mask = masks_np[i, 0]
                    pred_mask = outputs_np[i, 0]

                    # 二值化
                    pred_binary = (pred_mask > 0.5).astype(np.float32)
                    true_binary = (true_mask > 0.5).astype(np.float32)

                    # 计算指标
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

                    # 保存对比图像（每5个样本保存一个）
                    if batch_idx * test_loader.batch_size + i < 20:  # 保存前20个
                        self.save_comparison_image(
                            image, true_mask, pred_mask,
                            pred_binary, true_binary,
                            dicom_names[i], metrics
                        )

        # 计算平均指标
        avg_metrics = {}
        for key in ['dice', 'iou', 'precision', 'recall', 'specificity']:
            values = [m[key] for m in all_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)

        # 保存结果
        self.save_test_results(all_metrics, avg_metrics)

        return all_metrics, avg_metrics

    def save_comparison_image(self, image, true_mask, pred_mask,
                              pred_binary, true_binary,
                              dicom_name, metrics):
        """保存对比图像"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 原始图像
            axes[0, 0].imshow(image, cmap='gray')
            axes[0, 0].set_title(f'原始DSA图像\n{dicom_name}')
            axes[0, 0].axis('off')

            # 真实mask
            axes[0, 1].imshow(true_mask, cmap='gray')
            axes[0, 1].set_title('真实mask')
            axes[0, 1].axis('off')

            # 预测mask（概率）
            axes[0, 2].imshow(pred_mask, cmap='viridis')
            axes[0, 2].set_title('预测mask (概率)')
            axes[0, 2].axis('off')

            # 二值化预测
            axes[1, 0].imshow(pred_binary, cmap='gray')
            axes[1, 0].set_title('二值化预测')
            axes[1, 0].axis('off')

            # 重叠图像
            overlay = np.zeros((*image.shape, 3), dtype=np.float32)
            overlay[..., 0] = pred_binary * 0.7  # 红色：预测
            overlay[..., 1] = true_binary * 0.7  # 绿色：真实

            # 重叠区域为黄色
            overlap = np.logical_and(pred_binary > 0.5, true_binary > 0.5)
            overlay[overlap, 0] = 1.0
            overlay[overlap, 1] = 1.0
            overlay[overlap, 2] = 0.0

            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('重叠 (红:预测, 绿:真实, 黄:重叠)')
            axes[1, 1].axis('off')

            # 指标信息
            axes[1, 2].axis('off')
            info_text = f"评估指标:\n"
            info_text += f"Dice: {metrics['dice']:.4f}\n"
            info_text += f"IoU: {metrics['iou']:.4f}\n"
            info_text += f"Precision: {metrics['precision']:.4f}\n"
            info_text += f"Recall: {metrics['recall']:.4f}\n"
            info_text += f"Specificity: {metrics['specificity']:.4f}"

            axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                            fontsize=12, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f'分割结果对比 - {dicom_name}', fontsize=14)
            plt.tight_layout()

            # 保存
            save_path = self.comparison_dir / f"{dicom_name}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"保存对比图像失败: {e}")

    def save_test_results(self, all_metrics, avg_metrics):
        """保存测试结果"""
        # 保存为CSV
        df = pd.DataFrame(all_metrics)
        df.to_csv(self.test_output_dir / 'test_results.csv', index=False)

        # 保存统计摘要
        with open(self.test_output_dir / 'test_summary.txt', 'w', encoding='utf-8') as f:
            f.write("分割模型测试结果摘要\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试样本数: {len(all_metrics)}\n\n")

            f.write("平均指标:\n")
            f.write("-" * 40 + "\n")
            for key in ['dice', 'iou', 'precision', 'recall', 'specificity']:
                f.write(f"  {key}: {avg_metrics[f'avg_{key}']:.4f} (±{avg_metrics[f'std_{key}']:.4f})\n")

            f.write("\n\n详细结果保存在: test_results.csv")

        print(f"\n测试结果保存到: {self.test_output_dir}")
        print(f"平均Dice: {avg_metrics['avg_dice']:.4f} (±{avg_metrics['std_dice']:.4f})")
        print(f"平均IoU: {avg_metrics['avg_iou']:.4f} (±{avg_metrics['std_iou']:.4f})")


# ==============================
# 主程序
# ==============================

def main():
    """主函数"""
    print("轻量级U-Net分割模型训练程序")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'dicom_dir': "D:/med_data/ai/attention_images/dicom",
        'mask_dir': "D:/med_data/ai/train22",

        # 缓存和模型保存路径
        'cache_root': "D:/med_data/ai/stream_cache_seg",
        'model_save_root': "D:/med_data/ai/pre_mask",

        # 模型参数
        'image_size': (512, 512),
        'base_channels': 32,
        'dropout_rate': 0.1,

        # 训练参数
        'batch_size': 8,
        'test_batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,

        # 早停参数
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,

        # 数据划分
        'val_ratio': 0.1,  # 7%用于验证
        'test_ratio': 0.1,  # 7%用于测试

        # 其他参数
        'num_workers': 2,
        'use_existing_cache': False  # 是否使用已有缓存
    }

    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # ========== 数据预处理 ==========
    print("\n步骤 1: 数据预处理 (流式存储)")

    preprocessor = DataPreprocessor(
        dicom_dir=config['dicom_dir'],
        mask_dir=config['mask_dir'],
        image_size=config['image_size']
    )

    if config['use_existing_cache']:
        # 使用现有缓存
        stream_manager = StreamDataManager(cache_root=config['cache_root'])
        existing_sessions = stream_manager.get_existing_sessions()

        if existing_sessions:
            print(f"找到 {len(existing_sessions)} 个现有缓存会话:")
            for i, session in enumerate(existing_sessions):
                print(f"  [{i}] {session}")

            cache_session = existing_sessions[-1]
            print(f"使用最新会话: {cache_session}")
        else:
            print("未找到现有缓存会话，将创建新数据")
            cache_session = preprocessor.process_and_save(
                val_ratio=config['val_ratio'],
                test_ratio=config['test_ratio'],
                force_new=True
            )
    else:
        # 新建数据
        cache_session = preprocessor.process_and_save(
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            force_new=True
        )

    print(f"缓存会话路径: {cache_session}")

    # ========== 训练模型 ==========
    print("\n步骤 2: 训练 LightweightUNet 模型")
    trainer = SegmentationTrainer(config)
    trained_model_path = trainer.train(cache_session)

    # ========== 测试模型 ==========
    print("\n" + "=" * 60)
    print("步骤 3: 测试训练好的模型")

    if isinstance(trained_model_path, Path):
        trained_model_path = str(trained_model_path)

    tester = SegmentationTester(trained_model_path, config)
    test_results, avg_metrics = tester.test(cache_session)

    print("\n" + "=" * 60)
    print("程序完成!")
    print(f"训练输出目录: {trainer.output_dir}")
    print(f"测试输出目录: {tester.test_output_dir}")

    print(f"\n最终测试结果:")
    print(f"  平均Dice: {avg_metrics['avg_dice']:.4f} (±{avg_metrics['std_dice']:.4f})")
    print(f"  平均IoU: {avg_metrics['avg_iou']:.4f} (±{avg_metrics['std_iou']:.4f})")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()