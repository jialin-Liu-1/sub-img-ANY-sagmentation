"""
子图图卷积模型训练器 - 完整版
"""

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

# 导入模型和损失函数
from GCN_subimg_loc_only import SubImageGraphFusionModel, SubImageGraphLoss

warnings.filterwarnings('ignore')


# ==============================
# 1. 通用工具函数
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

        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            image = np.zeros_like(image)

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
    match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
    if match:
        return match.group(0)
    match = re.search(r'(\d+)', basename)
    if match:
        return match.group(1)
    return None


def save_training_history(history: Dict[str, List[float]], output_dir: Path, config: Dict[str, Any]):
    """保存训练历史"""
    max_len = max(len(history.get('train_loss', [])),
                  len(history.get('val_loss', [])),
                  len(history.get('learning_rate', [])))

    history_df = pd.DataFrame({
        'epoch': range(1, max_len + 1),
        'train_loss': history.get('train_loss', []) + [None] * (max_len - len(history.get('train_loss', []))),
        'val_loss': history.get('val_loss', []) + [None] * (max_len - len(history.get('val_loss', []))),
        'train_mse_loss': history.get('train_mse_loss', []) + [None] * (max_len - len(history.get('train_mse_loss', []))),
        'val_mse_loss': history.get('val_mse_loss', []) + [None] * (max_len - len(history.get('val_mse_loss', []))),
        'train_l1_loss': history.get('train_l1_loss', []) + [None] * (max_len - len(history.get('train_l1_loss', []))),
        'val_l1_loss': history.get('val_l1_loss', []) + [None] * (max_len - len(history.get('val_l1_loss', []))),
        'train_contrastive_loss': history.get('train_contrastive_loss', []) + [None] * (max_len - len(history.get('train_contrastive_loss', []))),
        'learning_rate': history.get('learning_rate', []) + [None] * (max_len - len(history.get('learning_rate', [])))
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4, default=str)

def plot_training_curves(history: Dict[str, List[float]], save_dir: Path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MSE Loss
    axes[0, 1].plot(epochs, history['train_mse_loss'], 'b-', label='Train MSE')
    axes[0, 1].plot(epochs, history['val_mse_loss'], 'r-', label='Val MSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # L1 Loss
    axes[1, 0].plot(epochs, history['train_l1_loss'], 'b-', label='Train L1')
    axes[1, 0].plot(epochs, history['val_l1_loss'], 'r-', label='Val L1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('L1 Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'g-', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_plot(image, pred_position, gt_position, filename, save_path, attention_map=None):
    """创建对比图 - DSA图像+位置标记 和 注意力叠加图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    h, w = image.shape
    pred_x, pred_y = int(pred_position[0] * w), int(pred_position[1] * h)
    gt_x, gt_y = int(gt_position[0] * w), int(gt_position[1] * h)

    position_error = np.sqrt((pred_position[0] - gt_position[0]) ** 2 + (pred_position[1] - gt_position[1]) ** 2)

    # 图1: DSA原图 + 位置标记
    axes[0].imshow(image, cmap='gray')
    axes[0].scatter(pred_x, pred_y, c='red', s=100, marker='+', linewidths=2, label='Predicted')
    axes[0].scatter(gt_x, gt_y, c='green', s=100, marker='+', linewidths=2, label='Ground Truth')
    axes[0].set_title(
        f'DSA Image\nPred: ({pred_position[0]:.3f}, {pred_position[1]:.3f}) | GT: ({gt_position[0]:.3f}, {gt_position[1]:.3f})\nError: {position_error:.4f}')
    axes[0].legend()
    axes[0].axis('off')

    # 图2: 注意力叠加图
    if attention_map is not None:
        axes[1].imshow(image, cmap='gray')
        im = axes[1].imshow(attention_map, cmap='hot', alpha=0.5)
        axes[1].set_title('Attention Map Overlay')
        plt.colorbar(im, ax=axes[1])
    else:
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title('Attention Map (Not Available)')
    axes[1].axis('off')

    plt.suptitle(f'Test Result: {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_focus_image(image, position, max_threshold_ratio):
    """
    创建聚焦图：以预测位置为圆心，最大阈值半径画圆

    Args:
        image: [H, W] 原始DSA图像
        position: (x_ratio, y_ratio) 预测位置
        max_threshold_ratio: 最大阈值半径（相对于图像尺寸的比例）

    Returns:
        focus_image: 聚焦图（圆内为原图像素，圆外为0）
    """
    h, w = image.shape
    center_x = int(position[0] * w)
    center_y = int(position[1] * h)

    # 计算半径（取图像对角线的一部分）
    radius = int(max_threshold_ratio * np.sqrt(h ** 2 + w ** 2))

    # 创建圆形mask
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist_from_center <= radius

    # 应用mask
    focus_image = image * mask

    return focus_image


def compute_position_metrics(pred_position, gt_position):
    """计算位置评估指标"""
    pos_error = np.sqrt((pred_position[0] - gt_position[0]) ** 2 + (pred_position[1] - gt_position[1]) ** 2)
    return {'position_error': pos_error}


def compute_size_metrics(pred_size, gt_size):
    """计算尺寸评估指标"""
    size_error = abs(pred_size - gt_size)
    relative_size_error = size_error / (gt_size + 1e-8)
    return {'size_error': size_error, 'relative_size_error': relative_size_error}


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
# 3. 流式数据管理器
# ==============================

class StreamDataManager:
    """流式数据管理器"""
    def __init__(self, cache_root: str = "D:/med_data/ai/stream_cache"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def create_cache_session(self, prefix: str = "subimg") -> str:
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        session_name = f"{prefix}_{date_str}_{random_num}"
        session_path = self.cache_root / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / "train").mkdir(exist_ok=True)
        (session_path / "test").mkdir(exist_ok=True)
        print(f"创建缓存会话: {session_path}")
        return str(session_path)

    def get_existing_sessions(self) -> List[str]:
        sessions = []
        for item in self.cache_root.iterdir():
            if item.is_dir() and re.match(r'subimg_\d{8}_\d{3}', item.name):
                sessions.append(str(item))
        return sorted(sessions)

    def save_sample(self, data: Dict, save_path: str):
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
        np.savez(save_path, **save_dict)

    def load_sample(self, npz_path: str) -> Dict:
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


# ==============================
# 4. 数据预处理器
# ==============================

class SubImageDataPreprocessor:
    """子图数据预处理器"""
    def __init__(self, image_dir: str, location_excel_path: str, segment_excel_path: str,
                 mask_dir: str = None, image_size: Tuple[int, int] = (512, 512)):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.location_excel_path = location_excel_path
        self.segment_excel_path = segment_excel_path
        self.stream_manager = StreamDataManager()

    def _create_location_loader(self):
        class LocationLoader:
            def __init__(self, path):
                self.location_dict = {}
                self.case_to_location = {}
                self._load(path)

            def _load(self, path):
                df = pd.read_excel(path)
                print(f"位置信息Excel列名: {df.columns.tolist()}")
                for _, row in df.iterrows():
                    filename = str(row.iloc[0]).strip()
                    basename = os.path.splitext(filename)[0]
                    try:
                        x_ratio = float(row.iloc[1]) if len(row) > 1 else 0.5
                        y_ratio = float(row.iloc[2]) if len(row) > 2 else 0.5
                    except Exception as e:
                        print(f"解析行数据出错: {e}")
                        x_ratio, y_ratio = 0.5, 0.5

                    x_ratio = max(0.0, min(1.0, x_ratio))
                    y_ratio = max(0.0, min(1.0, y_ratio))

                    self.location_dict[basename] = {
                        'x_ratio': x_ratio,
                        'y_ratio': y_ratio,
                    }
                    case_id = extract_case_id(filename)
                    if case_id:
                        self.case_to_location[case_id] = self.location_dict[basename]

            def get(self, filename):
                basename = os.path.splitext(filename)[0]
                if basename in self.location_dict:
                    return self.location_dict[basename]
                case_id = extract_case_id(filename)
                if case_id and case_id in self.case_to_location:
                    return self.case_to_location[case_id]
                return {
                    'x_ratio': 0.5,
                    'y_ratio': 0.5,
                }

        return LocationLoader(self.location_excel_path)

    def _create_segment_loader(self):
        class SegmentLoader:
            valid_classes = [1, 2, 4, 5, 6, 7]

            def __init__(self, path):
                self.segment_dict = {}
                self.case_to_segment = {}
                self._load(path)

            def _load(self, path):
                df = pd.read_excel(path)
                print(f"血管段信息Excel列名: {df.columns.tolist()}")
                for _, row in df.iterrows():
                    filename = str(row.iloc[0]).strip()
                    basename = os.path.splitext(filename)[0]
                    try:
                        original_class = int(row.iloc[1]) if not pd.isna(row.iloc[1]) else None
                        if original_class and original_class in self.valid_classes:
                            new_index = self.valid_classes.index(original_class)
                            self.segment_dict[basename] = new_index
                            case_id = extract_case_id(filename)
                            if case_id:
                                self.case_to_segment[case_id] = new_index
                    except:
                        continue

            def get(self, filename):
                basename = os.path.splitext(filename)[0]
                if basename in self.segment_dict:
                    idx = self.segment_dict[basename]
                else:
                    case_id = extract_case_id(filename)
                    if case_id and case_id in self.case_to_segment:
                        idx = self.case_to_segment[case_id]
                    else:
                        idx = 2
                onehot = np.zeros(6, dtype=np.float32)
                onehot[idx] = 1.0
                return onehot, idx

        return SegmentLoader(self.segment_excel_path)

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        return load_image(file_path, self.image_size)

    def process_and_save(self, output_session: str = None, train_ratio: float = 0.8,
                         max_samples: int = None, force_new: bool = True) -> str:
        if output_session is None:
            output_session = self.stream_manager.create_cache_session("subimg")
        else:
            output_path = Path(output_session)
            if output_path.exists() and not force_new:
                print(f"使用现有会话: {output_session}")
                return output_session
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "train").mkdir(exist_ok=True)
                (output_path / "test").mkdir(exist_ok=True)

        output_path = Path(output_session)
        train_dir = output_path / "train"
        test_dir = output_path / "test"

        print(f"\n开始处理数据，保存到: {output_session}")
        print("=" * 60)

        location_loader = self._create_location_loader()
        segment_loader = self._create_segment_loader()

        image_files = []
        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg', '']:
                    image_files.append(file_path)

        if max_samples:
            image_files = image_files[:max_samples]

        print(f"找到 {len(image_files)} 个图像文件")

        stats = {'total': 0, 'success': 0, 'failed': 0, 'segment_dist': {}}
        processed_samples = []

        for file_path in tqdm(image_files, desc="处理图像"):
            filename = file_path.name
            stats['total'] += 1

            try:
                image = self._load_image(file_path)
                if image is None:
                    stats['failed'] += 1
                    continue

                image = np.expand_dims(image, axis=0)
                location = location_loader.get(filename)
                segment_onehot, segment_idx = segment_loader.get(filename)

                stats['success'] += 1
                stats['segment_dist'][segment_idx] = stats['segment_dist'].get(segment_idx, 0) + 1

                # 在 process_and_save 方法中，修改 sample_data 的创建部分
                sample_data = {
                    'image': image,
                    'segment_onehot': segment_onehot,
                    'segment_idx': np.array([segment_idx]),
                    'x_ratio': np.array([location['x_ratio']]),
                    'y_ratio': np.array([location['y_ratio']]),
                    'filename': np.array([filename])
                }
                processed_samples.append((file_path.stem, sample_data))

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                stats['failed'] += 1

        random.shuffle(processed_samples)
        split_idx = int(len(processed_samples) * train_ratio)
        train_samples = processed_samples[:split_idx]
        test_samples = processed_samples[split_idx:]

        print(f"\n保存训练集 ({len(train_samples)}个样本)")
        for stem, sample_data in tqdm(train_samples, desc="保存训练集"):
            save_path = train_dir / f"{stem}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path))

        print(f"保存测试集 ({len(test_samples)}个样本)")
        for stem, sample_data in tqdm(test_samples, desc="保存测试集"):
            save_path = test_dir / f"{stem}.npz"
            self.stream_manager.save_sample(sample_data, str(save_path))

        metadata = {
            'total_files': stats['total'],
            'successful': stats['success'],
            'failed': stats['failed'],
            'train_samples': len(train_samples),
            'test_samples': len(test_samples),
            'train_ratio': train_ratio,
            'segment_distribution': stats['segment_dist'],
            'image_size': self.image_size,
            'created_at': datetime.now().isoformat()
        }
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"成功处理: {stats['success']}/{stats['total']}")
        print(f"训练集: {len(train_samples)}个样本")
        print(f"测试集: {len(test_samples)}个样本")

        return output_session


# ==============================
# 5. 流式数据集类
# ==============================

class StreamSubImageDataset(Dataset):
    """流式子图数据集"""
    def __init__(self, data_dir: str, split: str = "train", augment: bool = False):
        self.data_dir = Path(data_dir) / split
        self.augment = augment
        self.sample_files = list(self.data_dir.glob("*.npz"))
        self.sample_files.sort()
        print(f"加载{split}数据集: {len(self.sample_files)}个样本")

    def __len__(self):
        return len(self.sample_files)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
        return image

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        data = np.load(sample_path, allow_pickle=True)

        image = torch.from_numpy(data['image']).float()
        segment_onehot = torch.from_numpy(data['segment_onehot']).float()
        segment_idx = torch.tensor(data['segment_idx'].item(), dtype=torch.long)
        x_ratio = torch.tensor(data['x_ratio'].item(), dtype=torch.float32)
        y_ratio = torch.tensor(data['y_ratio'].item(), dtype=torch.float32)

        filename = str(data['filename'].item()) if data['filename'].ndim == 0 else str(data['filename'][0])

        if self.augment:
            image_np = image.numpy()
            image_aug = self._augment_image(image_np[0])
            image = torch.from_numpy(image_aug).unsqueeze(0).float()

        return {
            'image': image,
            'segment_onehot': segment_onehot,
            'segment_idx': segment_idx,
            'x_ratio': x_ratio,
            'y_ratio': y_ratio,
            'filename': filename
        }


# ==============================
# 6. 训练器
# ==============================
class ContrastiveAugmentation:
    """对比学习数据增强 - 只改变整体，不改变血管结构，同时记录增强方法"""

    def __init__(self, device='cuda'):
        self.device = device
        # 可用的增强方法列表
        self.aug_methods = ['contrast', 'brightness', 'noise', 'blur', 'gamma', 'none']

    def __call__(self, image: torch.Tensor, num_aug: int = 2) -> Tuple[torch.Tensor, Union[str, List[str]]]:
        """
        应用随机增强，可指定增强个数

        Args:
            image: [B, C, H, W] 或 [C, H, W]
            num_aug: 增强方法的数量（从可用方法中随机选择num_aug个）

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

            # 随机选择num_aug个增强类型（不重复）
            if num_aug >= len(self.aug_methods):
                selected_methods = self.aug_methods.copy()
            else:
                selected_methods = random.sample(self.aug_methods, num_aug)

            methods.append('+'.join(selected_methods))

            img_aug = img.clone()

            # 按顺序应用选中的增强方法
            for aug_type in selected_methods:
                if aug_type == 'contrast':
                    factor = random.uniform(0.7, 1.3)
                    mean = img_aug.mean()
                    img_aug = mean + factor * (img_aug - mean)

                elif aug_type == 'brightness':
                    delta = random.uniform(-0.15, 0.15)
                    img_aug = img_aug + delta

                elif aug_type == 'noise':
                    noise_std = random.uniform(0, 0.03)
                    noise = torch.randn_like(img_aug) * noise_std
                    img_aug = img_aug + noise

                elif aug_type == 'blur':
                    kernel_size = random.choice([3, 5])
                    padding = kernel_size // 2
                    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img_aug.device) / (
                                kernel_size * kernel_size)
                    img_aug = F.conv2d(img_aug.unsqueeze(0), kernel, padding=padding).squeeze(0)

                elif aug_type == 'gamma':
                    gamma = random.uniform(0.7, 1.3)
                    img_aug = torch.pow(torch.clamp(img_aug, 0, 1), gamma)

                elif aug_type == 'none':
                    pass

                img_aug = torch.clamp(img_aug, 0, 1)

            augmented.append(img_aug)

        result = torch.stack(augmented)

        if single_image:
            return result.squeeze(0), methods[0]
        return result, methods


class SubImageGraphTrainer:
    """子图图卷积模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.output_dir = setup_output_directory(
            config.get('model_save_root', "D:/med_data/ai/subimg_models"),
            prefix="subimg_graph"
        )

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.final_model_dir = self.output_dir / "final_models"
        self.test_dir = self.output_dir / "test"

        for dir_path in [self.checkpoint_dir, self.plots_dir, self.final_model_dir, self.test_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"Model save directory: {self.output_dir}")

        # 历史记录字典
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse_loss': [],
            'val_mse_loss': [],
            'train_l1_loss': [],
            'val_l1_loss': [],
            'train_position_loss': [],
            'val_position_loss': [],
            'train_contrastive_loss': [],
            'learning_rate': []
        }

        # 损失函数 - 支持独立MSE和L1权重
        self.criterion = SubImageGraphLoss(
            mse_weight=config.get('mse_weight', 1.0),
            l1_weight=config.get('l1_weight', 1.0)
        )

        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )

        # 初始化对比增强器
        self.augmentor = ContrastiveAugmentation(device=self.device)
        self.num_augmentations = config.get('num_augmentations', 2)
        self.contrastive_weight = config.get('contrastive_weight', 0.1)

    def _create_model(self):
        """创建模型"""
        model = SubImageGraphFusionModel(
            image_size=self.config.get('image_size', 512),
            patch_size=self.config.get('patch_size', 16),
            subimg_feature_dim=self.config.get('subimg_feature_dim', 128),
            gcn_hidden_dim=self.config.get('gcn_hidden_dim', 256),
            gcn_out_dim=self.config.get('gcn_out_dim', 256),
            num_segments=self.config.get('num_segments', 6),
            dilations=self.config.get('dilations', [1, 2, 4, 6]),
            num_dilated_blocks=self.config.get('num_dilated_blocks', 2),
            regressor_hidden_dim=self.config.get('regressor_hidden_dim', 256)
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数总量: {total_params / 1e6:.2f}M")

        return model

    def _create_dataloaders(self, cache_session: str):
        """创建数据加载器"""
        train_dataset = StreamSubImageDataset(data_dir=cache_session, split="train", augment=False)
        val_dataset = StreamSubImageDataset(data_dir=cache_session, split="test", augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'],
            shuffle=True, num_workers=self.config.get('num_workers', 2),
            pin_memory=True, drop_last=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'],
            shuffle=False, num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        print(f"训练集: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"验证集: {len(val_dataset)} samples ({len(val_loader)} batches)")

        return train_loader, val_loader

    def _build_targets(self, batch):
        """构建训练目标"""
        return {
            'position': torch.stack([batch['x_ratio'], batch['y_ratio']], dim=1).to(self.device),
            'has_aneurysm': torch.ones(batch['image'].shape[0], 1, device=self.device)
        }

    def _compute_contrastive_loss(self, outputs_original, outputs_augmented):
        """计算对比损失 - 原始输出与增强输出的差异"""
        position_diff = F.mse_loss(outputs_original['position'], outputs_augmented['position'])
        return position_diff

    def train_epoch(self, model, train_loader, optimizer, epoch):
        """训练一个epoch - 带对比增强"""
        model.train()

        total_loss = 0
        total_mse_loss = 0
        total_l1_loss = 0
        total_contrastive_loss = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Train]')

        for batch_idx, batch in enumerate(train_bar):
            images = batch['image'].to(self.device)
            segment_onehot = batch['segment_onehot'].to(self.device)
            targets = self._build_targets(batch)

            batch_loss = 0
            first_contrastive_loss_val = 0
            total_contrastive_loss_batch = 0

            loss_original_val = 0
            mse_loss_val = 0
            l1_loss_val = 0

            # 梯度累积：对每个增强版本分别计算损失并累积梯度
            for aug_idx in range(self.num_augmentations + 1):
                if aug_idx == 0:
                    images_aug = images
                else:
                    images_aug, aug_methods = self.augmentor(images, num_aug=self.num_augmentations)

                # 前向传播 - 训练时不生成注意力图
                outputs = model(images_aug, segment_onehot, return_attention_map=False)
                loss_task, loss_dict = self.criterion(outputs, targets)

                if aug_idx == 0:
                    loss_original_val = loss_task.item()
                    mse_loss_val = loss_dict.get('mse_loss', 0)
                    l1_loss_val = loss_dict.get('l1_loss', 0)
                    batch_loss = loss_task
                else:
                    with torch.no_grad():
                        outputs_original = model(images, segment_onehot, return_attention_map=False)

                    contrastive_loss = self._compute_contrastive_loss(outputs_original, outputs)

                    if aug_idx == 1:
                        first_contrastive_loss_val = contrastive_loss.item()

                    total_contrastive_loss_batch += contrastive_loss.item()
                    loss_aug_total = loss_task + self.contrastive_weight * contrastive_loss
                    batch_loss = batch_loss + loss_aug_total

                if aug_idx == 0:
                    loss_task.backward()
                else:
                    loss_aug_total.backward()

            batch_loss = batch_loss / (self.num_augmentations + 1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += batch_loss.item()
            total_mse_loss += mse_loss_val
            total_l1_loss += l1_loss_val
            total_contrastive_loss += total_contrastive_loss_batch / max(1, self.num_augmentations)

            # 更新进度条显示
            train_bar.set_postfix({
                'loss': f'{loss_original_val:.4f}',
                'mse': f'{mse_loss_val:.4f}',
                'l1': f'{l1_loss_val:.4f}',
                'contrast': f'{first_contrastive_loss_val:.4f}'
            })

        num_batches = len(train_loader)
        return {
            'loss': total_loss / num_batches,
            'mse_loss': total_mse_loss / num_batches,
            'l1_loss': total_l1_loss / num_batches,
            'position_loss': (total_mse_loss + total_l1_loss) / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches
        }

    def validate_epoch(self, model, val_loader):
        """验证一个epoch"""
        model.eval()

        total_loss = 0
        total_mse_loss = 0
        total_l1_loss = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='[Validation]')
            for batch in val_bar:
                images = batch['image'].to(self.device)
                segment_onehot = batch['segment_onehot'].to(self.device)
                targets = self._build_targets(batch)

                outputs = model(images, segment_onehot, return_attention_map=False)
                loss, loss_dict = self.criterion(outputs, targets)

                total_loss += loss.item()
                total_mse_loss += loss_dict.get('mse_loss', 0)
                total_l1_loss += loss_dict.get('l1_loss', 0)

                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{loss_dict.get("mse_loss", 0):.4f}',
                    'l1': f'{loss_dict.get("l1_loss", 0):.4f}'
                })

        num_batches = len(val_loader)
        return {
            'loss': total_loss / num_batches,
            'mse_loss': total_mse_loss / num_batches,
            'l1_loss': total_l1_loss / num_batches,
            'position_loss': (total_mse_loss + total_l1_loss) / num_batches
        }

    def test_and_visualize(self, model, cache_session):
        """测试并生成可视化结果（批处理版本）"""
        print("\n" + "=" * 60)
        print("开始测试模型...")
        print("=" * 60)

        test_dataset = StreamSubImageDataset(data_dir=cache_session, split="test", augment=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                 num_workers=self.config.get('num_workers', 2), pin_memory=True)

        model.eval()

        all_results = []
        comparison_dir = self.test_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)

        position_threshold = 0.01
        correct_predictions = 0
        total_predictions = 0
        x_errors = []
        y_errors = []

        # 损失统计
        total_mse_loss = 0
        total_l1_loss = 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Testing')
            for batch_idx, batch in enumerate(test_bar):
                images = batch['image'].to(self.device)
                segment_onehot = batch['segment_onehot'].to(self.device)
                targets = self._build_targets(batch)

                # 测试时生成注意力图
                outputs = model(images, segment_onehot, return_attention_map=True)

                # 计算损失（整个batch的平均损失）
                _, loss_dict = self.criterion(outputs, targets)
                batch_mse_loss = loss_dict.get('mse_loss', 0)
                batch_l1_loss = loss_dict.get('l1_loss', 0)

                # 累积损失（按样本数加权）
                batch_size = len(batch['filename'])
                total_mse_loss += batch_mse_loss * batch_size
                total_l1_loss += batch_l1_loss * batch_size

                # 批量处理每个样本
                pred_positions = outputs['position'].cpu().numpy()  # [batch_size, 2]
                gt_x_ratios = batch['x_ratio'].cpu().numpy()  # [batch_size]
                gt_y_ratios = batch['y_ratio'].cpu().numpy()  # [batch_size]
                filenames = batch['filename']
                images_np = batch['image'].cpu().numpy()  # [batch_size, 1, H, W]

                attention_maps = outputs.get('attention_map', None)
                if attention_maps is not None:
                    attention_maps_np = attention_maps.cpu().numpy()  # [batch_size, 1, H, W]

                # 遍历batch中的每个样本
                for i in range(batch_size):
                    pred_position = pred_positions[i]
                    gt_position = np.array([gt_x_ratios[i], gt_y_ratios[i]])
                    filename = filenames[i]
                    image_np = images_np[i, 0]  # 移除通道维度

                    # 获取对应的注意力图
                    att_map_np = None
                    if attention_maps is not None:
                        att_map_np = attention_maps_np[i, 0]

                    # 为每个样本生成可视化（可选：控制生成频率避免过多文件）
                    # 这里保持原有行为，为每个样本都生成可视化
                    create_comparison_plot(
                        image_np, pred_position, gt_position,
                        filename, comparison_dir / f"{filename}_comparison.png",
                        att_map_np
                    )

                    # 计算误差
                    position_error = np.sqrt(
                        (pred_position[0] - gt_position[0]) ** 2 +
                        (pred_position[1] - gt_position[1]) ** 2
                    )
                    is_correct = position_error < position_threshold
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1

                    x_error = abs(pred_position[0] - gt_position[0])
                    y_error = abs(pred_position[1] - gt_position[1])
                    x_errors.append(x_error)
                    y_errors.append(y_error)

                    all_results.append({
                        'filename': filename,
                        'x_pred': pred_position[0],
                        'x_true': gt_position[0],
                        'y_pred': pred_position[1],
                        'y_true': gt_position[1],
                        'position_error': position_error,
                        'x_error': x_error,
                        'y_error': y_error,
                        'is_correct': is_correct
                    })

                # 更新进度条显示
                test_bar.set_postfix({
                    'Acc': f'{correct_predictions / total_predictions:.3f}',
                    'MSE': f'{batch_mse_loss:.6f}'
                })

        # 计算平均损失（除以总样本数）
        total_samples = len(test_dataset)
        avg_mse_loss = total_mse_loss / total_samples if total_samples > 0 else 0
        avg_l1_loss = total_l1_loss / total_samples if total_samples > 0 else 0

        # 保存结果DataFrame
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(self.test_dir / "test_results.csv", index=False)

        # 计算统计指标
        position_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        mean_x_error = np.mean(x_errors)
        std_x_error = np.std(x_errors)
        mean_y_error = np.mean(y_errors)
        std_y_error = np.std(y_errors)

        # 生成测试报告
        report_path = self.test_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("子图图卷积模型测试报告（批处理版本）\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"测试样本数: {len(all_results)}\n")
            f.write(f"批处理大小: 8\n\n")
            f.write("定位性能指标\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  平均位置误差: {df_results['position_error'].mean():.4f} ± {df_results['position_error'].std():.4f}\n")
            f.write(
                f"  位置正确率 (阈值={position_threshold}): {position_accuracy:.2%} ({correct_predictions}/{total_predictions})\n")
            f.write(f"  X轴平均误差: {mean_x_error:.4f} ± {std_x_error:.4f}\n")
            f.write(f"  Y轴平均误差: {mean_y_error:.4f} ± {std_y_error:.4f}\n\n")
            f.write("损失统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"  平均MSE损失: {avg_mse_loss:.6f}\n")
            f.write(f"  平均L1损失: {avg_l1_loss:.6f}\n")

            # 添加性能对比信息
            f.write("\n性能说明\n")
            f.write("-" * 40 + "\n")
            f.write(f"  使用批处理测试（batch_size=8），GPU利用率更高\n")
            f.write(f"  总批次数: {len(test_loader)}\n")

        print(f"\n测试完成！")
        print(f"  总测试样本数: {len(all_results)}")
        print(f"  批处理大小: 8")
        print(f"  总批次数: {len(test_loader)}")
        print(f"  平均位置误差: {df_results['position_error'].mean():.4f}")
        print(f"  位置正确率: {position_accuracy:.2%} ({correct_predictions}/{total_predictions})")
        print(f"  X轴平均误差: {mean_x_error:.4f}")
        print(f"  Y轴平均误差: {mean_y_error:.4f}")
        print(f"  平均MSE损失: {avg_mse_loss:.6f}")
        print(f"  平均L1损失: {avg_l1_loss:.6f}")
        print(f"结果保存至: {self.test_dir}")

        return df_results

    def train(self, cache_session: str):
        """主训练函数"""
        print("\n" + "=" * 60)
        print("开始子图图卷积模型训练")
        print("=" * 60)

        model = self._create_model()
        train_loader, val_loader = self._create_dataloaders(cache_session)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        best_val_loss = float('inf')
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        best_model_state = None

        for epoch in range(self.config['num_epochs']):
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            val_metrics = self.validate_epoch(model, val_loader)

            scheduler.step(val_metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse_loss'].append(train_metrics['mse_loss'])
            self.history['val_mse_loss'].append(val_metrics['mse_loss'])
            self.history['train_l1_loss'].append(train_metrics['l1_loss'])
            self.history['val_l1_loss'].append(val_metrics['l1_loss'])
            self.history['train_position_loss'].append(train_metrics['position_loss'])
            self.history['val_position_loss'].append(val_metrics['position_loss'])
            self.history['train_contrastive_loss'].append(train_metrics.get('contrastive_loss', 0))
            self.history['learning_rate'].append(current_lr)

            # 打印信息
            print(f"\n  Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(
                f"  Train - Loss: {train_metrics['loss']:.6f} | MSE: {train_metrics['mse_loss']:.6f} | L1: {train_metrics['l1_loss']:.6f}")
            if 'contrastive_loss' in train_metrics:
                print(f"         Contrastive: {train_metrics['contrastive_loss']:.6f}")
            print(
                f"  Val   - Loss: {val_metrics['loss']:.6f} | MSE: {val_metrics['mse_loss']:.6f} | L1: {val_metrics['l1_loss']:.6f}")
            print(f"  LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': self.config,
                }, best_model_path)
                print(f"  ✓ 更新最佳模型 (Val Loss: {best_val_loss:.6f})")

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
                print(f"  ✓ 保存检查点")

            if self.early_stopping(val_metrics['loss']):
                print(f"\n🚨 早停触发! 最佳验证损失: {best_val_loss:.6f}")
                break

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        # 保存最终模型
        final_model_path = self.final_model_dir / 'subimg_graph_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': best_val_loss
        }, final_model_path)

        model_cpu = model.to('cpu')
        complete_model_path = self.final_model_dir / 'subimg_graph_model_complete.pth'
        torch.save({
            'model': model_cpu,
            'config': self.config,
            'history': self.history,
            'best_val_loss': best_val_loss
        }, complete_model_path)
        model.to(self.device)

        print(f"\n训练完成!")
        print(f"最佳验证损失: {best_val_loss:.6f}")

        # 绘制训练曲线
        plot_training_curves(self.history, self.plots_dir)
        save_training_history(self.history, self.output_dir, self.config)

        # 自动测试
        self.test_and_visualize(model, cache_session)

        return final_model_path


# ==============================
# 7. 主函数
# ==============================

def main_auto():
    """主函数 - 全自动模式"""
    print("=" * 60)
    print("子图图卷积模型训练程序（全自动模式）")
    print("=" * 60)

    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/translate/reverse/train",
        'location_excel_path': "D:/med_data/ai/translate/all_mask/location_all.xlsx",
        'segment_excel_path': "D:/med_data/ai/translate/reverse/classify_all_trans_updated.xlsx",
        'mask_dir': None,

        # 模型参数
        'image_size': 512,
        'patch_size': 16,
        'subimg_feature_dim': 144,
        'gcn_hidden_dim': 256,
        'gcn_out_dim': 256,
        'num_segments': 6,
        'dilations': [1, 2, 4, 6],
        'num_dilated_blocks': 2,
        'regressor_hidden_dim': 256,

        # 损失参数
        'mse_weight': 15.0,
        'l1_weight': 3.0,
        'num_augmentations': 3,
        'contrastive_weight': 3.0,

        # 训练参数
        'batch_size': 8,
        'num_epochs': 150,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'train_ratio': 0.8,

        # 早停参数
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0003,

        # 其他
        'num_workers': 3,
        'cache_root': "D:/med_data/ai/stream_cache",
        'model_save_root': "D:/med_data/ai/subimg_models",
    }

    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("步骤 1/4: 创建新的缓存数据")
    print("=" * 60)

    preprocessor = SubImageDataPreprocessor(
        image_dir=config['train_image_dir'],
        location_excel_path=config['location_excel_path'],
        segment_excel_path=config['segment_excel_path'],
        mask_dir=config.get('mask_dir'),
        image_size=(config['image_size'], config['image_size'])
    )

    cache_session = preprocessor.process_and_save(train_ratio=config['train_ratio'], force_new=True)
    print(f"缓存会话创建完成: {cache_session}")

    print("\n" + "=" * 60)
    print("步骤 2/4: 开始训练模型")
    print("=" * 60)

    trainer = SubImageGraphTrainer(config)
    model_path = trainer.train(cache_session)

    print("\n" + "=" * 60)
    print("步骤 3/4: 训练完成，模型已保存")
    print(f"模型路径: {model_path}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("步骤 4/4: 测试已完成，结果已保存")
    print(f"测试结果保存在: {trainer.test_dir}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"训练输出目录: {trainer.output_dir}")
    print(f"测试结果目录: {trainer.test_dir}")
    print("=" * 60)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main_auto()