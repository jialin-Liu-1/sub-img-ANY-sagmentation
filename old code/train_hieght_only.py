"""
高度预测器独立训练程序
只训练高度路径，不包含宽度预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List, Any
import warnings
import json
from datetime import datetime
import re
import random
import pickle

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
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'learning_rate': history['learning_rate']
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4, default=str)


def plot_training_curves(history: Dict[str, List[float]], save_dir: Path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning Rate曲线
    axes[1].plot(epochs, history['learning_rate'], 'g-', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

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
# 3. 位置信息加载器
# ==============================

class PositionInfoLoader:
    """加载动脉瘤类别信息（6类）"""

    def __init__(self, excel_path: str = None):
        self.position_dict = {}
        self.case_to_position = {}
        self.class_names = {
            0: "Segment 1 (原1类)",
            1: "Segment 2 (原2类)",
            2: "Segment 4 (原4类)",
            3: "Segment 5 (原5类)",
            4: "Segment 6 (原6类)",
            5: "Segment 7 (原7类)"
        }
        self.valid_classes = [1, 2, 4, 5, 6, 7]

        if excel_path:
            self._load_position_info(excel_path)

    def _load_position_info(self, excel_path: str):
        """加载类别信息"""
        try:
            df = pd.read_excel(excel_path)
            print(f"类别信息Excel列名: {df.columns.tolist()}")

            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()
                base_name = os.path.splitext(filename)[0]

                try:
                    original_class = int(row.iloc[1]) if not pd.isna(row.iloc[1]) else None
                    if original_class is None or original_class not in self.valid_classes:
                        continue

                    new_index = self.valid_classes.index(original_class)
                    self.position_dict[base_name] = {
                        'original_class': original_class,
                        'new_index': new_index
                    }

                    case_id = extract_case_id(filename)
                    if case_id:
                        self.case_to_position[case_id] = {
                            'original_class': original_class,
                            'new_index': new_index
                        }
                except Exception as e:
                    continue

            print(f"成功加载 {len(self.position_dict)} 个文件的类别信息")
        except Exception as e:
            print(f"加载类别信息失败: {e}")

    def get_position_for_image(self, filename: str):
        """获取图像对应的类别信息"""
        basename = os.path.splitext(filename)[0]

        if basename in self.position_dict:
            pos_info = self.position_dict[basename]
            position_tensor = torch.zeros(6)
            position_tensor[pos_info['new_index']] = 1.0
            return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        case_id = extract_case_id(filename)
        if case_id and case_id in self.case_to_position:
            pos_info = self.case_to_position[case_id]
            position_tensor = torch.zeros(6)
            position_tensor[pos_info['new_index']] = 1.0
            return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 默认返回类别2（Segment 4）
        position_tensor = torch.zeros(6)
        position_tensor[2] = 1.0
        return position_tensor, 2, "Segment 4 (default)"


# ==============================
# 4. 高度信息加载器
# ==============================

class HeightInfoLoader:
    """加载高度比例信息"""

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}
        self.case_to_location = {}
        self._load_location_info()

    def _load_location_info(self):
        """从Excel加载位置信息"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"高度信息Excel列名: {df.columns.tolist()}")

            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()

                try:
                    height_ratio = float(row.iloc[1])
                    height_ratio = max(0.0, min(1.0, height_ratio))
                except:
                    height_ratio = 0.5

                basename = os.path.splitext(filename)[0]
                location_info = {'height_ratio': height_ratio}
                self.location_dict[basename] = location_info

                case_id = extract_case_id(filename)
                if case_id:
                    self.case_to_location[case_id] = location_info

            print(f"成功加载 {len(self.location_dict)} 个文件的高度记录")
        except Exception as e:
            print(f"加载高度信息失败: {e}")
            raise

    def get_height_for_image(self, filename: str) -> float:
        """获取图像对应的高度比例"""
        basename = os.path.splitext(filename)[0]

        if basename in self.location_dict:
            return self.location_dict[basename]['height_ratio']

        case_id = extract_case_id(filename)
        if case_id and case_id in self.case_to_location:
            return self.case_to_location[case_id]['height_ratio']

        return 0.5


# ==============================
# 5. 数据集类
# ==============================

class HeightDataset(Dataset):
    """高度预测数据集"""

    def __init__(self,
                 image_dir: str,
                 height_excel_path: str,
                 position_excel_path: str = None,
                 image_size: Tuple[int, int] = (512, 512),
                 split: str = "train",
                 train_ratio: float = 0.8,
                 augment: bool = False):

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.split = split
        self.augment = augment

        # 加载高度信息
        self.height_loader = HeightInfoLoader(height_excel_path)

        # 加载位置信息（可选）
        self.position_loader = PositionInfoLoader(position_excel_path) if position_excel_path else None

        # 获取所有图像文件
        self.image_files = []
        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg', '']:
                    self.image_files.append(file_path)

        # 划分训练集和验证集
        random.shuffle(self.image_files)
        split_idx = int(len(self.image_files) * train_ratio)

        if split == "train":
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]

        print(f"{split}集: {len(self.image_files)} 个样本")

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """简单的图像增强"""
        if random.random() > 0.5:
            # 随机水平翻转
            image = np.fliplr(image).copy()

        if random.random() > 0.5:
            # 随机亮度调整
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        return image

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        filename = file_path.name

        # 加载图像
        image = load_image(file_path, self.image_size)
        if image is None:
            # 返回默认值
            image = np.zeros(self.image_size, dtype=np.float32)

        # 应用增强
        if self.augment and self.split == "train":
            image = self._augment_image(image)

        # 添加通道维度
        image = np.expand_dims(image, axis=0)
        image_tensor = torch.from_numpy(image).float()

        # 获取高度信息
        height_ratio = self.height_loader.get_height_for_image(filename)
        height_tensor = torch.tensor(height_ratio, dtype=torch.float32)

        # 获取位置信息（如果有）
        if self.position_loader:
            position_tensor, position_idx, position_name = self.position_loader.get_position_for_image(filename)
        else:
            position_tensor = torch.zeros(6)
            position_tensor[2] = 1.0
            position_idx = 2

        return {
            'image': image_tensor,
            'position': position_tensor,
            'height_ratio': height_tensor,
            'filename': filename,
            'position_idx': position_idx
        }


# ==============================
# 6. 高度预测模型
# ==============================

class FeatureReuseDilatedBlock(nn.Module):
    """特征重用的多尺度空洞块"""

    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super().__init__()
        num_branches = len(dilations)
        branch_channels = out_channels // num_branches
        assert out_channels % num_branches == 0
        assert branch_channels % 2 == 0

        self.out_channels = out_channels
        self.dilations = dilations

        self.shared_depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.shared_bn = nn.BatchNorm2d(in_channels)

        self.pointwise_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            ) for _ in dilations
        ])

        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3,
                      padding=d, dilation=d, groups=branch_channels, bias=False)
            for d in dilations
        ])
        self.dilated_bns = nn.ModuleList([nn.BatchNorm2d(branch_channels) for _ in dilations])

        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        shared = self.shared_depthwise(x)
        shared = self.shared_bn(shared)
        shared = self.relu(shared)

        outputs = []
        for pointwise, dilated_conv, dilated_bn in zip(
                self.pointwise_convs, self.dilated_convs, self.dilated_bns):
            branch = pointwise(shared)
            branch = dilated_conv(branch)
            branch = dilated_bn(branch)
            branch = self.relu(branch)
            outputs.append(branch)

        out = torch.cat(outputs, dim=1)
        return self.relu(out + identity)


class HeightEncoder(nn.Module):
    """高度预测编码器"""

    def __init__(self, in_channels=1, base_channels=32, num_blocks=2,
                 block1_dilations=[1, 2, 4, 6], block2_dilations=[8, 10, 12, 14],
                 output_dim=256):
        super().__init__()
        assert base_channels % 2 == 0

        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.output_dim = output_dim

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        current_channels = base_channels
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Block 1
        self.blocks.append(FeatureReuseDilatedBlock(
            current_channels, current_channels * 2, dilations=block1_dilations
        ))
        current_channels = current_channels * 2
        self.downs.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))

        # Block 2
        if num_blocks >= 2:
            self.blocks.append(FeatureReuseDilatedBlock(
                current_channels, current_channels * 2, dilations=block2_dilations
            ))
            current_channels = current_channels * 2
            self.downs.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))

        self.current_channels = current_channels

        # 特征增强块
        self.enhance_block = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True)
        )

        # 最终输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x, return_feature_map=False):
        x = self.initial_conv(x)

        for block, down in zip(self.blocks, self.downs):
            x = block(x)
            x = down(x)

        x_enhanced = self.enhance_block(x)

        if return_feature_map:
            feature_map = x_enhanced

        x = self.final_conv(x_enhanced)
        x = x.view(x.size(0), -1)

        if return_feature_map:
            return x, feature_map
        return x


class PositionEncoder(nn.Module):
    """位置编码器"""

    def __init__(self, num_classes=6, out_features=64, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, out_features * 2),
            nn.BatchNorm1d(out_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FeatureCompressor(nn.Module):
    """特征压缩器"""

    def __init__(self, in_features=256, out_features=64, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.BatchNorm1d(out_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FeatureFusionLayer(nn.Module):
    """特征融合层"""

    def __init__(self, input_dim, output_dim=256, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HeightPredictionHead(nn.Module):
    """高度预测头"""

    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeightOnlyPredictor(nn.Module):
    """
    纯高度预测模型
    只预测高度比例，不包含宽度预测
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 encoder_channels: int = 32,
                 encoder_blocks: int = 2,
                 encoder_dilations1: List[int] = [1, 2, 4, 6],
                 encoder_dilations2: List[int] = [8, 10, 12, 14],
                 num_position_classes: int = 6,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.image_size = image_size
        self.num_position_classes = num_position_classes
        self.target_size = 144

        self.compressed_dim = 64
        self.fusion_dim = 256
        self.head_hidden_dim = 256

        # 位置权重
        self.register_buffer('position_weights', self._create_position_weights())

        # 图像编码器
        self.image_encoder = HeightEncoder(
            in_channels=1,
            base_channels=encoder_channels,
            num_blocks=encoder_blocks,
            block1_dilations=encoder_dilations1,
            block2_dilations=encoder_dilations2,
            output_dim=256
        )

        # 特征压缩
        self.feature_compressor = FeatureCompressor(
            in_features=256,
            out_features=self.compressed_dim,
            dropout_rate=dropout_rate
        )

        # 位置编码
        self.position_encoder = PositionEncoder(
            num_classes=num_position_classes,
            out_features=self.compressed_dim,
            dropout_rate=dropout_rate
        )

        # 特征融合
        fusion_input_dim = self.compressed_dim + self.compressed_dim
        self.fusion_layer = FeatureFusionLayer(
            input_dim=fusion_input_dim,
            output_dim=self.fusion_dim,
            dropout_rate=dropout_rate
        )

        # 高度预测头
        self.height_head = HeightPredictionHead(
            in_features=self.fusion_dim,
            hidden_dim=self.head_hidden_dim,
            dropout_rate=dropout_rate
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'=' * 60}")
        print(f"纯高度预测模型初始化完成")
        print(f"{'=' * 60}")
        print(f"模型参数总量: {total_params / 1e6:.2f}M")
        print(f"编码器块数: {encoder_blocks}")
        print(f"{'=' * 60}\n")

    def _create_position_weights(self):
        weights = torch.ones(6, 6) * 0.02
        adjacency = [
            [0, 1], [0, 1, 2], [1, 2, 3],
            [2, 3, 4], [3, 4, 5], [4, 5]
        ]
        for i in range(6):
            weights[i, i] = 0.64
            for j in adjacency[i]:
                if j != i:
                    weights[i, j] = 0.15
        return weights

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode_position(self, position_info: torch.Tensor) -> torch.Tensor:
        B = position_info.shape[0]
        if position_info.dim() == 1 or position_info.shape[1] == 1:
            indices = position_info.long().view(-1)
            indices = torch.clamp(indices, 0, self.num_position_classes - 1)
            position_dist = self.position_weights[indices]
        else:
            indices = position_info.argmax(dim=1)
            indices = torch.clamp(indices, 0, self.num_position_classes - 1)
            position_dist = self.position_weights[indices]
        return self.position_encoder(position_dist)

    def forward(self, image: torch.Tensor, position_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = image.size(0)

        image_resized = F.interpolate(
            image, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False
        )

        # 图像特征提取
        global_features = self.image_encoder(image_resized)

        # 特征压缩
        image_features = self.feature_compressor(global_features)

        # 位置编码
        position_features = self.encode_position(position_info)

        # 特征融合
        combined = torch.cat([image_features, position_features], dim=1)
        fused_features = self.fusion_layer(combined)

        # 高度预测
        height_ratio = self.height_head(fused_features)

        return {'height_ratio': height_ratio}


# ==============================
# 7. 训练器
# ==============================

class HeightTrainer:
    """高度预测训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 创建输出目录
        self.output_dir = setup_output_directory(
            config.get('model_save_root', "D:/med_data/ai/height_models"),
            prefix="height"
        )

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.final_model_dir = self.output_dir / "final_models"

        for dir_path in [self.checkpoint_dir, self.plots_dir, self.final_model_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"Model save directory: {self.output_dir}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def _create_model(self):
        """创建模型"""
        model = HeightOnlyPredictor(
            image_size=self.config['image_size'],
            encoder_channels=self.config.get('encoder_channels', 32),
            encoder_blocks=self.config.get('encoder_blocks', 2),
            encoder_dilations1=self.config.get('encoder_dilations1', [1, 2, 4, 6]),
            encoder_dilations2=self.config.get('encoder_dilations2', [8, 10, 12, 14]),
            num_position_classes=6,
            dropout_rate=self.config.get('dropout_rate', 0.1)
        ).to(self.device)

        return model

    def _create_dataloaders(self):
        """创建数据加载器"""
        train_dataset = HeightDataset(
            image_dir=self.config['train_image_dir'],
            height_excel_path=self.config['height_excel_path'],
            position_excel_path=self.config.get('position_excel_path'),
            image_size=self.config['image_size'],
            split="train",
            train_ratio=self.config.get('train_ratio', 0.8),
            augment=True
        )

        val_dataset = HeightDataset(
            image_dir=self.config['train_image_dir'],
            height_excel_path=self.config['height_excel_path'],
            position_excel_path=self.config.get('position_excel_path'),
            image_size=self.config['image_size'],
            split="val",
            train_ratio=self.config.get('train_ratio', 0.8),
            augment=False
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
        """训练一个epoch"""
        model.train()

        train_loss_total = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Train]')

        for batch in train_bar:
            images = batch['image'].to(self.device)
            positions = batch['position'].to(self.device)
            heights = batch['height_ratio'].to(self.device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, positions)
            height_pred = outputs['height_ratio']

            # 计算损失
            loss = F.mse_loss(height_pred, heights)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return train_loss_total / len(train_loader)

    def validate_epoch(self, model, val_loader):
        """验证一个epoch"""
        model.eval()

        val_loss_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='[Validation]')
            for batch in val_bar:
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)
                heights = batch['height_ratio'].to(self.device)

                outputs = model(images, positions)
                height_pred = outputs['height_ratio']

                loss = F.mse_loss(height_pred, heights)
                val_loss_total += loss.item()

                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return val_loss_total / len(val_loader)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, train_loss, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'

        torch.save(checkpoint, path)
        return path

    def save_final_model(self, model, best_val_loss, epoch):
        """保存最终模型"""
        print(f"\nSaving final model...")

        model_cpu = model.to('cpu')
        final_model_path = self.final_model_dir / 'height_predictor.pth'

        torch.save({
            'model': model_cpu,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'history': self.history
        }, final_model_path)

        model.to(self.device)
        print(f"✅ Model saved to: {final_model_path}")
        return final_model_path

    def train(self):
        """主训练函数"""
        print("\n" + "=" * 60)
        print("Starting Height Predictor Training")
        print("=" * 60)

        # 创建模型
        model = self._create_model()

        # 创建数据加载器
        train_loader, val_loader = self._create_dataloaders()

        # 优化器
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

        # 早停
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=self.config.get('early_stopping_min_delta', 0.001)
        )

        best_val_loss = float('inf')
        final_epoch = 0
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        best_model_state = None

        for epoch in range(self.config['num_epochs']):
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, epoch)

            # 验证
            val_loss = self.validate_epoch(model, val_loader)

            # 更新学习率
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # 打印信息
            print(f"\n  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                final_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.config,
                }, best_model_path)
                print(f"  ✓ Updated best model (Loss: {best_val_loss:.6f})")

            # 每10个epoch保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss)

            # 早停检查
            if early_stopping(val_loss):
                print(f"\n🚨 Early stopping triggered!")
                break

        # 保存最终模型
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        final_model_path = self.save_final_model(model, best_val_loss, final_epoch)

        # 保存训练历史
        save_training_history(self.history, self.output_dir, self.config)
        plot_training_curves(self.history, self.plots_dir)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best model saved to: {best_model_path}")
        print(f"Final model saved to: {final_model_path}")

        return final_model_path


# ==============================
# 8. 测试器
# ==============================

class HeightTester:
    """高度预测测试器"""

    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 创建输出目录
        self.output_dir = setup_output_directory(
            "D:/med_data/ai/height_test",
            prefix="height_test"
        )
        self.output_dir.mkdir(exist_ok=True)

        print(f"Test results will be saved to: {self.output_dir}")

    def _load_model(self, model_path: str):
        """加载模型"""
        print(f"\nLoading model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'model' in checkpoint:
            model = checkpoint['model']
            model = model.to(self.device)
            print(f"✅ Loaded complete model")
        elif 'model_state_dict' in checkpoint:
            model = HeightOnlyPredictor(
                image_size=self.config['image_size'],
                encoder_channels=self.config.get('encoder_channels', 32),
                encoder_blocks=self.config.get('encoder_blocks', 2),
                num_position_classes=6,
                dropout_rate=self.config.get('dropout_rate', 0.1)
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded weights from checkpoint")
        else:
            raise ValueError(f"Unknown checkpoint format")

        return model

    def test(self):
        """测试模型"""
        print("\n" + "=" * 60)
        print("Starting Height Predictor Testing")
        print("=" * 60)

        # 创建数据集
        test_dataset = HeightDataset(
            image_dir=self.config['test_image_dir'],
            height_excel_path=self.config['height_excel_path'],
            position_excel_path=self.config.get('position_excel_path'),
            image_size=self.config['image_size'],
            split="test",
            train_ratio=self.config.get('train_ratio', 0.8),
            augment=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        results = []
        total_loss = 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Testing')
            for batch in test_bar:
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)
                heights_gt = batch['height_ratio'].to(self.device)
                filenames = batch['filename']

                outputs = self.model(images, positions)
                heights_pred = outputs['height_ratio']

                loss = F.mse_loss(heights_pred, heights_gt)
                total_loss += loss.item()

                for i in range(len(filenames)):
                    results.append({
                        'filename': filenames[i],
                        'height_gt': heights_gt[i].item(),
                        'height_pred': heights_pred[i].item(),
                        'error': abs(heights_pred[i].item() - heights_gt[i].item())
                    })

        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        avg_loss = total_loss / len(test_loader)
        avg_error = results_df['error'].mean()

        print(f"\n{'=' * 60}")
        print("Test Results")
        print(f"{'=' * 60}")
        print(f"Average MSE Loss: {avg_loss:.6f}")
        print(f"Average Absolute Error: {avg_error:.4f}")
        print(f"Results saved to: {self.output_dir}")

        # 绘制误差分布
        plt.figure(figsize=(10, 5))
        plt.hist(results_df['error'], bins=50, alpha=0.7, color='blue')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Height Prediction Error Distribution')
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        return results


# ==============================
# 9. 主函数
# ==============================

def main():
    """主函数"""
    print("=" * 60)
    print("Height Predictor Training and Testing Program")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/translate/reverse/train",
        'test_image_dir': "D:/med_data/ai/translate/reverse/test",
        'height_excel_path': "D:/med_data/ai/translate/reverse/location_contrast_size.xlsx",
        'position_excel_path': "D:/med_data/ai/translate/reverse/classify_all_trans_updated.xlsx",

        # 模型参数
        'image_size': (512, 512),
        'encoder_channels': 32,
        'encoder_blocks': 2,
        'encoder_dilations1': [1, 2, 4, 6],
        'encoder_dilations2': [8, 10, 12, 14],
        'dropout_rate': 0.1,

        # 训练参数
        'batch_size': 16,
        'num_epochs': 60,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'train_ratio': 0.8,

        # 早停参数
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 0.0005,

        # 其他
        'num_workers': 4,
        'model_save_root': "D:/med_data/ai/height_models",
    }

    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 选择模式
    mode = input("\n选择模式 (1: 训练, 2: 测试, 3: 训练+测试): ").strip()

    if mode in ['1', '3']:
        # 训练
        trainer = HeightTrainer(config)
        model_path = trainer.train()

        if mode == '1':
            print(f"\n训练完成！模型保存在: {model_path}")
            return

    if mode in ['2', '3']:
        # 测试 - 需要指定模型路径
        if mode == '2':
            model_path = input("请输入模型路径: ").strip()

        tester = HeightTester(model_path, config)
        results = tester.test()

        print(f"\n测试完成！")
        if results:
            print(f"平均绝对误差: {sum(r['error'] for r in results) / len(results):.4f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()