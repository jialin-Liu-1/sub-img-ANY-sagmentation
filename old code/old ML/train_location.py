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
from typing import Tuple, Dict, Optional, List
import warnings
import json
from datetime import datetime
import re

warnings.filterwarnings('ignore')


# ==============================
# 1. 模型定义
# ==============================

class CoarseInfoExtractor(nn.Module):
    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 base_channels: int = 32,
                 num_position_classes: int = 8,
                 dropout_rate: float = 0.1,
                 pretrain_mode: bool = False):
        super().__init__()

        self.image_size = image_size
        self.base_channels = base_channels
        self.num_position_classes = num_position_classes
        self.dropout_rate = dropout_rate
        self.pretrain_mode = pretrain_mode

        # ========== 图像特征提取路径 ==========
        self.image_encoder = nn.Sequential(
            # 512×512 → 128×128
            nn.Conv2d(1, base_channels, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 128×128 → 32×32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 32×32 → 8×8
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 图像特征压缩
        self.image_feature_compressor = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ========== 位置信息处理路径 ==========
        self.position_encoder = nn.Sequential(
            nn.Linear(num_position_classes, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ========== 特征融合和预测 ==========
        self.feature_fusion = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # 主要输出：高度比例和宽度比例
        self.main_output = nn.Sequential(
            nn.Linear(base_channels, 2),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor, position_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = image.shape

        # 提取图像特征
        image_features = self.image_encoder(image)
        image_features = self.image_feature_compressor(image_features)

        # 处理位置信息
        if position_info.dim() == 1 or position_info.shape[1] == 1:
            indices = position_info.long().view(-1)
            position_onehot = F.one_hot(indices, num_classes=self.num_position_classes).float()
        else:
            position_onehot = position_info.float()

        position_features = self.position_encoder(position_onehot)

        # 特征融合
        combined_features = torch.cat([image_features, position_features], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # 主要预测
        main_params = self.main_output(fused_features)

        height_ratio = main_params[:, 0]  # 高度比例 (窗位)
        width_ratio = main_params[:, 1]  # 宽度比例 (窗宽)

        output_dict = {
            'height_ratio': height_ratio,
            'width_ratio': width_ratio,
            'features': fused_features
        }

        return output_dict


class AttentionMaskGenerator(nn.Module):
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.H, self.W = image_size
        self.device = torch.device('cpu')

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, height_ratio: torch.Tensor,
                width_ratio: torch.Tensor) -> torch.Tensor:
        """
        生成长方形注意力mask

        参数:
            height_ratio: [B,] 高度比例（窗位），0-1之间
            width_ratio: [B,] 宽度比例（窗宽），0-1之间

        返回:
            attention_mask: [B, 1, H, W] 注意力mask
        """
        B = height_ratio.shape[0]
        H, W = self.H, self.W

        if height_ratio.device != self.device:
            self.device = height_ratio.device

        batch_masks = []

        for b in range(B):
            h_ratio = height_ratio[b].item()
            w_ratio = width_ratio[b].item()

            # 计算中心高度位置
            y_center = int(h_ratio * (H - 1))
            y_center = max(0, min(y_center, H - 1))

            # 计算窗宽
            window_height = int(w_ratio * H)
            if window_height < 1:
                window_height = 1
            elif window_height > H:
                window_height = H

            # 计算长方形边界
            half_height = window_height // 2
            y_min = max(0, y_center - half_height)
            y_max = min(H - 1, y_center + half_height)

            # 如果窗宽是奇数，调整边界
            if window_height % 2 == 1:
                if y_min > 0:
                    y_min -= 1
                elif y_max < H - 1:
                    y_max += 1

            # 创建长方形mask
            attention_mask = torch.zeros(H, W, device=self.device)
            attention_mask[y_min:y_max + 1, :] = 1.0

            batch_masks.append(attention_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(batch_masks, dim=0)


# ==============================
# 2. 数据加载和处理
# ==============================

class LocationInfoLoader:
    """加载location.xlsx中的位置信息"""

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}
        self._load_location_info()

    def _load_location_info(self):
        """加载位置信息Excel"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"位置信息Excel列名: {df.columns.tolist()}")

            # 标准化列名
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if '病历' in col_lower or 'filename' in col_lower or '病例' in col_lower:
                    column_mapping[col] = 'filename'
                elif '高度' in col_lower or 'height' in col_lower:
                    column_mapping[col] = 'height_ratio'
                elif '宽度' in col_lower or 'width' in col_lower or 'radius' in col_lower:
                    column_mapping[col] = 'width_ratio'

            df = df.rename(columns=column_mapping)

            # 处理数据
            for _, row in df.iterrows():
                filename = str(row['filename']).strip()

                # 去除可能的扩展名
                filename = os.path.splitext(filename)[0]

                # 获取高度和宽度比例
                try:
                    height_ratio = float(row['height_ratio'])
                    width_ratio = float(row['width_ratio'])

                    # 确保值在0-1范围内
                    height_ratio = max(0.0, min(1.0, height_ratio))
                    width_ratio = max(0.0, min(1.0, width_ratio))

                    self.location_dict[filename] = {
                        'height_ratio': height_ratio,
                        'width_ratio': width_ratio
                    }
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    continue

            print(f"成功加载 {len(self.location_dict)} 条位置信息")

        except Exception as e:
            print(f"加载位置信息失败: {e}")
            raise

    def get_location_for_image(self, filename: str):
        """获取图像的位置信息"""
        # 尝试不同的文件名格式
        basename = os.path.splitext(filename)[0]

        # 尝试直接匹配
        if basename in self.location_dict:
            return self.location_dict[basename]

        # 尝试提取病历号
        parts = basename.split('_')
        if len(parts) >= 2:
            record_num = parts[1]
            for key in self.location_dict.keys():
                if record_num in key:
                    return self.location_dict[key]

        # 返回默认值
        print(f"警告: 未找到图像 {filename} 的位置信息，使用默认值")
        return {'height_ratio': 0.5, 'width_ratio': 0.3}


class PositionInfoLoader:
    """加载分类位置信息（如果需要的话）"""

    def __init__(self, excel_path: str = None):
        self.position_dict = {}
        if excel_path:
            self._load_position_info(excel_path)

    def _load_position_info(self, excel_path: str):
        """加载位置分类信息"""
        try:
            df = pd.read_excel(excel_path)
            print(f"位置分类Excel列名: {df.columns.tolist()}")

            # 这里需要根据实际表格格式进行调整
            # 假设第一列是病历号，第二列是位置分类
            for _, row in df.iterrows():
                record_num = str(row.iloc[0]).strip()
                position_num = int(row.iloc[1]) if not pd.isna(row.iloc[1]) else 0

                # 确保位置编号在0-7之间
                position_num = max(0, min(7, position_num))
                self.position_dict[record_num] = position_num

            print(f"成功加载 {len(self.position_dict)} 条位置分类信息")

        except Exception as e:
            print(f"加载位置分类信息失败: {e}")
            # 如果没有分类信息，使用默认值
            pass

    def get_position_for_image(self, filename: str):
        """获取图像的位置分类"""
        basename = os.path.splitext(filename)[0]

        # 尝试提取病历号
        match = re.search(r'(\d+)', basename)
        if match:
            record_num = match.group(1)
            if record_num in self.position_dict:
                position_num = self.position_dict[record_num]
                # 转换为one-hot编码
                position_tensor = torch.zeros(8)
                position_tensor[position_num] = 1.0
                return position_tensor

        # 返回默认位置（中间位置）
        position_tensor = torch.zeros(8)
        position_tensor[4] = 1.0  # 假设中间位置是4
        return position_tensor


class CoarseExtractorDataset(Dataset):
    """CoarseInfoExtractor的训练数据集"""

    def __init__(self,
                 image_dir: str,
                 location_excel_path: str,
                 position_excel_path: str = None,
                 image_size: Tuple[int, int] = (512, 512),
                 max_samples: int = None):

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # 加载位置信息
        self.location_loader = LocationInfoLoader(location_excel_path)

        # 加载位置分类信息（可选）
        self.position_loader = PositionInfoLoader(position_excel_path)

        # 获取图像文件列表
        self.image_files = self._get_image_files()

        if max_samples:
            self.image_files = self.image_files[:max_samples]

        print(f"数据集初始化完成，共 {len(self.image_files)} 个样本")

    def _get_image_files(self):
        """获取图像文件列表"""
        image_files = []

        # 支持无后缀文件和常见图像格式
        valid_extensions = {'.dcm', '.dicom', '', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}

        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in valid_extensions or file_path.suffix == '':
                    image_files.append(file_path.name)

        return image_files

    def _load_image(self, filename: str) -> np.ndarray:
        """加载图像"""
        file_path = self.image_dir / filename

        try:
            # 尝试加载DICOM文件
            if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(str(file_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                # 加载普通图像
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"无法加载图像: {file_path}")
                image = image.astype(np.float32)

            # 归一化
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # 调整尺寸
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载图像 {filename} 失败: {e}")
            return np.zeros(self.image_size, dtype=np.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        try:
            # 加载图像
            image = self._load_image(filename)
            image = np.expand_dims(image, axis=0)  # 添加通道维度
            image_tensor = torch.from_numpy(image).float()

            # 获取位置信息（训练目标）
            location_info = self.location_loader.get_location_for_image(filename)
            height_ratio = torch.tensor(location_info['height_ratio'], dtype=torch.float32)
            width_ratio = torch.tensor(location_info['width_ratio'], dtype=torch.float32)

            # 获取位置分类信息（模型输入）
            position_tensor = self.position_loader.get_position_for_image(filename)

            return {
                'image': image_tensor,
                'position': position_tensor,
                'height_ratio': height_ratio,
                'width_ratio': width_ratio,
                'filename': filename
            }

        except Exception as e:
            print(f"处理样本 {filename} 失败: {e}")
            # 返回默认张量
            return {
                'image': torch.zeros((1, *self.image_size), dtype=torch.float32),
                'position': torch.zeros(8, dtype=torch.float32),
                'height_ratio': torch.tensor(0.5, dtype=torch.float32),
                'width_ratio': torch.tensor(0.3, dtype=torch.float32),
                'filename': filename
            }


# ==============================
# 3. 训练器
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


class CoarseExtractorTrainer:
    """CoarseInfoExtractor训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"D:/med_data/ai/coarse_extractor_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"

        for dir_path in [self.checkpoint_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"输出目录: {self.output_dir}")

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

    def _create_model(self):
        """创建模型"""
        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=self.config['num_position_classes'],
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
        ).to(self.device)

        return model

    def _create_dataloaders(self):
        """创建数据加载器"""
        train_dataset = CoarseExtractorDataset(
            image_dir=self.config['train_image_dir'],
            location_excel_path=self.config['location_excel_path'],
            position_excel_path=self.config.get('position_excel_path'),
            image_size=self.config['image_size'],
            max_samples=self.config.get('max_train_samples')
        )

        val_dataset = CoarseExtractorDataset(
            image_dir=self.config['val_image_dir'],
            location_excel_path=self.config['location_excel_path'],
            position_excel_path=self.config.get('position_excel_path'),
            image_size=self.config['image_size'],
            max_samples=self.config.get('max_val_samples')
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

        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")

        return train_loader, val_loader

    def train(self):
        """训练模型"""
        print("\n" + "=" * 60)
        print("开始训练 CoarseInfoExtractor")
        print("=" * 60)

        # 创建模型
        model = self._create_model()

        # 创建数据加载器
        train_loader, val_loader = self._create_dataloaders()

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

        # 损失函数
        criterion = nn.MSELoss()

        # 训练循环
        best_val_loss = float('inf')
        best_model_path = None

        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            model.train()
            train_loss_total = 0
            train_loss_height = 0
            train_loss_width = 0

            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Train]')
            for batch_idx, batch in enumerate(train_bar):
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)
                height_targets = batch['height_ratio'].to(self.device)
                width_targets = batch['width_ratio'].to(self.device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(images, positions)
                height_pred = outputs['height_ratio']
                width_pred = outputs['width_ratio']

                # 计算损失
                height_loss = criterion(height_pred, height_targets)
                width_loss = criterion(width_pred, width_targets)
                loss = height_loss + width_loss

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 记录损失
                train_loss_total += loss.item()
                train_loss_height += height_loss.item()
                train_loss_width += width_loss.item()

                # 更新进度条
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'height': f'{height_loss.item():.4f}',
                    'width': f'{width_loss.item():.4f}'
                })

            # 验证阶段
            model.eval()
            val_loss_total = 0
            val_loss_height = 0
            val_loss_width = 0

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [Val]')
                for batch in val_bar:
                    images = batch['image'].to(self.device)
                    positions = batch['position'].to(self.device)
                    height_targets = batch['height_ratio'].to(self.device)
                    width_targets = batch['width_ratio'].to(self.device)

                    # 前向传播
                    outputs = model(images, positions)
                    height_pred = outputs['height_ratio']
                    width_pred = outputs['width_ratio']

                    # 计算损失
                    height_loss = criterion(height_pred, height_targets)
                    width_loss = criterion(width_pred, width_targets)
                    loss = height_loss + width_loss

                    # 记录损失
                    val_loss_total += loss.item()
                    val_loss_height += height_loss.item()
                    val_loss_width += width_loss.item()

                    val_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'height': f'{height_loss.item():.4f}',
                        'width': f'{width_loss.item():.4f}'
                    })

            # 计算平均损失
            train_loss_avg = train_loss_total / len(train_loader)
            val_loss_avg = val_loss_total / len(val_loader)

            train_height_avg = train_loss_height / len(train_loader)
            train_width_avg = train_loss_width / len(train_loader)
            val_height_avg = val_loss_height / len(val_loader)
            val_width_avg = val_loss_width / len(val_loader)

            # 更新学习率
            scheduler.step(val_loss_avg)

            # 记录历史
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss_avg)
            self.history['train_height_loss'].append(train_height_avg)
            self.history['train_width_loss'].append(train_width_avg)
            self.history['val_height_loss'].append(val_height_avg)
            self.history['val_width_loss'].append(val_width_avg)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 打印统计信息
            print(f"\nEpoch {epoch + 1} 统计:")
            print(f"  训练损失: {train_loss_avg:.4f} (高度: {train_height_avg:.4f}, 宽度: {train_width_avg:.4f})")
            print(f"  验证损失: {val_loss_avg:.4f} (高度: {val_height_avg:.4f}, 宽度: {val_width_avg:.4f})")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_path = self.checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'config': self.config,
                    'history': self.history
                }, best_model_path)

                print(f"  ✓ 保存最佳模型到: {best_model_path}")

            # 每5个epoch保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg
                }, checkpoint_path)
                print(f"  ✓ 保存检查点到: {checkpoint_path}")

            # 早停检查
            if early_stopping(val_loss_avg):
                print(f"\n🚨 早停触发! 连续 {early_stopping.patience} 个epoch未改善")
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

        print(f"\n训练完成! 共训练 {epoch + 1} 个epoch")
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

        # 总损失
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('总损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 高度损失
        axes[0, 1].plot(epochs, self.history['train_height_loss'], 'b-', label='训练高度损失')
        axes[0, 1].plot(epochs, self.history['val_height_loss'], 'r-', label='验证高度损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].set_title('高度比例损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 宽度损失
        axes[1, 0].plot(epochs, self.history['train_width_loss'], 'b-', label='训练宽度损失')
        axes[1, 0].plot(epochs, self.history['val_width_loss'], 'r-', label='验证宽度损失')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].set_title('宽度比例损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 学习率
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('学习率')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练曲线保存到: {self.plots_dir / 'training_history.png'}")

    def _save_training_history(self):
        """保存训练历史"""
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_height_loss': self.history['train_height_loss'],
            'train_width_loss': self.history['train_width_loss'],
            'val_height_loss': self.history['val_height_loss'],
            'val_width_loss': self.history['val_width_loss'],
            'learning_rate': self.history['learning_rate']
        })

        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)

        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, default=str)

        print(f"训练历史保存到: {self.output_dir / 'training_history.csv'}")


# ==============================
# 4. 测试和推理
# ==============================
class CoarseExtractorTester:
    """CoarseInfoExtractor测试器"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 创建注意力mask生成器
        self.mask_generator = AttentionMaskGenerator(
            image_size=config['image_size']
        ).to(self.device)

        # 加载位置信息
        self.location_loader = LocationInfoLoader(config['location_excel_path'])
        self.position_loader = PositionInfoLoader(config.get('position_excel_path'))

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"D:/med_data/ai/test_results_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.masks_dir = self.output_dir / "masks"
        self.focused_dir = self.output_dir / "focused_images"
        self.comparison_dir = self.output_dir / "comparison"

        for dir_path in [self.masks_dir, self.focused_dir, self.comparison_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"测试结果将保存到: {self.output_dir}")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=self.config['num_position_classes'],
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # 安全地打印训练信息
        epoch = checkpoint.get('epoch', '未知')
        val_loss = checkpoint.get('val_loss')

        print(f"模型加载成功，训练轮数: {epoch}")
        if isinstance(val_loss, (int, float)):
            print(f"验证损失: {val_loss:.4f}")
        else:
            print(f"验证损失: 未知")

        return model

    def _load_image(self, image_path):
        """加载图像"""
        try:
            # 尝试加载DICOM文件
            if image_path.suffix == '' or image_path.suffix.lower() in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(str(image_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                # 加载普通图像
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"无法加载图像: {image_path}")
                image = image.astype(np.float32)

            # 归一化
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # 调整尺寸
            if image.shape != self.config['image_size']:
                image = cv2.resize(image, self.config['image_size'], interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载图像 {image_path.name} 失败: {e}")
            return None

    def _save_image(self, image, path, normalize=True):
        """保存图像"""
        if normalize and image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)

        cv2.imwrite(str(path), image)

    def test_single_image(self, image_path):
        """测试单个图像"""
        filename = image_path.name
        basename = os.path.splitext(filename)[0]

        print(f"\n处理图像: {filename}")

        # 加载图像
        image = self._load_image(image_path)
        if image is None:
            print(f"  跳过图像 {filename}")
            return None

        # 准备输入
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        position_tensor = self.position_loader.get_position_for_image(filename).unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(image_tensor, position_tensor)
            height_pred = outputs['height_ratio'].item()
            width_pred = outputs['width_ratio'].item()

        print(f"  预测值 - 高度: {height_pred:.4f}, 宽度: {width_pred:.4f}")

        # 获取真实值（如果存在）
        try:
            true_info = self.location_loader.get_location_for_image(filename)
            height_true = true_info['height_ratio']
            width_true = true_info['width_ratio']
            print(f"  真实值 - 高度: {height_true:.4f}, 宽度: {width_true:.4f}")
            print(f"  误差 - 高度: {abs(height_pred - height_true):.4f}, 宽度: {abs(width_pred - width_true):.4f}")
        except:
            height_true = width_true = None

        # 生成注意力mask
        height_tensor = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
        width_tensor = torch.tensor([width_pred], dtype=torch.float32).to(self.device)

        attention_mask = self.mask_generator(height_tensor, width_tensor)
        attention_mask_np = attention_mask.squeeze().cpu().numpy()

        # 应用注意力mask
        focused_image = image * attention_mask_np

        # 保存结果
        # 1. 保存原始图像
        self._save_image(image, self.output_dir / f"{basename}_original.png")

        # 2. 保存注意力mask
        mask_uint8 = (attention_mask_np * 255).astype(np.uint8)
        self._save_image(mask_uint8, self.masks_dir / f"{basename}_mask.png", normalize=False)

        # 3. 保存应用了注意力的图像
        self._save_image(focused_image, self.focused_dir / f"{basename}_focused.png")

        # 4. 保存对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].imshow(attention_mask_np, cmap='gray')
        axes[1].set_title(f'注意力mask\n高度: {height_pred:.3f}, 宽度: {width_pred:.3f}')
        axes[1].axis('off')

        axes[2].imshow(focused_image, cmap='gray')
        axes[2].set_title('应用注意力的图像')
        axes[2].axis('off')

        if height_true is not None:
            plt.suptitle(
                f"{filename}\n预测: 高度={height_pred:.3f}, 宽度={width_pred:.3f}\n真实: 高度={height_true:.3f}, 宽度={width_true:.3f}")
        else:
            plt.suptitle(f"{filename}\n预测: 高度={height_pred:.3f}, 宽度={width_pred:.3f}")

        plt.tight_layout()
        plt.savefig(self.comparison_dir / f"{basename}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'filename': filename,
            'height_pred': height_pred,
            'width_pred': width_pred,
            'height_true': height_true,
            'width_true': width_true,
            'height_error': abs(height_pred - height_true) if height_true else None,
            'width_error': abs(width_pred - width_true) if width_true else None
        }

    def test_all_images(self, test_image_dir):
        """测试所有图像"""
        test_dir = Path(test_image_dir)

        # 获取图像文件
        image_files = []
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']:
                    image_files.append(file_path)

        print(f"\n开始测试 {len(image_files)} 个图像...")
        print("=" * 60)

        results = []

        for i, image_path in enumerate(tqdm(image_files, desc="测试进度")):
            result = self.test_single_image(image_path)
            if result:
                results.append(result)

            # 每10个图像打印一次进度
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(image_files)} 个图像")

        # 保存测试结果
        if results:
            self._save_test_results(results)

        return results

    def _save_test_results(self, results):
        """保存测试结果"""
        # 保存为CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        # 计算统计信息
        if results_df['height_error'].notna().any() and results_df['width_error'].notna().any():
            height_errors = results_df['height_error'].dropna()
            width_errors = results_df['width_error'].dropna()

            stats = {
                'total_images': len(results),
                'images_with_ground_truth': len(height_errors),
                'mean_height_error': height_errors.mean(),
                'std_height_error': height_errors.std(),
                'mean_width_error': width_errors.mean(),
                'std_width_error': width_errors.std(),
                'max_height_error': height_errors.max(),
                'max_width_error': width_errors.max(),
                'min_height_error': height_errors.min(),
                'min_width_error': width_errors.min()
            }

            # 保存统计信息
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(self.output_dir / 'test_statistics.csv', index=False)

            # 保存为文本文件
            with open(self.output_dir / 'test_summary.txt', 'w') as f:
                f.write("测试结果总结\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"总测试图像数: {stats['total_images']}\n")
                f.write(f"有真实值的图像数: {stats['images_with_ground_truth']}\n\n")
                f.write("高度比例误差统计:\n")
                f.write(f"  均值: {stats['mean_height_error']:.6f}\n")
                f.write(f"  标准差: {stats['std_height_error']:.6f}\n")
                f.write(f"  最大值: {stats['max_height_error']:.6f}\n")
                f.write(f"  最小值: {stats['min_height_error']:.6f}\n\n")
                f.write("宽度比例误差统计:\n")
                f.write(f"  均值: {stats['mean_width_error']:.6f}\n")
                f.write(f"  标准差: {stats['std_width_error']:.6f}\n")
                f.write(f"  最大值: {stats['max_width_error']:.6f}\n")
                f.write(f"  最小值: {stats['min_width_error']:.6f}\n")

            print(f"\n测试结果统计:")
            print(f"  高度比例误差均值: {stats['mean_height_error']:.6f}")
            print(f"  宽度比例误差均值: {stats['mean_width_error']:.6f}")

        print(f"\n测试结果保存到: {self.output_dir}")


# ==============================
# 5. 主程序
# ==============================
def main():
    """主函数"""
    print("CoarseInfoExtractor 训练和测试程序")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/train1",
        'val_image_dir': "D:/med_data/ai/test1",
        'test_image_dir': "D:/med_data/ai/test1",  # 测试时使用test1
        'location_excel_path': "D:/med_data/ai/location.xlsx",
        'position_excel_path': "D:/med_data/ai/classify.xlsx",  # 可选

        # 模型参数
        'image_size': (512, 512),
        'base_channels': 32,
        'num_position_classes': 8,
        'dropout_rate': 0.1,

        # 训练参数
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,

        # 早停参数
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,

        # 其他参数
        'num_workers': 2,
        'max_train_samples': None,
        'max_val_samples': None,
    }

    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # 步骤1: 训练模型
    print("\n步骤1: 训练 CoarseInfoExtractor 模型")
    trainer = CoarseExtractorTrainer(config)
    trained_model_path = trainer.train()

    # 步骤2: 测试模型
    print("\n" + "=" * 60)
    print("步骤2: 测试训练好的模型")

    # 确保trained_model_path是字符串
    if isinstance(trained_model_path, Path):
        trained_model_path = str(trained_model_path)

    tester = CoarseExtractorTester(trained_model_path, config)
    test_results = tester.test_all_images(config['test_image_dir'])

    print("\n" + "=" * 60)
    print("程序完成!")
    print(f"训练输出目录: {trainer.output_dir}")
    print(f"测试输出目录: {tester.output_dir}")

    # 显示一些示例结果
    if test_results:
        print(f"\n测试了 {len(test_results)} 个图像")

        # 显示前5个结果
        print("\n前5个测试结果:")
        for i, result in enumerate(test_results[:5]):
            print(f"  {i + 1}. {result['filename']}:")
            print(f"     预测 - 高度: {result['height_pred']:.4f}, 宽度: {result['width_pred']:.4f}")
            if result['height_true']:
                print(f"     真实 - 高度: {result['height_true']:.4f}, 宽度: {result['width_true']:.4f}")
                print(f"     误差 - 高度: {result['height_error']:.4f}, 宽度: {result['width_error']:.4f}")


if __name__ == "__main__":
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 运行主程序
    main()