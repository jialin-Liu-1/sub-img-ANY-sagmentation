"""
CoarseInfoExtractor 模型测试代码
包含两种测试方法：
1. 直接读取位置信息和图像进行测试
2. 遍历所有位置类别（0-7）生成对比图像
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
from typing import Tuple, Dict, Optional, List, Union
import warnings
import json
from datetime import datetime
import re
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')


# ==============================
# 复用训练代码中的模型定义
# ==============================

class CoarseInfoExtractor(nn.Module):
    """改进的CoarseInfoExtractor，支持6类动脉瘤类别（1,2,4,5,6,7）"""

    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 base_channels: int = 32,
                 num_position_classes: int = 6,  # 改为6类
                 dropout_rate: float = 0.1,
                 pretrain_mode: bool = False):
        super().__init__()

        self.image_size = image_size
        self.base_channels = base_channels
        self.num_position_classes = num_position_classes  # 6类
        self.dropout_rate = dropout_rate
        self.pretrain_mode = pretrain_mode

        # 定义类别映射（原始8类到新6类的映射）
        self.class_mapping = {
            1: 0,  # 原始1类 -> 新0类
            2: 1,  # 原始2类 -> 新1类
            4: 2,  # 原始4类 -> 新2类
            5: 3,  # 原始5类 -> 新3类
            6: 4,  # 原始6类 -> 新4类
            7: 5  # 原始7类 -> 新5类
        }

        # 反向映射（用于解码）
        self.inverse_mapping = {v: k for k, v in self.class_mapping.items()}

        # 类别名称
        self.class_names = {
            0: "Segment 1 (原1类)",
            1: "Segment 2 (原2类)",
            2: "Segment 4 (原4类)",
            3: "Segment 5 (原5类)",
            4: "Segment 6 (原6类)",
            5: "Segment 7 (原7类)"
        }

        # ========== Image feature extraction path ==========
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

        # Image feature compression
        self.image_feature_compressor = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ========== Position information processing path (6类) ==========
        self.position_encoder = nn.Sequential(
            nn.Linear(num_position_classes, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ========== Feature fusion and prediction ==========
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

        # Main output: height ratio and width ratio
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

        # Extract image features
        image_features = self.image_encoder(image)
        image_features = self.image_feature_compressor(image_features)

        # Process position information (6类one-hot)
        if position_info.dim() == 1 or position_info.shape[1] == 1:
            indices = position_info.long().view(-1)
            # 确保索引在0-5范围内
            indices = torch.clamp(indices, 0, self.num_position_classes - 1)
            position_onehot = F.one_hot(indices, num_classes=self.num_position_classes).float()
        else:
            # 如果输入是one-hot，确保维度正确
            if position_info.shape[1] != self.num_position_classes:
                # 如果维度不匹配，尝试转换
                if position_info.shape[1] == 8:  # 如果是8类one-hot
                    # 提取有效类别(1,2,4,5,6,7)对应的索引
                    valid_indices = [1, 2, 4, 5, 6, 7]
                    position_onehot = torch.zeros(B, self.num_position_classes, device=position_info.device)
                    for i, idx in enumerate(valid_indices):
                        if idx < position_info.shape[1]:
                            position_onehot[:, i] = position_info[:, idx]
                else:
                    position_onehot = position_info.float()
            else:
                position_onehot = position_info.float()

        position_features = self.position_encoder(position_onehot)

        # Feature fusion
        combined_features = torch.cat([image_features, position_features], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Main prediction
        main_params = self.main_output(fused_features)

        height_ratio = main_params[:, 0]  # Height ratio (window level)
        width_ratio = main_params[:, 1]  # Width ratio (window width)

        output_dict = {
            'height_ratio': height_ratio,
            'width_ratio': width_ratio,
            'features': fused_features
        }

        return output_dict


class AttentionMaskGenerator(nn.Module):
    def __init__(self, image_size: Tuple[int, int] = (512, 512), min_radius_ratio: float = 0.06):
        super().__init__()
        self.H, self.W = image_size
        self.min_radius_ratio = min_radius_ratio
        self.device = torch.device('cpu')

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

            # Calculate center height position
            y_center = int(h_ratio * (H - 1))
            y_center = max(0, min(y_center, H - 1))

            # Calculate window width
            window_height = int(w_ratio * H)
            if window_height < 1:
                window_height = 1
            elif window_height > H:
                window_height = H

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

        if excel_path and Path(excel_path).exists():
            self._load_position_info(excel_path)

    def _load_position_info(self, excel_path: str):
        """加载类别信息"""
        try:
            df = pd.read_excel(excel_path)
            print(f"类别信息Excel列名: {df.columns.tolist()}")

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
                        match = re.search(r'([A-Za-z]+_)?(\d+)', base_name)
                        if match:
                            case_id = match.group(0)  # 完整匹配
                            # 保存病历号到类别的映射
                            if case_id not in self.case_to_position:
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

        # 方法1: 直接匹配
        if basename in self.position_dict:
            pos_info = self.position_dict[basename]
            position_tensor = torch.zeros(6)
            position_tensor[pos_info['new_index']] = 1.0
            return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 方法2: 提取病历号进行匹配
        match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
        if match:
            case_id = match.group(0)
            if case_id in self.case_to_position:
                pos_info = self.case_to_position[case_id]
                position_tensor = torch.zeros(6)
                position_tensor[pos_info['new_index']] = 1.0
                return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 返回默认值
        print(f"警告: 未找到图像 {filename} 的类别信息，使用默认类别0")
        position_tensor = torch.zeros(6)
        position_tensor[0] = 1.0
        return position_tensor, 0, "Segment 1 (default)"


class LocationInfoLoader:
    """加载位置信息"""

    def __init__(self, excel_path: str):
        self.location_dict = {}
        self.case_to_location = {}
        if Path(excel_path).exists():
            self._load_location_info(excel_path)

    def _load_location_info(self, excel_path: str):
        """从Excel加载位置信息"""
        try:
            df = pd.read_excel(excel_path)

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

            for _, row in df.iterrows():
                filename = str(row['filename']).strip()
                basename = os.path.splitext(filename)[0]

                try:
                    height_ratio = float(row['height_ratio'])
                    width_ratio = float(row['width_ratio'])

                    height_ratio = max(0.0, min(1.0, height_ratio))
                    width_ratio = max(0.0, min(1.0, width_ratio))

                    location_info = {
                        'height_ratio': height_ratio,
                        'width_ratio': width_ratio
                    }

                    self.location_dict[basename] = location_info

                    # 提取病历号
                    match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
                    if match:
                        case_id = match.group(0)
                        if case_id not in self.case_to_location:
                            self.case_to_location[case_id] = location_info

                except Exception as e:
                    continue

            print(f"成功加载 {len(self.location_dict)} 个文件的位置记录")

        except Exception as e:
            print(f"加载位置信息失败: {e}")

    def get_location_for_image(self, filename: str):
        """获取图像对应的位置信息"""
        basename = os.path.splitext(filename)[0]

        # 方法1: 直接匹配
        if basename in self.location_dict:
            return self.location_dict[basename]

        # 方法2: 提取病历号匹配
        match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
        if match:
            case_id = match.group(0)
            if case_id in self.case_to_location:
                return self.case_to_location[case_id]

        # 返回默认值
        return {'height_ratio': 0.5, 'width_ratio': 0.3}


# ==============================
# 模型测试器（包含两种测试方法）
# ==============================

class ModelTester:
    """CoarseInfoExtractor 模型测试器"""

    def __init__(self, model_path: str, config: dict):
        """
        初始化测试器

        Args:
            model_path: 训练好的模型路径
            config: 配置参数
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 创建注意力掩码生成器
        self.mask_generator = AttentionMaskGenerator(
            image_size=config.get('image_size', (512, 512))
        ).to(self.device)

        # 加载位置信息（如果提供）
        self.location_loader = None
        if 'location_excel_path' in config:
            self.location_loader = LocationInfoLoader(config['location_excel_path'])

        self.position_loader = None
        if 'position_excel_path' in config:
            self.position_loader = PositionInfoLoader(config.get('position_excel_path'))

        # 创建输出目录
        self._setup_output_dirs()

        # 类别名称映射
        self.class_names = {
            0: "C1 (原1类)",
            1: "C2 (原2类)",
            2: "C4 (原4类)",
            3: "C5 (原5类)",
            4: "C6 (原6类)",
            5: "C7 (原7类)"
        }

        # 颜色映射（用于不同类别的显示）
        self.class_colors = [
            (1, 0, 0),  # 红色 - C1
            (0, 1, 0),  # 绿色 - C2
            (0, 0, 1),  # 蓝色 - C4
            (1, 1, 0),  # 黄色 - C5
            (1, 0, 1),  # 品红 - C6
            (0, 1, 1)  # 青色 - C7
        ]

    def _setup_output_dirs(self):
        """创建输出目录"""
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"./test_results_{date_str}")
        self.output_dir.mkdir(exist_ok=True)

        # 方法1的输出目录
        self.method1_dir = self.output_dir / "method1_single_image"
        self.method1_dir.mkdir(exist_ok=True)

        # 方法2的输出目录
        self.method2_dir = self.output_dir / "method2_position_scan"
        self.method2_dir.mkdir(exist_ok=True)

        print(f"测试结果保存到: {self.output_dir}")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = CoarseInfoExtractor(
            image_size=self.config.get('image_size', (512, 512)),
            base_channels=self.config.get('base_channels', 32),
            num_position_classes=6,
            dropout_rate=self.config.get('dropout_rate', 0.2)
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        print(f"模型加载成功 - Epoch: {epoch}, Val Loss: {val_loss}")

        return model

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """加载并预处理图像"""
        try:
            # 尝试加载DICOM
            if image_path.suffix.lower() in ['.dcm', '.dicom'] or image_path.suffix == '':
                dicom_data = pydicom.dcmread(str(image_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                # 加载普通图像
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return None
                image = image.astype(np.float32)

            # 归一化
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # 调整大小
            target_size = self.config.get('image_size', (512, 512))
            if image.shape != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None

    def _get_position_tensor(self, position_idx: int) -> torch.Tensor:
        """
        根据位置索引创建one-hot张量

        Args:
            position_idx: 位置索引 (0-5)

        Returns:
            one-hot张量
        """
        position_tensor = torch.zeros(6)
        position_tensor[position_idx] = 1.0
        return position_tensor

    def _predict(self, image: np.ndarray, position_idx: int) -> Tuple[float, float, np.ndarray]:
        """
        对单张图像进行预测

        Args:
            image: 输入图像 (H, W)
            position_idx: 位置索引 (0-5)

        Returns:
            height_pred: 预测的高度比例
            width_pred: 预测的宽度比例
            attention_mask: 注意力掩膜
        """
        # 准备输入
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        position_tensor = self._get_position_tensor(position_idx).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor, position_tensor)
            height_pred = outputs['height_ratio'].item()
            width_pred = outputs['width_ratio'].item()

        # 生成注意力掩膜
        height_tensor = torch.tensor([height_pred], device=self.device)
        width_tensor = torch.tensor([width_pred], device=self.device)
        attention_mask = self.mask_generator(height_tensor, width_tensor)
        attention_mask_np = attention_mask.squeeze().cpu().numpy()

        return height_pred, width_pred, attention_mask_np

    def _draw_rectangle_on_image(self, image: np.ndarray, mask: np.ndarray,
                                 color: Tuple[float, float, float] = (1, 0, 0)) -> np.ndarray:
        """在图像上绘制矩形区域"""
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image.copy()

        # 找到掩膜的非零区域
        y_indices, x_indices = np.where(mask > 0.5)
        if len(y_indices) > 0 and len(x_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # 绘制矩形边框
            image_rgb[y_min:y_max, x_min:x_min + 2] = color  # 左边框
            image_rgb[y_min:y_max, x_max - 2:x_max] = color  # 右边框
            image_rgb[y_min:y_min + 2, x_min:x_max] = color  # 上边框
            image_rgb[y_max - 2:y_max, x_min:x_max] = color  # 下边框

            # 添加半透明覆盖
            overlay = np.zeros_like(image_rgb)
            overlay[y_min:y_max, x_min:x_max] = color
            image_rgb = image_rgb * 0.7 + overlay * 0.3

        return np.clip(image_rgb, 0, 1)

    # ========== 测试方法1: 直接读取位置信息进行测试 ==========

    def test_method1_single_image(self, image_path: Path, position_idx: int = None):
        """
        方法1: 直接读取单张图像进行测试
        如果提供position_idx，则使用该位置；否则从position_loader获取
        """
        filename = image_path.name
        basename = os.path.splitext(filename)[0]

        print(f"\n处理图像: {filename}")

        # 加载图像
        image = self._load_image(image_path)
        if image is None:
            print(f"  图像加载失败")
            return None

        # 获取位置索引
        if position_idx is None and self.position_loader:
            _, position_idx, position_name = self.position_loader.get_position_for_image(filename)
        elif position_idx is None:
            position_idx = 0
            position_name = self.class_names[0]
        else:
            position_name = self.class_names.get(position_idx, f"位置{position_idx}")

        print(f"  使用位置: {position_name}")

        # 预测
        height_pred, width_pred, attention_mask = self._predict(image, position_idx)
        print(f"  预测 - 高度: {height_pred:.4f}, 宽度: {width_pred:.4f}")

        # 获取真实值（如果有）
        height_true = None
        width_true = None
        if self.location_loader:
            true_info = self.location_loader.get_location_for_image(filename)
            height_true = true_info['height_ratio']
            width_true = true_info['width_ratio']
            print(f"  真实值 - 高度: {height_true:.4f}, 宽度: {width_true:.4f}")
            print(f"  误差 - 高度: {abs(height_pred - height_true):.4f}, 宽度: {abs(width_pred - width_true):.4f}")

        # 创建结果图像
        self._save_method1_result(image, attention_mask, filename, basename,
                                  position_idx, position_name, height_pred, width_pred,
                                  height_true, width_true)

        return {
            'filename': filename,
            'position_idx': position_idx,
            'position_name': position_name,
            'height_pred': height_pred,
            'width_pred': width_pred,
            'height_true': height_true,
            'width_true': width_true,
            'height_error': abs(height_pred - height_true) if height_true else None,
            'width_error': abs(width_pred - width_true) if width_true else None
        }

    def _save_method1_result(self, image, attention_mask, filename, basename,
                             position_idx, position_name, height_pred, width_pred,
                             height_true, width_true):
        """保存方法1的测试结果"""
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        # 注意力掩膜
        axes[1].imshow(attention_mask, cmap='hot', alpha=0.7)
        axes[1].set_title(f'注意力掩膜\n位置: {position_name}')
        axes[1].axis('off')

        # 叠加图像
        overlay_image = self._draw_rectangle_on_image(image, attention_mask, self.class_colors[position_idx])
        axes[2].imshow(overlay_image)

        title = f'预测区域\n高度: {height_pred:.3f}, 宽度: {width_pred:.3f}'
        if height_true:
            title += f'\n真实: 高度={height_true:.3f}, 宽度={width_true:.3f}'
        axes[2].set_title(title)
        axes[2].axis('off')

        plt.suptitle(f"{filename}\n位置类别: {position_name}", fontsize=12)
        plt.tight_layout()

        # 保存
        save_path = self.method1_dir / f"{basename}_pos{position_idx}_result.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  结果保存到: {save_path}")

    def test_method1_batch(self, image_dir: str, specific_position: int = None):
        """
        批量测试方法1

        Args:
            image_dir: 图像目录
            specific_position: 指定位置索引，如果为None则从position_loader获取
        """
        image_dir = Path(image_dir)
        image_files = []

        # 收集图像文件
        for ext in ['*.dcm', '*.png', '*.jpg', '*.jpeg']:
            image_files.extend(image_dir.glob(ext))
        image_files.extend([f for f in image_dir.glob('*') if f.suffix == ''])  # 无扩展名的DICOM

        print(f"\n{'=' * 60}")
        print(f"方法1测试 - 批量处理 {len(image_files)} 张图像")
        print(f"{'=' * 60}")

        results = []
        for image_path in tqdm(image_files, desc="处理图像"):
            if specific_position is not None:
                result = self.test_method1_single_image(image_path, specific_position)
            else:
                result = self.test_method1_single_image(image_path)
            if result:
                results.append(result)

        # 保存结果汇总
        if results:
            self._save_method1_summary(results)

        return results

    def _save_method1_summary(self, results):
        """保存方法1的测试结果汇总"""
        df = pd.DataFrame(results)
        df.to_csv(self.method1_dir / 'test_results.csv', index=False)

        # 统计信息
        if df['height_error'].notna().any():
            stats = {
                'total': len(df),
                'mean_height_error': df['height_error'].mean(),
                'std_height_error': df['height_error'].std(),
                'mean_width_error': df['width_error'].mean(),
                'std_width_error': df['width_error'].std(),
                'max_height_error': df['height_error'].max(),
                'max_width_error': df['width_error'].max()
            }

            # 按位置统计
            position_stats = df.groupby('position_name').agg({
                'height_error': ['mean', 'std'],
                'width_error': ['mean', 'std']
            }).round(6)

            position_stats.to_csv(self.method1_dir / 'position_stats.csv')

            print(f"\n方法1测试统计:")
            print(f"  总图像数: {stats['total']}")
            print(f"  平均高度误差: {stats['mean_height_error']:.6f}")
            print(f"  平均宽度误差: {stats['mean_width_error']:.6f}")

    # ========== 测试方法2: 遍历所有位置类别 ==========

    def test_method2_position_scan(self, image_path: Path):
        """
        方法2: 遍历所有位置类别生成对比图像

        Args:
            image_path: 输入图像路径
        """
        filename = image_path.name
        basename = os.path.splitext(filename)[0]

        print(f"\n方法2测试 - 遍历位置类别: {filename}")

        # 提取病历号作为标题
        match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
        case_id = match.group(0) if match else basename

        # 加载图像
        image = self._load_image(image_path)
        if image is None:
            print(f"  图像加载失败")
            return None

        # 获取真实位置信息（如果有）
        true_position_idx = None
        if self.position_loader:
            _, true_position_idx, true_position_name = self.position_loader.get_position_for_image(filename)
            print(f"  真实位置: {true_position_name}")

        # 遍历所有位置类别 (0-5)
        results = []
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for pos_idx in range(6):
            # 预测
            height_pred, width_pred, attention_mask = self._predict(image, pos_idx)

            # 创建叠加图像
            color = self.class_colors[pos_idx]
            overlay_image = self._draw_rectangle_on_image(image, attention_mask, color)

            # 显示
            ax = axes[pos_idx]
            ax.imshow(overlay_image)

            # 标题包含位置信息和预测值
            is_true = (true_position_idx == pos_idx) if true_position_idx is not None else False
            title = f'{self.class_names[pos_idx]}\n高度: {height_pred:.3f}, 宽度: {width_pred:.3f}'
            if is_true:
                title += '\n✓ 真实位置'
                # 高亮真实位置的子图边框
                for spine in ax.spines.values():
                    spine.set_color('green')
                    spine.set_linewidth(3)
            else:
                for spine in ax.spines.values():
                    spine.set_color('gray')
                    spine.set_linewidth(1)

            ax.set_title(title, fontsize=10)
            ax.axis('off')

            # 记录结果
            results.append({
                'position_idx': pos_idx,
                'position_name': self.class_names[pos_idx],
                'height_pred': height_pred,
                'width_pred': width_pred,
                'is_true_position': is_true
            })

        # 设置总标题 - 使用病历号
        plt.suptitle(f"病历号: {case_id} - 不同位置类别的预测结果", fontsize=14, y=0.98)
        plt.tight_layout()

        # 保存图像
        save_path = self.method2_dir / f"{basename}_position_scan.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  对比图像保存到: {save_path}")

        # 打印预测结果
        print("  预测结果:")
        for r in results:
            marker = "✓" if r['is_true_position'] else " "
            print(f"    {marker} {r['position_name']}: 高度={r['height_pred']:.4f}, 宽度={r['width_pred']:.4f}")

        return results

    def test_method2_batch(self, image_dir: str):
        """
        批量测试方法2

        Args:
            image_dir: 图像目录
        """
        image_dir = Path(image_dir)
        image_files = []

        for ext in ['*.dcm', '*.png', '*.jpg', '*.jpeg']:
            image_files.extend(image_dir.glob(ext))
        image_files.extend([f for f in image_dir.glob('*') if f.suffix == ''])

        print(f"\n{'=' * 60}")
        print(f"方法2测试 - 批量处理 {len(image_files)} 张图像")
        print(f"{'=' * 60}")

        all_results = []
        for image_path in tqdm(image_files, desc="处理图像"):
            results = self.test_method2_position_scan(image_path)
            if results:
                all_results.extend(results)

        # 保存汇总
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(self.method2_dir / 'position_scan_results.csv', index=False)

            # 统计不同位置的预测差异
            position_summary = df.groupby('position_name').agg({
                'height_pred': ['mean', 'std'],
                'width_pred': ['mean', 'std']
            }).round(4)

            position_summary.to_csv(self.method2_dir / 'position_summary.csv')

            print(f"\n方法2测试完成，结果保存到: {self.method2_dir}")


# ==============================
# 主测试函数
# ==============================

def main():
    """主测试函数"""
    print("=" * 60)
    print("CoarseInfoExtractor 模型测试")
    print("包含两种测试方法")
    print("=" * 60)

    # 配置参数
    config = {
        'image_size': (512, 512),
        'base_channels': 32,
        'dropout_rate': 0.2,
        'location_excel_path': "D:/med_data/ai/location3.xlsx",  # 可选
        'position_excel_path': "D:/med_data/ai/classify1.xlsx",  # 可选
    }

    # 模型路径 - 请修改为您的模型路径
    model_path = "D:/med_data/ai/pre_loc/20260221_517/checkpoints/best_model_epoch_13.pth"

    # 测试图像目录
    test_image_dir = "D:/med_data/ai/test1"

    # 初始化测试器
    tester = ModelTester(model_path, config)

    # ========== 方法1测试: 直接读取位置信息 ==========
    print("\n" + "=" * 60)
    print("执行测试方法1: 直接读取位置信息")
    print("=" * 60)

    """ 
    # 方式1a: 批量测试，使用图像自带的真实位置
    tester.test_method1_batch(test_image_dir)

    
    # 方式1b: 测试单张图像，指定特定位置
    # 获取第一张图像进行单张测试
    image_dir = Path(test_image_dir)
    test_images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.dcm"))
    if test_images:
        print("\n" + "=" * 60)
        print("执行单张图像测试（指定位置为C1）")
        print("=" * 60)
        tester.test_method1_single_image(test_images[0], position_idx=0)
    """

    # ========== 方法2测试: 遍历所有位置类别 ==========
    print("\n" + "=" * 60)
    print("执行测试方法2: 遍历所有位置类别")
    print("=" * 60)

    # 方法2: 遍历所有位置
    tester.test_method2_batch(test_image_dir)

    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"所有结果保存到: {tester.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()