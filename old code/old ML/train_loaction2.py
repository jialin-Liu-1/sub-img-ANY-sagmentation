import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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
# 1. 模型定义 (保持不变)
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

        # ========== Position information processing path ==========
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

        # Process position information
        if position_info.dim() == 1 or position_info.shape[1] == 1:
            indices = position_info.long().view(-1)
            position_onehot = F.one_hot(indices, num_classes=self.num_position_classes).float()
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
        self.min_radius_ratio = min_radius_ratio  # 最小半径比例限制
        self.device = torch.device('cpu')

        #print(f"AttentionMaskGenerator初始化: 图像尺寸={image_size}, 最小半径比例={min_radius_ratio}")

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

            # Calculate window width (应用最小限制后的值)
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


# ==============================
# 2. 改进的数据加载和处理
# ==============================

class LocationInfoLoader:
    """Load position information from location.xlsx"""

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}
        self._load_location_info()

    def _load_location_info(self):
        """Load position information from Excel"""
        try:
            df = pd.read_excel(self.excel_path)
            print(f"Location Excel columns: {df.columns.tolist()}")

            # Standardize column names
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

            # Process data
            success_count = 0
            for _, row in df.iterrows():
                filename = str(row['filename']).strip()

                # Remove possible extensions
                filename = os.path.splitext(filename)[0]

                # Get height and width ratios
                try:
                    height_ratio = float(row['height_ratio'])
                    width_ratio = float(row['width_ratio'])

                    # Ensure values are in 0-1 range
                    height_ratio = max(0.0, min(1.0, height_ratio))
                    width_ratio = max(0.0, min(1.0, width_ratio))

                    self.location_dict[filename] = {
                        'height_ratio': height_ratio,
                        'width_ratio': width_ratio
                    }
                    success_count += 1
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue

            print(f"Successfully loaded {success_count} position records from {len(df)} total rows")

        except Exception as e:
            print(f"Failed to load position information: {e}")
            raise

    def get_location_for_image(self, filename: str):
        """Get position information for an image"""
        # Try different filename formats
        basename = os.path.splitext(filename)[0]

        # Try direct match
        if basename in self.location_dict:
            return self.location_dict[basename]

        # Try to extract record number
        parts = basename.split('_')
        if len(parts) >= 2:
            # Try to match with the last part or combination
            for i in range(len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    partial_name = '_'.join(parts[i:j])
                    for key in self.location_dict.keys():
                        if partial_name in key:
                            return self.location_dict[key]

        # Return default values
        print(f"Warning: No location information found for image {filename}, using defaults")
        return {'height_ratio': 0.5, 'width_ratio': 0.3}


class PositionInfoLoader:
    """Load artery segment position classification information"""

    def __init__(self, excel_path: str = None, debug_mode: bool = True):
        self.position_dict = {}
        self.debug_mode = debug_mode

        # 更新分类名称映射
        self.position_names = {
            0: "Segment 1", 1: "Segment 2", 2: "Segment 4",
            3: "Segment 5", 4: "Segment 6", 5: "Segment 7"
        }

        # 创建原始分类到新索引的映射
        # 原始分类: 1,2,4,5,6,7 映射到新索引: 0,1,2,3,4,5
        self.original_to_new_mapping = {
            1: 0,  # 原始1 -> 新索引0
            2: 1,  # 原始2 -> 新索引1
            4: 2,  # 原始4 -> 新索引2
            5: 3,  # 原始5 -> 新索引3
            6: 4,  # 原始6 -> 新索引4
            7: 5  # 原始7 -> 新索引5
        }

        # 创建反向映射（用于显示）
        self.new_to_original_mapping = {v: k for k, v in self.original_to_new_mapping.items()}

        # 存储所有可能的键，用于调试
        self.all_keys = []

        if excel_path:
            self._load_position_info(excel_path)

    def _load_position_info(self, excel_path: str):
        """Load position classification information"""
        try:
            df = pd.read_excel(excel_path)
            print(f"\nPosition classification Excel columns: {df.columns.tolist()}")
            print(f"First 5 rows:\n{df.head()}")

            # Assuming first column is filename, second column is position
            valid_count = 0
            invalid_count = 0
            position_stats = {}

            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()

                # Extract base filename (remove extension if any)
                base_name = os.path.splitext(filename)[0]
                self.all_keys.append(base_name)

                # Get position number
                try:
                    original_position = int(row.iloc[1]) if not pd.isna(row.iloc[1]) else None

                    # 检查原始分类是否在允许的范围内 (1,2,4,5,6,7)
                    if original_position in self.original_to_new_mapping:
                        # 映射到新的索引（0-5）
                        new_position = self.original_to_new_mapping[original_position]
                        self.position_dict[base_name] = {
                            'original': original_position,
                            'mapped': new_position
                        }
                        valid_count += 1

                        # 统计原始位置分布
                        if original_position not in position_stats:
                            position_stats[original_position] = 0
                        position_stats[original_position] += 1
                    else:
                        if self.debug_mode:
                            print(f"Debug: Skipping invalid position {original_position} for file {filename}")
                        invalid_count += 1

                except Exception as e:
                    if self.debug_mode:
                        print(f"Debug: Error processing row for {filename}: {e}")
                    invalid_count += 1
                    continue

            print(f"\nPosition loading summary:")
            print(f"  Successfully loaded: {valid_count} valid position records")
            print(f"  Skipped: {invalid_count} invalid records")
            print(f"  Position distribution (original values): {position_stats}")

            # 显示映射后的分布
            if self.position_dict:
                mapped_positions = [v['mapped'] for v in self.position_dict.values()]
                distribution = pd.Series(mapped_positions).value_counts().sort_index()

                print("\nMapped Position Distribution:")
                for new_idx, count in distribution.items():
                    original_val = self.new_to_original_mapping[new_idx]
                    print(f"  Original {original_val} -> New Index {new_idx}: {count} samples")

        except Exception as e:
            print(f"Failed to load position classification information: {e}")
            # Use default values if no classification information
            pass

    def get_position_for_image(self, filename: str):
        """Get position classification for an image (returns 8-dimensional one-hot vector)"""
        basename = os.path.splitext(filename)[0]

        # Try different matching strategies with debug output
        position_data = self._find_position_in_dict(basename)

        if position_data is not None:
            # 获取映射后的新索引（0-5）
            new_position_num = position_data['mapped']
            original_position = position_data['original']

            # 创建8维的one-hot向量（只使用前6个位置）
            position_tensor = torch.zeros(8)
            position_tensor[new_position_num] = 1.0

            position_name = self.position_names.get(new_position_num, f"Segment {original_position}")

            #if self.debug_mode and new_position_num != original_position:
            #print(f"Debug: Mapped {filename}: original={original_position} -> new={new_position_num}")

            return position_tensor, new_position_num, position_name

        # 如果没有找到对应的位置信息，返回默认值
        if self.debug_mode:
            print(f"Debug: No position found for '{basename}', using default (Segment 4)")

        # 默认使用新索引2（对应原始分类4）
        default_new_idx = 2
        position_tensor = torch.zeros(8)
        position_tensor[default_new_idx] = 1.0
        return position_tensor, default_new_idx, "Segment 4 (default)"

    def _find_position_in_dict(self, basename: str):
        """Enhanced method to find position data in dictionary with multiple matching strategies"""

        # Strategy 1: Direct match
        if basename in self.position_dict:
            #if self.debug_mode:
            #print(f"Debug: Direct match found for '{basename}'")
            return self.position_dict[basename]

        # Strategy 2: Try to match with different parts of the filename
        parts = basename.split('_')

        # Try full basename without numbers at the end
        for i in range(len(parts), 0, -1):
            test_name = '_'.join(parts[:i])
            if test_name in self.position_dict:
                #if self.debug_mode:
                #print(f"Debug: Partial match found: '{test_name}' for '{basename}'")
                return self.position_dict[test_name]

        # Strategy 3: Try to match with record number (e.g., "001" from "ANY_001_0")
        for part in parts:
            if part.isdigit() and len(part) >= 3:
                for key in self.position_dict.keys():
                    if part in key:
                        #if self.debug_mode:
                              #print(f"Debug: Number match found: part='{part}' in key='{key}' for '{basename}'")
                        return self.position_dict[key]

        # Strategy 4: Try to match with any combination
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                test_name = '_'.join(parts[i:j])
                for key in self.position_dict.keys():
                    if test_name in key:
                        #if self.debug_mode:
                        #print(f"Debug: Substring match: '{test_name}' in '{key}' for '{basename}'")
                        return self.position_dict[key]

        return None

    def get_original_position(self, mapped_position: int) -> int:
        """Get original position value from mapped index"""
        return self.new_to_original_mapping.get(mapped_position, mapped_position)


class CoarseExtractorDataset(Dataset):
    """CoarseInfoExtractor training dataset"""

    def __init__(self,
                 image_dir: str,
                 location_excel_path: str,
                 position_excel_path: str = None,
                 image_size: Tuple[int, int] = (512, 512),
                 max_samples: int = None,
                 debug_mode: bool = True):

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.debug_mode = debug_mode

        # Load position information
        self.location_loader = LocationInfoLoader(location_excel_path)

        # Load position classification information
        self.position_loader = PositionInfoLoader(position_excel_path, debug_mode=debug_mode)

        # Get image file list
        self.image_files = self._get_image_files()

        if max_samples:
            self.image_files = self.image_files[:max_samples]

        print(f"\nDataset initialized with {len(self.image_files)} samples")
        print(f"Image directory: {image_dir}")

        # Verify position information usage
        self._verify_position_usage()

    def _verify_position_usage(self):
        """Verify that position information is correctly loaded"""
        print("\n" + "=" * 60)
        print("VERIFYING POSITION INFORMATION USAGE")
        print("=" * 60)

        found_count = 0
        default_count = 0

        if len(self.image_files) > 0:
            # Check a few samples
            for i in range(min(10, len(self.image_files))):
                filename = self.image_files[i]
                try:
                    # Get position information
                    position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)

                    # Check if default was used
                    if position_name == "Segment 4 (default)":
                        default_count += 1
                        status = "⚠️ DEFAULT"
                    else:
                        found_count += 1
                        status = "✓ FOUND"

                    # Get original position (if available)
                    try:
                        original_position = self.position_loader.get_original_position(position_num)
                        position_display = f"{position_name} (original: {original_position})"
                    except:
                        position_display = position_name

                    # Get location information
                    location_info = self.location_loader.get_location_for_image(filename)

                    print(f"Sample {i + 1}: {filename} {status}")
                    print(f"  Position: {position_display} (new index {position_num})")
                    print(
                        f"  Location: height={location_info['height_ratio']:.3f}, width={location_info['width_ratio']:.3f}")
                    print(f"  Position tensor: {[f'{x:.1f}' for x in position_tensor.tolist()]}")
                    print()

                except Exception as e:
                    print(f"Error verifying sample {filename}: {e}")

        print(f"Verification summary: {found_count} found, {default_count} defaults")
        print("=" * 60)

    def _get_image_files(self):
        """Get image file list"""
        image_files = []

        # Support no-extension files and common image formats
        valid_extensions = {'.dcm', '.dicom', '', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}

        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in valid_extensions or file_path.suffix == '':
                    image_files.append(file_path.name)

        return sorted(image_files)  # Sort for reproducibility

    def _load_image(self, filename: str) -> np.ndarray:
        """Load image"""
        file_path = self.image_dir / filename

        try:
            # Try to load DICOM file
            if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(str(file_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                # Load regular image
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Cannot load image: {file_path}")
                image = image.astype(np.float32)

            # Normalize
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # Resize
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"Failed to load image {filename}: {e}")
            return np.zeros(self.image_size, dtype=np.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        try:
            # Load image
            image = self._load_image(filename)
            image = np.expand_dims(image, axis=0)  # Add channel dimension
            image_tensor = torch.from_numpy(image).float()

            # Get position information (training target)
            location_info = self.location_loader.get_location_for_image(filename)
            height_ratio = torch.tensor(location_info['height_ratio'], dtype=torch.float32)
            width_ratio = torch.tensor(location_info['width_ratio'], dtype=torch.float32)

            # Get position classification information (model input)
            position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)

            return {
                'image': image_tensor,
                'position': position_tensor,
                'height_ratio': height_ratio,
                'width_ratio': width_ratio,
                'filename': filename,
                'position_name': position_name,
                'position_num': position_num
            }

        except Exception as e:
            print(f"Error processing sample {filename}: {e}")
            # Return default tensors
            return {
                'image': torch.zeros((1, *self.image_size), dtype=torch.float32),
                'position': torch.zeros(8, dtype=torch.float32),
                'height_ratio': torch.tensor(0.5, dtype=torch.float32),
                'width_ratio': torch.tensor(0.3, dtype=torch.float32),
                'filename': filename,
                'position_name': "Segment 4 (default)",
                'position_num': 4
            }


# ==============================
# 3. 训练器（修改为使用自动划分的测试集）
# ==============================

class EarlyStopping:
    """Early stopping mechanism"""

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
    """CoarseInfoExtractor trainer with automatic train/val/test split"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"D:/med_data/ai/coarse_extractor_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        self.test_results_dir = self.output_dir / "test_results"

        for dir_path in [self.checkpoint_dir, self.results_dir, self.plots_dir, self.test_results_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"Output directory: {self.output_dir}")

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_height_loss': [],
            'train_width_loss': [],
            'val_height_loss': [],
            'val_width_loss': [],
            'learning_rate': []
        }

        # Test set indices (will be set during dataset creation)
        self.test_indices = None
        self.test_dataset = None

    def _create_model(self):
        """Create model"""
        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=self.config['num_position_classes'],
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
        ).to(self.device)

        return model

    def _create_datasets_with_split(self):
        """Create datasets with automatic train/val/test split"""

        # Load full dataset
        full_dataset = CoarseExtractorDataset(
            image_dir=self.config['image_dir'],  # Use single image directory
            location_excel_path=self.config['location_excel_path'],
            position_excel_path=self.config.get('position_excel_path'),
            image_size=self.config['image_size'],
            max_samples=self.config.get('max_samples'),
            debug_mode=self.config.get('debug_mode', True)
        )

        # Calculate split sizes
        total_size = len(full_dataset)
        test_size = int(total_size * self.config.get('test_split', 0.1))
        val_size = int(total_size * self.config.get('val_split', 0.1))
        train_size = total_size - test_size - val_size

        print(f"\nDataset split:")
        print(f"  Total samples: {total_size}")
        print(f"  Train: {train_size} ({train_size / total_size:.1%})")
        print(f"  Validation: {val_size} ({val_size / total_size:.1%})")
        print(f"  Test: {test_size} ({test_size / total_size:.1%})")

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        # Store test dataset for later use
        self.test_dataset = test_dataset
        self.test_indices = test_dataset.indices

        return train_dataset, val_dataset, test_dataset

    def _create_dataloaders(self):
        """Create data loaders with automatic split"""

        train_dataset, val_dataset, test_dataset = self._create_datasets_with_split()

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

        # Save test set information
        test_info = {
            'total_test_samples': len(test_dataset),
            'test_indices': self.test_indices
        }

        with open(self.output_dir / 'test_set_info.json', 'w') as f:
            json.dump(test_info, f, indent=4, default=str)

        print(f"\nTraining set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples (saved for final evaluation)")

        return train_loader, val_loader, test_dataset

    def train(self):
        """Train model"""
        print("\n" + "=" * 60)
        print("STARTING CoarseInfoExtractor TRAINING")
        print("=" * 60)

        # Create model
        model = self._create_model()

        # Create data loaders
        train_loader, val_loader, test_dataset = self._create_dataloaders()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=self.config.get('early_stopping_min_delta', 0.001)
        )

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        best_model_path = None

        for epoch in range(self.config['num_epochs']):
            # Training phase
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

                # Forward pass
                outputs = model(images, positions)
                height_pred = outputs['height_ratio']
                width_pred = outputs['width_ratio']

                # Calculate loss
                height_loss = criterion(height_pred, height_targets)
                width_loss = criterion(width_pred, width_targets)
                loss = height_loss + width_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Record loss
                train_loss_total += loss.item()
                train_loss_height += height_loss.item()
                train_loss_width += width_loss.item()

                # Update progress bar
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'height': f'{height_loss.item():.4f}',
                    'width': f'{width_loss.item():.4f}'
                })

            # Validation phase
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

                    # Forward pass
                    outputs = model(images, positions)
                    height_pred = outputs['height_ratio']
                    width_pred = outputs['width_ratio']

                    # Calculate loss
                    height_loss = criterion(height_pred, height_targets)
                    width_loss = criterion(width_pred, width_targets)
                    loss = height_loss + width_loss

                    # Record loss
                    val_loss_total += loss.item()
                    val_loss_height += height_loss.item()
                    val_loss_width += width_loss.item()

                    val_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'height': f'{height_loss.item():.4f}',
                        'width': f'{width_loss.item():.4f}'
                    })

            # Calculate average losses
            train_loss_avg = train_loss_total / len(train_loader)
            val_loss_avg = val_loss_total / len(val_loader)

            train_height_avg = train_loss_height / len(train_loader)
            train_width_avg = train_loss_width / len(train_loader)
            val_height_avg = val_loss_height / len(val_loader)
            val_width_avg = val_loss_width / len(val_loader)

            # Update learning rate
            scheduler.step(val_loss_avg)

            # Record history
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss_avg)
            self.history['train_height_loss'].append(train_height_avg)
            self.history['train_width_loss'].append(train_width_avg)
            self.history['val_height_loss'].append(val_height_avg)
            self.history['val_width_loss'].append(val_width_avg)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Print statistics
            print(f"\nEpoch {epoch + 1} Statistics:")
            print(f"  Train Loss: {train_loss_avg:.4f} (Height: {train_height_avg:.4f}, Width: {train_width_avg:.4f})")
            print(f"  Val Loss: {val_loss_avg:.4f} (Height: {val_height_avg:.4f}, Width: {val_width_avg:.4f})")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
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

                print(f"  ✓ Saved best model to: {best_model_path}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint to: {checkpoint_path}")

            # Early stopping check
            if early_stopping(val_loss_avg):
                print(f"\n🚨 Early stopping triggered! No improvement for {early_stopping.patience} epochs")
                break

        # Save final model
        final_model_path = self.checkpoint_dir / 'final_model.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': best_val_loss
        }, final_model_path)

        print(f"\nTraining completed! Trained for {epoch + 1} epochs")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final model saved to: {final_model_path}")

        # Plot training curves
        self._plot_training_history()

        # Save training history
        self._save_training_history()

        # Run test on the held-out test set
        self._run_test_on_held_out_set(model, test_dataset)

        return final_model_path

    def _run_test_on_held_out_set(self, model, test_dataset):
        """Run evaluation on the held-out test set"""
        print("\n" + "=" * 60)
        print("EVALUATING ON HELD-OUT TEST SET")
        print("=" * 60)

        # Create tester and run evaluation
        tester = CoarseExtractorTester(
            model=model,
            config=self.config,
            output_dir=self.test_results_dir,
            device=self.device
        )

        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )

        # Run evaluation
        test_results = tester.evaluate(test_loader)

        print(f"\nTest evaluation completed. Results saved to: {self.test_results_dir}")

    def _plot_training_history(self):
        """Plot training history curves"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Total Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Height Loss
        axes[0, 1].plot(epochs, self.history['train_height_loss'], 'b-', label='Train Height Loss')
        axes[0, 1].plot(epochs, self.history['val_height_loss'], 'r-', label='Val Height Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Height Ratio Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Width Loss
        axes[1, 0].plot(epochs, self.history['train_width_loss'], 'b-', label='Train Width Loss')
        axes[1, 0].plot(epochs, self.history['val_width_loss'], 'r-', label='Val Width Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Width Ratio Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {self.plots_dir / 'training_history.png'}")

    def _save_training_history(self):
        """Save training history"""
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

        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, default=str)

        print(f"Training history saved to: {self.output_dir / 'training_history.csv'}")


# ==============================
# 4. 改进的测试器
# ==============================

class CoarseExtractorTester:
    """CoarseInfoExtractor tester for held-out test sets"""

    def __init__(self, model=None, model_path=None, config=None, output_dir=None, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path and config:
            # Load model from file
            self.model = self._load_model(model_path)
        elif model:
            # Use provided model
            self.model = model
        else:
            raise ValueError("Either model or model_path must be provided")

        self.model.eval()

        # Create attention mask generator
        self.mask_generator = AttentionMaskGenerator(
            image_size=config['image_size']
        ).to(self.device)

        # Load position information
        self.location_loader = LocationInfoLoader(config['location_excel_path'])
        self.position_loader = PositionInfoLoader(config.get('position_excel_path'), debug_mode=False)

        # Set output directory
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
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

    def _load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=self.config['num_position_classes'],
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss')

        print(f"Model loaded successfully, training epochs: {epoch}")
        if isinstance(val_loss, (int, float)):
            print(f"Validation loss: {val_loss:.4f}")
        else:
            print(f"Validation loss: Unknown")

        return model

    def _load_aneurysm_mask(self, filename: str) -> Optional[np.ndarray]:
        """Load aneurysm mask image (TIF format)"""
        try:
            # Build mask file path
            basename = os.path.splitext(filename)[0]
            mask_dir = Path("D:/med_data/ai/test2")

            # Try different filename formats
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

            # Load mask image
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Try using PIL (for some TIF formats)
                from PIL import Image
                pil_image = Image.open(str(mask_path))
                mask = np.array(pil_image)

            if mask is None:
                return None

            # Binarize (ensure it's 0-1 mask)
            if mask.max() > 1.0:
                mask = mask / 255.0

            # Threshold to ensure binary mask
            mask = (mask > 0.5).astype(np.float32)

            # Resize to 512x512
            if mask.shape != self.config['image_size']:
                mask = cv2.resize(mask, self.config['image_size'], interpolation=cv2.INTER_NEAREST)

            return mask

        except Exception as e:
            return None

    def _save_image(self, image, path, normalize=True):
        """Save image"""
        if normalize and image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        cv2.imwrite(str(path), image)

    def _create_overlay_image(self, dsa_image: np.ndarray,
                              attention_mask: np.ndarray,
                              aneurysm_mask: np.ndarray,
                              attention_color: str = 'red') -> np.ndarray:
        """Create overlay image: DSA image + attention region + aneurysm mask (green)"""
        # Convert grayscale to RGB
        if len(dsa_image.shape) == 2:
            dsa_rgb = np.stack([dsa_image] * 3, axis=-1)
        else:
            dsa_rgb = dsa_image.copy()

        # Ensure values are in 0-1 range
        dsa_rgb = np.clip(dsa_rgb, 0, 1)

        # Create attention region overlay
        attention_overlay = np.zeros_like(dsa_rgb)
        if attention_color == 'red':
            attention_overlay[:, :, 0] = attention_mask * 0.5
        elif attention_color == 'blue':
            attention_overlay[:, :, 2] = attention_mask * 0.5

        # Create aneurysm mask overlay (green)
        aneurysm_overlay = np.zeros_like(dsa_rgb)
        aneurysm_overlay[:, :, 1] = aneurysm_mask

        # Overlay images
        overlay_image = dsa_rgb.copy()
        overlay_image = overlay_image * (1 - 0.3 * attention_mask[:, :, np.newaxis]) + attention_overlay * 0.3
        overlay_image = overlay_image + aneurysm_overlay * 0.7

        # Ensure values are in 0-1 range
        overlay_image = np.clip(overlay_image, 0, 1)

        return overlay_image

    def evaluate(self, test_loader):
        """Evaluate on test loader"""
        print("\nEvaluating on test set...")

        results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                images = batch['image'].to(self.device)
                positions = batch['position'].to(self.device)
                height_targets = batch['height_ratio'].to(self.device)
                width_targets = batch['width_ratio'].to(self.device)
                filenames = batch['filename']

                # Forward pass
                outputs = self.model(images, positions)
                height_pred = outputs['height_ratio']
                width_pred = outputs['width_ratio']

                # Calculate errors
                height_errors = (height_pred - height_targets).abs().cpu().numpy()
                width_errors = (width_pred - width_targets).abs().cpu().numpy()

                # Process each sample in batch
                for i, filename in enumerate(filenames):
                    result = {
                        'filename': filename,
                        'height_pred': height_pred[i].item(),
                        'width_pred': width_pred[i].item(),
                        'height_true': height_targets[i].item(),
                        'width_true': width_targets[i].item(),
                        'height_error': height_errors[i],
                        'width_error': width_errors[i],
                        'position_num': batch['position_num'][i].item() if 'position_num' in batch else None
                    }

                    # Generate visualizations for first few samples
                    if batch_idx < 5 and i < 2:  # Limit number of visualizations
                        self._generate_visualizations(
                            images[i].cpu().numpy().squeeze(),
                            height_pred[i].item(),
                            width_pred[i].item(),
                            height_targets[i].item(),
                            width_targets[i].item(),
                            filename,
                            batch['position_name'][i] if 'position_name' in batch else "Unknown"
                        )

                    results.append(result)

        # Save results
        self._save_test_results(results)

        return results

    def _generate_visualizations(self, image, height_pred, width_pred, height_true, width_true, filename,
                                 position_name):
        """Generate visualization for a single test sample"""
        basename = os.path.splitext(filename)[0]

        # Generate predicted attention mask
        height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
        width_tensor_pred = torch.tensor([width_pred], dtype=torch.float32).to(self.device)
        attention_mask_pred = self.mask_generator(height_tensor_pred, width_tensor_pred)
        attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

        # Generate ground truth attention mask
        height_tensor_gt = torch.tensor([height_true], dtype=torch.float32).to(self.device)
        width_tensor_gt = torch.tensor([width_true], dtype=torch.float32).to(self.device)
        attention_mask_gt = self.mask_generator(height_tensor_gt, width_tensor_gt)
        attention_mask_gt_np = attention_mask_gt.squeeze().cpu().numpy()

        # Load aneurysm mask (if exists)
        aneurysm_mask = self._load_aneurysm_mask(filename)

        # Save original image
        self._save_image(image, self.output_dir / f"{basename}_original.png")

        # Save attention masks
        mask_pred_uint8 = (attention_mask_pred_np * 255).astype(np.uint8)
        self._save_image(mask_pred_uint8, self.masks_dir / f"{basename}_pred_mask.png", normalize=False)

        mask_gt_uint8 = (attention_mask_gt_np * 255).astype(np.uint8)
        self._save_image(mask_gt_uint8, self.standard_masks_dir / f"{basename}_standard_mask.png", normalize=False)

        # Save focused images
        focused_image_pred = image * attention_mask_pred_np
        self._save_image(focused_image_pred, self.focused_dir / f"{basename}_pred_focused.png")

        # Create comparison figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(attention_mask_pred_np, cmap='gray')
        axes[1].set_title(f'Predicted Mask\nH:{height_pred:.3f}, W:{width_pred:.3f}')
        axes[1].axis('off')

        axes[2].imshow(attention_mask_gt_np, cmap='gray')
        axes[2].set_title(f'Ground Truth Mask\nH:{height_true:.3f}, W:{width_true:.3f}')
        axes[2].axis('off')

        if aneurysm_mask is not None:
            overlay_image = self._create_overlay_image(image, attention_mask_pred_np, aneurysm_mask)
            axes[3].imshow(overlay_image)
            axes[3].set_title('Overlay (Predicted)')
        else:
            axes[3].imshow(image, cmap='gray')
            axes[3].set_title('No Aneurysm Mask')
        axes[3].axis('off')

        plt.suptitle(f"{filename}\nPosition: {position_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.comparison_dir / f"{basename}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_test_results(self, results):
        """Save test results"""
        # Save as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        # Calculate statistics
        height_errors = results_df['height_error'].values
        width_errors = results_df['width_error'].values

        stats = {
            'total_samples': len(results),
            'mean_height_error': float(np.mean(height_errors)),
            'std_height_error': float(np.std(height_errors)),
            'mean_width_error': float(np.mean(width_errors)),
            'std_width_error': float(np.std(width_errors)),
            'max_height_error': float(np.max(height_errors)),
            'max_width_error': float(np.max(width_errors)),
            'min_height_error': float(np.min(height_errors)),
            'min_width_error': float(np.min(width_errors)),
            'median_height_error': float(np.median(height_errors)),
            'median_width_error': float(np.median(width_errors))
        }

        # Save statistics
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(self.output_dir / 'test_statistics.csv', index=False)

        # Save as text file
        with open(self.output_dir / 'test_summary.txt', 'w', encoding='utf-8') as f:
            f.write("TEST RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples tested: {stats['total_samples']}\n\n")

            f.write("Height Ratio Error Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Mean: {stats['mean_height_error']:.6f}\n")
            f.write(f"  Std Dev: {stats['std_height_error']:.6f}\n")
            f.write(f"  Median: {stats['median_height_error']:.6f}\n")
            f.write(f"  Max: {stats['max_height_error']:.6f}\n")
            f.write(f"  Min: {stats['min_height_error']:.6f}\n\n")

            f.write("Width Ratio Error Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Mean: {stats['mean_width_error']:.6f}\n")
            f.write(f"  Std Dev: {stats['std_width_error']:.6f}\n")
            f.write(f"  Median: {stats['median_width_error']:.6f}\n")
            f.write(f"  Max: {stats['max_width_error']:.6f}\n")
            f.write(f"  Min: {stats['min_width_error']:.6f}\n")

        print(f"\nTest Results Statistics:")
        print(f"  Mean Height Error: {stats['mean_height_error']:.6f}")
        print(f"  Mean Width Error: {stats['mean_width_error']:.6f}")
        print(f"  Test results saved to: {self.output_dir}")


# ==============================
# 5. 主程序
# ==============================

def main():
    """Main function"""
    print("CoarseInfoExtractor Training and Testing Program")
    print("=" * 60)

    # Configuration parameters
    config = {
        # Data paths - now using single directory
        'image_dir': "D:/med_data/ai/train11",  # Single directory containing all images
        'location_excel_path': "D:/med_data/ai/location3.xlsx",
        'position_excel_path': "D:/med_data/ai/classify1.xlsx",  # Artery segment information

        # Dataset split ratios
        'test_split': 0.1,  # 10% for testing
        'val_split': 0.1,  # 10% for validation
        # Train will be the remaining 80%

        # Model parameters
        'image_size': (512, 512),
        'base_channels': 32,
        'num_position_classes': 8,
        'dropout_rate': 0.2,

        # Training parameters
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,

        # Early stopping parameters
        'early_stopping_patience': 6,
        'early_stopping_min_delta': 0.0003,

        # Other parameters
        'num_workers': 2,
        'max_samples': None,  # Limit number of samples for debugging
        'debug_mode': True,  # Enable debug output for position matching
    }

    print("Configuration Parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # Train model (this will automatically split data and test on held-out set)
    print("\nStep 1: Training CoarseInfoExtractor Model with Automatic Train/Val/Test Split")
    trainer = CoarseExtractorTrainer(config)
    trained_model_path = trainer.train()

    print("\n" + "=" * 60)
    print("Program Completed!")
    print(f"Training output directory: {trainer.output_dir}")
    print(f"Test results directory: {trainer.test_results_dir}")


if __name__ == "__main__":
    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run main program
    main()