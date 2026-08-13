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
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue

            print(f"Successfully loaded {len(self.location_dict)} position records")

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
            record_num = parts[1]
            for key in self.location_dict.keys():
                if record_num in key:
                    return self.location_dict[key]

        # Return default values
        print(f"Warning: No position information found for image {filename}, using defaults")
        return {'height_ratio': 0.5, 'width_ratio': 0.3}

class PositionInfoLoader:
    """Load artery segment position classification information"""

    def __init__(self, excel_path: str = None):
        self.position_dict = {}
        self.position_names = {
            0: "Segment 1", 1: "Segment 2", 2: "Segment 3", 3: "Segment 4",
            4: "Segment 5", 5: "Segment 6", 6: "Segment 7", 7: "Segment 8"
        }
        if excel_path:
            self._load_position_info(excel_path)

    def _load_position_info(self, excel_path: str):
        """Load position classification information"""
        try:
            df = pd.read_excel(excel_path)
            print(f"Position classification Excel columns: {df.columns.tolist()}")
            print(f"First 5 rows:\n{df.head()}")

            # Assuming first column is filename, second column is position
            for _, row in df.iterrows():
                filename = str(row.iloc[0]).strip()

                # Extract base filename (remove extension if any)
                base_name = os.path.splitext(filename)[0]

                # Get position number
                try:
                    position_num = int(row.iloc[1]) if not pd.isna(row.iloc[1]) else 0

                    # Ensure position number is in 0-7 range
                    position_num = max(0, min(7, position_num))

                    self.position_dict[base_name] = position_num

                except Exception as e:
                    print(f"Error processing row for {filename}: {e}")
                    continue

            print(f"Successfully loaded {len(self.position_dict)} position classification records")
            print(f"Position distribution: {pd.Series(list(self.position_dict.values())).value_counts().to_dict()}")

        except Exception as e:
            print(f"Failed to load position classification information: {e}")
            # Use default values if no classification information
            pass

    def get_position_for_image(self, filename: str):
        """Get position classification for an image"""
        basename = os.path.splitext(filename)[0]

        # Try different matching strategies
        # 1. Direct match
        if basename in self.position_dict:
            position_num = self.position_dict[basename]
            position_tensor = torch.zeros(8)
            position_tensor[position_num] = 1.0
            position_name = self.position_names.get(position_num, f"Segment {position_num}")
            return position_tensor, position_num, position_name

        # 2. Try to extract record number
        parts = basename.split('_')
        if len(parts) >= 2:
            # Try pattern like "ANY_450_0"
            if parts[0].isalpha() and len(parts) >= 3:
                search_name = f"{parts[0]}_{parts[1]}"
                for key in self.position_dict.keys():
                    if search_name in key:
                        position_num = self.position_dict[key]
                        position_tensor = torch.zeros(8)
                        position_tensor[position_num] = 1.0
                        position_name = self.position_names.get(position_num, f"Segment {position_num}")
                        return position_tensor, position_num, position_name

        # 3. Try numeric extraction
        match = re.search(r'(\d+)', basename)
        if match:
            record_num = match.group(1)
            for key in self.position_dict.keys():
                if record_num in key:
                    position_num = self.position_dict[key]
                    position_tensor = torch.zeros(8)
                    position_tensor[position_num] = 1.0
                    position_name = self.position_names.get(position_num, f"Segment {position_num}")
                    return position_tensor, position_num, position_name

        # Return default position (middle position)
        print(f"Warning: No position classification found for {filename}, using default (Segment 4)")
        position_tensor = torch.zeros(8)
        position_tensor[4] = 1.0  # Default to middle position
        return position_tensor, 4, "Segment 4 (default)"

class CoarseExtractorDataset(Dataset):
    """CoarseInfoExtractor training dataset"""

    def __init__(self,
                 image_dir: str,
                 location_excel_path: str,
                 position_excel_path: str = None,
                 image_size: Tuple[int, int] = (512, 512),
                 max_samples: int = None):

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Load position information
        self.location_loader = LocationInfoLoader(location_excel_path)

        # Load position classification information
        self.position_loader = PositionInfoLoader(position_excel_path)

        # Get image file list
        self.image_files = self._get_image_files()

        if max_samples:
            self.image_files = self.image_files[:max_samples]

        print(f"Dataset initialized with {len(self.image_files)} samples")

        # Verify position information usage
        self._verify_position_usage()

    def _verify_position_usage(self):
        """Verify that position information is correctly loaded"""
        print("\n" + "=" * 60)
        print("VERIFYING POSITION INFORMATION USAGE")
        print("=" * 60)

        if len(self.image_files) > 0:
            # Check a few samples
            for i in range(min(5, len(self.image_files))):
                filename = self.image_files[i]
                try:
                    # Get position information
                    position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)

                    # Get location information
                    location_info = self.location_loader.get_location_for_image(filename)

                    print(f"Sample {i + 1}: {filename}")
                    print(f"  Position: {position_name} (index {position_num})")
                    print(
                        f"  Location: height={location_info['height_ratio']:.3f}, width={location_info['width_ratio']:.3f}")
                    print(f"  Position tensor: {position_tensor.tolist()}")
                    print()

                except Exception as e:
                    print(f"Error verifying sample {filename}: {e}")

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

        return image_files

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
    """CoarseInfoExtractor trainer"""

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

        for dir_path in [self.checkpoint_dir, self.results_dir, self.plots_dir]:
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

    def _create_dataloaders(self):
        """Create data loaders"""
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

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")

        return train_loader, val_loader

    def train(self):
        """Train model"""
        print("\n" + "=" * 60)
        print("STARTING CoarseInfoExtractor TRAINING")
        print("=" * 60)

        # Create model
        model = self._create_model()

        # Create data loaders
        train_loader, val_loader = self._create_dataloaders()

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

        return final_model_path

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
# 4. 测试和推理
# ==============================

class CoarseExtractorTester:
    """CoarseInfoExtractor tester"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Create attention mask generator
        self.mask_generator = AttentionMaskGenerator(
            image_size=config['image_size']
        ).to(self.device)

        # Load position information
        self.location_loader = LocationInfoLoader(config['location_excel_path'])
        self.position_loader = PositionInfoLoader(config.get('position_excel_path'))

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"D:/med_data/ai/test_results_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.masks_dir = self.output_dir / "masks"
        self.focused_dir = self.output_dir / "focused_images"
        self.comparison_dir = self.output_dir / "comparison"
        self.overlay_dir = self.output_dir / "overlay"
        self.gt_overlay_dir = self.output_dir / "gt_overlay"  # Ground truth overlay directory
        self.standard_masks_dir = self.output_dir / "standard_masks"  # Standard masks directory

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

        # Safely print training information
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss')

        print(f"Model loaded successfully, training epochs: {epoch}")
        if isinstance(val_loss, (int, float)):
            print(f"Validation loss: {val_loss:.4f}")
        else:
            print(f"Validation loss: Unknown")

        return model

    def _load_image(self, image_path):
        """Load image"""
        try:
            # Try to load DICOM file
            if image_path.suffix == '' or image_path.suffix.lower() in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(str(image_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                # Load regular image
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Cannot load image: {image_path}")
                image = image.astype(np.float32)

            # Normalize
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # Resize
            if image.shape != self.config['image_size']:
                image = cv2.resize(image, self.config['image_size'], interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"Failed to load image {image_path.name}: {e}")
            return None

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
                print(f"Warning: No aneurysm mask found for {filename}")
                return None

            # Load mask image
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Try using PIL (for some TIF formats)
                from PIL import Image
                pil_image = Image.open(str(mask_path))
                mask = np.array(pil_image)

            if mask is None:
                print(f"Warning: Cannot load mask image: {mask_path}")
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
            print(f"Failed to load aneurysm mask {filename}: {e}")
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
            attention_overlay[:, :, 0] = attention_mask * 0.5  # Red channel
        elif attention_color == 'blue':
            attention_overlay[:, :, 2] = attention_mask * 0.5  # Blue channel

        # Create aneurysm mask overlay (green)
        aneurysm_overlay = np.zeros_like(dsa_rgb)
        aneurysm_overlay[:, :, 1] = aneurysm_mask  # Green channel

        # Overlay images
        # 1. Base DSA image
        overlay_image = dsa_rgb.copy()

        # 2. Overlay attention region (semi-transparent)
        overlay_image = overlay_image * (1 - 0.3 * attention_mask[:, :, np.newaxis]) + attention_overlay * 0.3

        # 3. Overlay aneurysm mask (green)
        overlay_image = overlay_image + aneurysm_overlay * 0.7

        # Ensure values are in 0-1 range
        overlay_image = np.clip(overlay_image, 0, 1)

        return overlay_image

    def test_single_image(self, image_path):
        """Test single image"""
        filename = image_path.name
        basename = os.path.splitext(filename)[0]

        print(f"\nProcessing image: {filename}")

        # Load image
        image = self._load_image(image_path)
        if image is None:
            print(f"  Skipping image {filename}")
            return None

        # Get position information (artery segment)
        position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)

        print(f"  Artery Segment: {position_name}")

        # Prepare input
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        position_tensor = position_tensor.unsqueeze(0).to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(image_tensor, position_tensor)
            height_pred = outputs['height_ratio'].item()
            width_pred = outputs['width_ratio'].item()

        print(f"  Predicted - Height: {height_pred:.4f}, Width: {width_pred:.4f}")

        # Get ground truth position information (from location.xlsx)
        try:
            true_info = self.location_loader.get_location_for_image(filename)
            height_true = true_info['height_ratio']
            width_true = true_info['width_ratio']
            print(f"  Ground Truth - Height: {height_true:.4f}, Width: {width_true:.4f}")
            print(f"  Error - Height: {abs(height_pred - height_true):.4f}, Width: {abs(width_pred - width_true):.4f}")
        except:
            height_true = width_true = None

        # Generate predicted attention mask
        height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
        width_tensor_pred = torch.tensor([width_pred], dtype=torch.float32).to(self.device)

        attention_mask_pred = self.mask_generator(height_tensor_pred, width_tensor_pred)
        attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

        # Apply predicted attention mask
        focused_image_pred = image * attention_mask_pred_np

        # Generate standard attention mask (from ground truth)
        attention_mask_gt_np = None
        focused_image_gt = None
        if height_true is not None and width_true is not None:
            height_tensor_gt = torch.tensor([height_true], dtype=torch.float32).to(self.device)
            width_tensor_gt = torch.tensor([width_true], dtype=torch.float32).to(self.device)

            attention_mask_gt = self.mask_generator(height_tensor_gt, width_tensor_gt)
            attention_mask_gt_np = attention_mask_gt.squeeze().cpu().numpy()

            # Apply standard attention mask
            focused_image_gt = image * attention_mask_gt_np

        # Load aneurysm mask
        aneurysm_mask = self._load_aneurysm_mask(filename)
        has_aneurysm_mask = aneurysm_mask is not None

        # Calculate overlap between attention masks and aneurysm mask
        pred_overlap_metrics = None
        gt_overlap_metrics = None

        if has_aneurysm_mask:
            # Calculate overlap for predicted mask
            intersection_pred = np.logical_and(attention_mask_pred_np > 0.5, aneurysm_mask > 0.5).sum()
            union_pred = np.logical_or(attention_mask_pred_np > 0.5, aneurysm_mask > 0.5).sum()
            iou_pred = intersection_pred / (union_pred + 1e-8)

            aneurysm_in_pred = (aneurysm_mask * attention_mask_pred_np).sum() / (aneurysm_mask.sum() + 1e-8)

            pred_overlap_metrics = {
                'iou': iou_pred,
                'aneurysm_in_attention': aneurysm_in_pred
            }

            # Calculate overlap for ground truth mask
            if attention_mask_gt_np is not None:
                intersection_gt = np.logical_and(attention_mask_gt_np > 0.5, aneurysm_mask > 0.5).sum()
                union_gt = np.logical_or(attention_mask_gt_np > 0.5, aneurysm_mask > 0.5).sum()
                iou_gt = intersection_gt / (union_gt + 1e-8)

                aneurysm_in_gt = (aneurysm_mask * attention_mask_gt_np).sum() / (aneurysm_mask.sum() + 1e-8)

                gt_overlap_metrics = {
                    'iou': iou_gt,
                    'aneurysm_in_attention': aneurysm_in_gt
                }

                print(f"  Predicted Mask - IoU: {iou_pred:.4f}, Aneurysm in Attention: {aneurysm_in_pred:.4f}")
                print(f"  Standard Mask - IoU: {iou_gt:.4f}, Aneurysm in Attention: {aneurysm_in_gt:.4f}")
            else:
                print(f"  Predicted Mask - IoU: {iou_pred:.4f}, Aneurysm in Attention: {aneurysm_in_pred:.4f}")

        # Save results
        # 1. Save original image
        self._save_image(image, self.output_dir / f"{basename}_original.png")

        # 2. Save predicted attention mask
        mask_pred_uint8 = (attention_mask_pred_np * 255).astype(np.uint8)
        self._save_image(mask_pred_uint8, self.masks_dir / f"{basename}_pred_mask.png", normalize=False)

        # 3. Save standard attention mask (if available)
        if attention_mask_gt_np is not None:
            mask_gt_uint8 = (attention_mask_gt_np * 255).astype(np.uint8)
            self._save_image(mask_gt_uint8, self.standard_masks_dir / f"{basename}_standard_mask.png", normalize=False)

        # 4. Save focused images
        self._save_image(focused_image_pred, self.focused_dir / f"{basename}_pred_focused.png")
        if focused_image_gt is not None:
            self._save_image(focused_image_gt, self.focused_dir / f"{basename}_standard_focused.png")

        # 5. Save aneurysm mask (if exists)
        if has_aneurysm_mask:
            aneurysm_uint8 = (aneurysm_mask * 255).astype(np.uint8)
            self._save_image(aneurysm_uint8, self.output_dir / f"{basename}_aneurysm_mask.png", normalize=False)

            # 6. Create and save predicted overlay image
            overlay_image_pred = self._create_overlay_image(image, attention_mask_pred_np, aneurysm_mask,
                                                            attention_color='red')
            overlay_pred_uint8 = (overlay_image_pred * 255).astype(np.uint8)
            self._save_image(overlay_pred_uint8, self.overlay_dir / f"{basename}_pred_overlay.png", normalize=False)

            # 7. Create and save standard overlay image
            if attention_mask_gt_np is not None:
                overlay_image_gt = self._create_overlay_image(image, attention_mask_gt_np, aneurysm_mask,
                                                              attention_color='blue')
                overlay_gt_uint8 = (overlay_image_gt * 255).astype(np.uint8)
                self._save_image(overlay_gt_uint8, self.gt_overlay_dir / f"{basename}_standard_overlay.png",
                                 normalize=False)

        # 8. Save comparison figure (5 columns)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Predicted attention mask
        axes[1].imshow(attention_mask_pred_np, cmap='gray')
        axes[1].set_title(f'Predicted Mask\nHeight: {height_pred:.3f}\nWidth: {width_pred:.3f}')
        axes[1].axis('off')

        # Standard attention mask (if available)
        if attention_mask_gt_np is not None:
            axes[2].imshow(attention_mask_gt_np, cmap='gray')
            axes[2].set_title(f'Standard Mask\nHeight: {height_true:.3f}\nWidth: {width_true:.3f}')
        else:
            axes[2].imshow(np.zeros_like(image), cmap='gray')
            axes[2].set_title('No Standard Mask')
        axes[2].axis('off')

        # Predicted overlay image (if aneurysm mask exists)
        if has_aneurysm_mask:
            overlay_image_pred_display = self._create_overlay_image(image, attention_mask_pred_np, aneurysm_mask,
                                                                    attention_color='red')
            axes[3].imshow(overlay_image_pred_display)
            title = 'Predicted Overlay\n'
            if pred_overlap_metrics:
                title += f'IoU: {pred_overlap_metrics["iou"]:.3f}\n'
                title += f'Aneurysm in Attention: {pred_overlap_metrics["aneurysm_in_attention"]:.1%}'
            axes[3].set_title(title)
        else:
            axes[3].imshow(image, cmap='gray')
            axes[3].set_title('No Aneurysm Mask')
        axes[3].axis('off')

        # Standard overlay image (if aneurysm mask and ground truth exist)
        if has_aneurysm_mask and attention_mask_gt_np is not None:
            overlay_image_gt_display = self._create_overlay_image(image, attention_mask_gt_np, aneurysm_mask,
                                                                  attention_color='blue')
            axes[4].imshow(overlay_image_gt_display)
            title = 'Standard Overlay\n'
            if gt_overlap_metrics:
                title += f'IoU: {gt_overlap_metrics["iou"]:.3f}\n'
                title += f'Aneurysm in Attention: {gt_overlap_metrics["aneurysm_in_attention"]:.1%}'
            axes[4].set_title(title)
        else:
            axes[4].imshow(image, cmap='gray')
            axes[4].set_title('No Overlay')
        axes[4].axis('off')

        # Set main title
        if height_true is not None:
            plt.suptitle(
                f"{filename}\n"
                f"Artery Segment: {position_name}\n"
                f"Predicted: Height={height_pred:.3f}, Width={width_pred:.3f} | "
                f"Standard: Height={height_true:.3f}, Width={width_true:.3f}",
                fontsize=12
            )
        else:
            plt.suptitle(
                f"{filename}\n"
                f"Artery Segment: {position_name}\n"
                f"Predicted: Height={height_pred:.3f}, Width={width_pred:.3f}",
                fontsize=12
            )

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
            'width_error': abs(width_pred - width_true) if width_true else None,
            'position_name': position_name,
            'position_num': position_num,
            'has_aneurysm_mask': has_aneurysm_mask,
            'pred_iou': pred_overlap_metrics['iou'] if has_aneurysm_mask and pred_overlap_metrics else None,
            'pred_aneurysm_in_attention': pred_overlap_metrics[
                'aneurysm_in_attention'] if has_aneurysm_mask and pred_overlap_metrics else None,
            'gt_iou': gt_overlap_metrics['iou'] if has_aneurysm_mask and gt_overlap_metrics else None,
            'gt_aneurysm_in_attention': gt_overlap_metrics[
                'aneurysm_in_attention'] if has_aneurysm_mask and gt_overlap_metrics else None
        }

    def test_all_images(self, test_image_dir):
        """Test all images"""
        test_dir = Path(test_image_dir)

        # Get image files
        image_files = []
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']:
                    image_files.append(file_path)

        print(f"\nStarting test for {len(image_files)} images...")
        print("=" * 60)

        results = []

        for i, image_path in enumerate(tqdm(image_files, desc="Testing Progress")):
            result = self.test_single_image(image_path)
            if result:
                results.append(result)

            # Print progress every 10 images
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")

        # Save test results
        if results:
            self._save_test_results(results)

        return results

    def _save_test_results(self, results):
        """Save test results"""
        # Save as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        # Calculate statistics
        if results_df['height_error'].notna().any() and results_df['width_error'].notna().any():
            height_errors = results_df['height_error'].dropna()
            width_errors = results_df['width_error'].dropna()

            # Calculate overlap statistics (if exists)
            pred_overlap_stats = {}
            gt_overlap_stats = {}

            if 'pred_iou' in results_df.columns and results_df['pred_iou'].notna().any():
                pred_iou_values = results_df['pred_iou'].dropna()
                pred_overlap_stats = {
                    'mean_pred_iou': pred_iou_values.mean(),
                    'std_pred_iou': pred_iou_values.std(),
                    'max_pred_iou': pred_iou_values.max(),
                    'min_pred_iou': pred_iou_values.min(),
                    'mean_pred_aneurysm_in_attention': results_df['pred_aneurysm_in_attention'].dropna().mean(),
                }

            if 'gt_iou' in results_df.columns and results_df['gt_iou'].notna().any():
                gt_iou_values = results_df['gt_iou'].dropna()
                gt_overlap_stats = {
                    'mean_gt_iou': gt_iou_values.mean(),
                    'std_gt_iou': gt_iou_values.std(),
                    'max_gt_iou': gt_iou_values.max(),
                    'min_gt_iou': gt_iou_values.min(),
                    'mean_gt_aneurysm_in_attention': results_df['gt_aneurysm_in_attention'].dropna().mean(),
                }

            # Position distribution
            position_dist = results_df['position_name'].value_counts().to_dict()

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
                'min_width_error': width_errors.min(),
                'position_distribution': position_dist,
                **pred_overlap_stats,
                **gt_overlap_stats
            }

            # Save statistics
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(self.output_dir / 'test_statistics.csv', index=False)

            # Save as text file
            with open(self.output_dir / 'test_summary.txt', 'w', encoding='utf-8') as f:
                f.write("TEST RESULTS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total images tested: {stats['total_images']}\n")
                f.write(f"Images with ground truth: {stats['images_with_ground_truth']}\n")
                if 'mean_pred_iou' in stats:
                    f.write(f"Images with aneurysm mask: {len(results_df['pred_iou'].dropna())}\n\n")
                else:
                    f.write(f"Images with aneurysm mask: 0\n\n")

                f.write("Artery Segment Distribution:\n")
                f.write("-" * 40 + "\n")
                for position, count in position_dist.items():
                    f.write(f"  {position}: {count} images\n")
                f.write("\n")

                f.write("Height Ratio Error Statistics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Mean: {stats['mean_height_error']:.6f}\n")
                f.write(f"  Std Dev: {stats['std_height_error']:.6f}\n")
                f.write(f"  Max: {stats['max_height_error']:.6f}\n")
                f.write(f"  Min: {stats['min_height_error']:.6f}\n\n")

                f.write("Width Ratio Error Statistics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Mean: {stats['mean_width_error']:.6f}\n")
                f.write(f"  Std Dev: {stats['std_width_error']:.6f}\n")
                f.write(f"  Max: {stats['max_width_error']:.6f}\n")
                f.write(f"  Min: {stats['min_width_error']:.6f}\n\n")

                if 'mean_pred_iou' in stats:
                    f.write("Predicted Mask - Aneurysm Overlap Statistics:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Mean IoU: {stats['mean_pred_iou']:.6f}\n")
                    f.write(f"  IoU Std Dev: {stats['std_pred_iou']:.6f}\n")
                    f.write(f"  Max IoU: {stats['max_pred_iou']:.6f}\n")
                    f.write(f"  Min IoU: {stats['min_pred_iou']:.6f}\n")
                    f.write(f"  Mean Aneurysm in Attention: {stats['mean_pred_aneurysm_in_attention']:.2%}\n\n")

                if 'mean_gt_iou' in stats:
                    f.write("Standard Mask - Aneurysm Overlap Statistics:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Mean IoU: {stats['mean_gt_iou']:.6f}\n")
                    f.write(f"  IoU Std Dev: {stats['std_gt_iou']:.6f}\n")
                    f.write(f"  Max IoU: {stats['max_gt_iou']:.6f}\n")
                    f.write(f"  Min IoU: {stats['min_gt_iou']:.6f}\n")
                    f.write(f"  Mean Aneurysm in Attention: {stats['mean_gt_aneurysm_in_attention']:.2%}\n")

            print(f"\nTest Results Statistics:")
            print(f"  Mean Height Ratio Error: {stats['mean_height_error']:.6f}")
            print(f"  Mean Width Ratio Error: {stats['mean_width_error']:.6f}")
            if 'mean_pred_iou' in stats:
                print(f"  Predicted Mask - Mean IoU: {stats['mean_pred_iou']:.6f}")
                print(f"  Predicted Mask - Mean Aneurysm in Attention: {stats['mean_pred_aneurysm_in_attention']:.2%}")
            if 'mean_gt_iou' in stats:
                print(f"  Standard Mask - Mean IoU: {stats['mean_gt_iou']:.6f}")
                print(f"  Standard Mask - Mean Aneurysm in Attention: {stats['mean_gt_aneurysm_in_attention']:.2%}")

            print(f"\nArtery Segment Distribution:")
            for position, count in position_dist.items():
                print(f"  {position}: {count} images")

        print(f"\nTest results saved to: {self.output_dir}")


# ==============================
# 5. 主程序
# ==============================
def main():
    """Main function"""
    print("CoarseInfoExtractor Training and Testing Program")
    print("=" * 60)

    # Configuration parameters
    config = {
        # Data paths
        'train_image_dir': "D:/med_data/ai/train11",
        'val_image_dir': "D:/med_data/ai/test1",
        'test_image_dir': "D:/med_data/ai/test1",  # Use test1 for testing
        'location_excel_path': "D:/med_data/ai/location3.xlsx",
        'position_excel_path': "D:/med_data/ai/classify1.xlsx",  # Artery segment information

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
        'max_train_samples': None,
        'max_val_samples': None,
    }

    print("Configuration Parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # Step 1: Train model
    print("\nStep 1: Training CoarseInfoExtractor Model")
    trainer = CoarseExtractorTrainer(config)
    trained_model_path = trainer.train()

    # Step 2: Test model
    print("\n" + "=" * 60)
    print("Step 2: Testing Trained Model")

    # Ensure trained_model_path is string
    if isinstance(trained_model_path, Path):
        trained_model_path = str(trained_model_path)

    tester = CoarseExtractorTester(trained_model_path, config)
    test_results = tester.test_all_images(config['test_image_dir'])

    print("\n" + "=" * 60)
    print("Program Completed!")
    print(f"Training output directory: {trainer.output_dir}")
    print(f"Testing output directory: {tester.output_dir}")

    # Display some example results
    if test_results:
        print(f"\nTested {len(test_results)} images")

        # Display first 5 results
        print("\nFirst 5 Test Results:")
        for i, result in enumerate(test_results[:5]):
            print(f"  {i + 1}. {result['filename']}:")
            print(f"      Artery Segment: {result['position_name']}")
            print(f"      Predicted - Height: {result['height_pred']:.4f}, Width: {result['width_pred']:.4f}")
            if result['height_true']:
                print(f"      Standard - Height: {result['height_true']:.4f}, Width: {result['width_true']:.4f}")
                print(f"      Error - Height: {result['height_error']:.4f}, Width: {result['width_error']:.4f}")
            if result['has_aneurysm_mask']:
                print(f"      Predicted Mask - IoU: {result['pred_iou']:.4f}")
                print(f"      Predicted Mask - Aneurysm in Attention: {result['pred_aneurysm_in_attention']:.2%}")
                if result['gt_iou']:
                    print(f"      Standard Mask - IoU: {result['gt_iou']:.4f}")
                    print(f"      Standard Mask - Aneurysm in Attention: {result['gt_aneurysm_in_attention']:.2%}")


if __name__ == "__main__":
    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run main program
    main()