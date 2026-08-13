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
import random
import pickle
import shutil
from multi.locate_net_locate_encoder_simple import CoarseInfoExtractor
warnings.filterwarnings('ignore')


# ==============================
# 1. 改进的CoarseInfoExtractor模型 (支持6类动脉瘤)
# ==============================

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


# ==============================
# 2. 改进的数据加载器 (支持病历号匹配逻辑)
# ==============================

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

                        # 提取病历号 (如 ANY_001)
                        # 匹配模式：字母_数字 或 数字
                        match = re.search(r'([A-Za-z]+_)?(\d+)', base_name)
                        if match:
                            case_id = match.group(0)  # 完整匹配
                            # 保存病历号到类别的映射
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

        # 方法2: 提取病历号进行匹配 (如 ANY_001)
        # 模式: 字母_数字 或 数字
        match = re.search(r'([A-Za-z]+_)?(\d+)', basename)
        if match:
            case_id = match.group(0)  # 完整匹配
            if case_id in self.case_to_position:
                pos_info = self.case_to_position[case_id]
                position_tensor = torch.zeros(6)
                position_tensor[pos_info['new_index']] = 1.0
                return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 方法3: 只匹配数字部分
        match = re.search(r'(\d+)', basename)
        if match:
            num_part = match.group(1)
            for case_id in self.case_to_position.keys():
                if num_part in case_id:
                    pos_info = self.case_to_position[case_id]
                    position_tensor = torch.zeros(6)
                    position_tensor[pos_info['new_index']] = 1.0
                    return position_tensor, pos_info['new_index'], self.class_names[pos_info['new_index']]

        # 返回默认值
        print(f"警告: 未找到图像 {filename} 的类别信息，使用默认类别0")
        position_tensor = torch.zeros(6)
        position_tensor[0] = 1.0  # 默认使用类别0
        return position_tensor, 0, "Segment 1 (default)"


class LocationInfoLoader:
    """加载位置信息"""

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.location_dict = {}
        self.case_to_location = {}  # 存储病历号到位置信息的映射
        self._load_location_info()

    def _load_location_info(self):
        """从Excel加载位置信息"""
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
                basename = os.path.splitext(filename)[0]

                # 获取高度和宽度比例
                try:
                    height_ratio = float(row['height_ratio'])
                    width_ratio = float(row['width_ratio'])

                    # 确保值在0-1范围内
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
                    print(f"处理文件 {filename} 时出错: {e}")
                    continue

            print(f"成功加载 {len(self.location_dict)} 个文件的位置记录")
            print(f"成功加载 {len(self.case_to_location)} 个病历号的位置记录")

        except Exception as e:
            print(f"加载位置信息失败: {e}")
            raise

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

        # 方法3: 只匹配数字部分
        match = re.search(r'(\d+)', basename)
        if match:
            num_part = match.group(1)
            for case_id in self.case_to_location.keys():
                if num_part in case_id:
                    return self.case_to_location[case_id]

        # 返回默认值
        print(f"警告: 未找到图像 {filename} 的位置信息，使用默认值")
        return {'height_ratio': 0.5, 'width_ratio': 0.3}


# ==============================
# 3. 流式数据存储和读取
# ==============================

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

        return {
            'image': image,
            'position': position,
            'height_ratio': height_ratio,
            'width_ratio': width_ratio,
            'filename': filename,
            'position_name': position_name,
            'position_num': position_num
        }


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
        self.location_loader = LocationInfoLoader(location_excel_path)

        # 加载类别信息
        self.position_loader = PositionInfoLoader(position_excel_path)

        # 流式数据管理器
        self.stream_manager = StreamDataManager()

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """加载图像"""
        try:
            # 尝试加载DICOM文件
            if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(str(file_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                # 加载普通图像
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
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载图像 {file_path.name} 失败: {e}")
            return None

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
            'position_dist': {}
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

                # 准备保存的数据
                sample_data = {
                    'image': image,
                    'position': position_tensor.numpy(),
                    'height_ratio': np.array([location_info['height_ratio']]),
                    'width_ratio': np.array([location_info['width_ratio']]),
                    'filename': np.array([filename]),
                    'position_name': np.array([position_name]),
                    'position_num': np.array([position_num])
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

        return output_session


# ==============================
# 4. Early Stopping
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
# 5. 改进的训练器
# ==============================

class CoarseExtractorTrainer:
    """CoarseInfoExtractor训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录（使用日期_三位随机数格式）
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        model_dir_name = f"{date_str}_{random_num}"

        # 使用配置中的模型保存路径或默认路径
        if 'model_save_root' in config:
            self.output_dir = Path(config['model_save_root']) / model_dir_name
        else:
            self.output_dir = Path("D:/med_data/ai/pre_loc") / model_dir_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def train(self, cache_session: str):
        """训练模型"""
        print("\n" + "=" * 60)
        print("开始 CoarseInfoExtractor 训练")
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

            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [训练]')
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
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{self.config["num_epochs"]} [验证]')
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
        axes[1, 1].set_title('学习率调度')
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
# 6. 测试器
# ==============================

class CoarseExtractorTester:
    """CoarseInfoExtractor测试器"""

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
        self.location_loader = LocationInfoLoader(config['location_excel_path'])
        self.position_loader = PositionInfoLoader(config.get('position_excel_path'))

        # 创建输出目录
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = random.randint(100, 999)
        self.output_dir = Path(f"D:/med_data/ai/test_results_{date_str}_{random_num}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        model = CoarseInfoExtractor(
            image_size=self.config['image_size'],
            base_channels=self.config['base_channels'],
            num_position_classes=6,  # 6类
            dropout_rate=self.config['dropout_rate'],
            pretrain_mode=False
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # 安全地打印训练信息
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
        try:
            if image_path.suffix == '' or image_path.suffix.lower() in ['.dcm', '.dicom']:
                dicom_data = pydicom.dcmread(str(image_path), force=True)
                image = dicom_data.pixel_array.astype(np.float32)
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"无法加载图像: {image_path}")
                image = image.astype(np.float32)

            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            if image.shape != self.config['image_size']:
                image = cv2.resize(image, self.config['image_size'], interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载图像 {image_path.name} 失败: {e}")
            return None

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

    def _save_image(self, image, path, normalize=True):
        """保存图像"""
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

        aneurysm_overlay = np.zeros_like(dsa_rgb)
        aneurysm_overlay[:, :, 1] = aneurysm_mask

        overlay_image = dsa_rgb.copy()
        overlay_image = overlay_image * (1 - 0.3 * attention_mask[:, :, np.newaxis]) + attention_overlay * 0.3
        overlay_image = overlay_image + aneurysm_overlay * 0.7
        overlay_image = np.clip(overlay_image, 0, 1)

        return overlay_image

    def test_single_image(self, image_path):
        """测试单张图像"""
        filename = image_path.name
        basename = os.path.splitext(filename)[0]

        print(f"\n处理图像: {filename}")

        image = self._load_image(image_path)
        if image is None:
            print(f"  跳过图像 {filename}")
            return None

        position_tensor, position_num, position_name = self.position_loader.get_position_for_image(filename)

        print(f"  动脉瘤类别: {position_name}")

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        position_tensor = position_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor, position_tensor)
            height_pred = outputs['height_ratio'].item()
            width_pred = outputs['width_ratio'].item()

        print(f"  预测 - 高度: {height_pred:.4f}, 宽度: {width_pred:.4f}")

        try:
            true_info = self.location_loader.get_location_for_image(filename)
            height_true = true_info['height_ratio']
            width_true = true_info['width_ratio']
            print(f"  真实值 - 高度: {height_true:.4f}, 宽度: {width_true:.4f}")
            print(f"  误差 - 高度: {abs(height_pred - height_true):.4f}, 宽度: {abs(width_pred - width_true):.4f}")
        except:
            height_true = width_true = None

        height_tensor_pred = torch.tensor([height_pred], dtype=torch.float32).to(self.device)
        width_tensor_pred = torch.tensor([width_pred], dtype=torch.float32).to(self.device)

        attention_mask_pred = self.mask_generator(height_tensor_pred, width_tensor_pred)
        attention_mask_pred_np = attention_mask_pred.squeeze().cpu().numpy()

        focused_image_pred = image * attention_mask_pred_np

        attention_mask_gt_np = None
        focused_image_gt = None
        if height_true is not None and width_true is not None:
            height_tensor_gt = torch.tensor([height_true], dtype=torch.float32).to(self.device)
            width_tensor_gt = torch.tensor([width_true], dtype=torch.float32).to(self.device)

            attention_mask_gt = self.mask_generator(height_tensor_gt, width_tensor_gt)
            attention_mask_gt_np = attention_mask_gt.squeeze().cpu().numpy()
            focused_image_gt = image * attention_mask_gt_np

        aneurysm_mask = self._load_aneurysm_mask(filename)
        has_aneurysm_mask = aneurysm_mask is not None

        pred_overlap_metrics = None
        gt_overlap_metrics = None

        if has_aneurysm_mask:
            intersection_pred = np.logical_and(attention_mask_pred_np > 0.5, aneurysm_mask > 0.5).sum()
            union_pred = np.logical_or(attention_mask_pred_np > 0.5, aneurysm_mask > 0.5).sum()
            iou_pred = intersection_pred / (union_pred + 1e-8)

            aneurysm_in_pred = (aneurysm_mask * attention_mask_pred_np).sum() / (aneurysm_mask.sum() + 1e-8)

            pred_overlap_metrics = {
                'iou': iou_pred,
                'aneurysm_in_attention': aneurysm_in_pred
            }

            if attention_mask_gt_np is not None:
                intersection_gt = np.logical_and(attention_mask_gt_np > 0.5, aneurysm_mask > 0.5).sum()
                union_gt = np.logical_or(attention_mask_gt_np > 0.5, aneurysm_mask > 0.5).sum()
                iou_gt = intersection_gt / (union_gt + 1e-8)

                aneurysm_in_gt = (aneurysm_mask * attention_mask_gt_np).sum() / (aneurysm_mask.sum() + 1e-8)

                gt_overlap_metrics = {
                    'iou': iou_gt,
                    'aneurysm_in_attention': aneurysm_in_gt
                }

                print(f"  预测掩膜 - IoU: {iou_pred:.4f}, 动脉瘤在注意力区域内: {aneurysm_in_pred:.4f}")
                print(f"  标准掩膜 - IoU: {iou_gt:.4f}, 动脉瘤在注意力区域内: {aneurysm_in_gt:.4f}")
            else:
                print(f"  预测掩膜 - IoU: {iou_pred:.4f}, 动脉瘤在注意力区域内: {aneurysm_in_pred:.4f}")

        self._save_image(image, self.output_dir / f"{basename}_original.png")

        mask_pred_uint8 = (attention_mask_pred_np * 255).astype(np.uint8)
        self._save_image(mask_pred_uint8, self.masks_dir / f"{basename}_pred_mask.png", normalize=False)

        if attention_mask_gt_np is not None:
            mask_gt_uint8 = (attention_mask_gt_np * 255).astype(np.uint8)
            self._save_image(mask_gt_uint8, self.standard_masks_dir / f"{basename}_standard_mask.png", normalize=False)

        self._save_image(focused_image_pred, self.focused_dir / f"{basename}_pred_focused.png")
        if focused_image_gt is not None:
            self._save_image(focused_image_gt, self.focused_dir / f"{basename}_standard_focused.png")

        if has_aneurysm_mask:
            aneurysm_uint8 = (aneurysm_mask * 255).astype(np.uint8)
            self._save_image(aneurysm_uint8, self.output_dir / f"{basename}_aneurysm_mask.png", normalize=False)

            overlay_image_pred = self._create_overlay_image(image, attention_mask_pred_np, aneurysm_mask,
                                                            attention_color='red')
            overlay_pred_uint8 = (overlay_image_pred * 255).astype(np.uint8)
            self._save_image(overlay_pred_uint8, self.overlay_dir / f"{basename}_pred_overlay.png", normalize=False)

            if attention_mask_gt_np is not None:
                overlay_image_gt = self._create_overlay_image(image, attention_mask_gt_np, aneurysm_mask,
                                                              attention_color='blue')
                overlay_gt_uint8 = (overlay_image_gt * 255).astype(np.uint8)
                self._save_image(overlay_gt_uint8, self.gt_overlay_dir / f"{basename}_standard_overlay.png",
                                 normalize=False)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].imshow(attention_mask_pred_np, cmap='gray')
        axes[1].set_title(f'预测掩膜\n高度: {height_pred:.3f}\n宽度: {width_pred:.3f}')
        axes[1].axis('off')

        if attention_mask_gt_np is not None:
            axes[2].imshow(attention_mask_gt_np, cmap='gray')
            axes[2].set_title(f'标准掩膜\n高度: {height_true:.3f}\n宽度: {width_true:.3f}')
        else:
            axes[2].imshow(np.zeros_like(image), cmap='gray')
            axes[2].set_title('无标准掩膜')
        axes[2].axis('off')

        if has_aneurysm_mask:
            overlay_image_pred_display = self._create_overlay_image(image, attention_mask_pred_np, aneurysm_mask,
                                                                    attention_color='red')
            axes[3].imshow(overlay_image_pred_display)
            title = '预测叠加\n'
            if pred_overlap_metrics:
                title += f'IoU: {pred_overlap_metrics["iou"]:.3f}\n'
                title += f'动脉瘤在注意力: {pred_overlap_metrics["aneurysm_in_attention"]:.1%}'
            axes[3].set_title(title)
        else:
            axes[3].imshow(image, cmap='gray')
            axes[3].set_title('无动脉瘤掩膜')
        axes[3].axis('off')

        if has_aneurysm_mask and attention_mask_gt_np is not None:
            overlay_image_gt_display = self._create_overlay_image(image, attention_mask_gt_np, aneurysm_mask,
                                                                  attention_color='blue')
            axes[4].imshow(overlay_image_gt_display)
            title = '标准叠加\n'
            if gt_overlap_metrics:
                title += f'IoU: {gt_overlap_metrics["iou"]:.3f}\n'
                title += f'动脉瘤在注意力: {gt_overlap_metrics["aneurysm_in_attention"]:.1%}'
            axes[4].set_title(title)
        else:
            axes[4].imshow(image, cmap='gray')
            axes[4].set_title('无叠加')
        axes[4].axis('off')

        if height_true is not None:
            plt.suptitle(
                f"{filename}\n"
                f"动脉瘤类别: {position_name}\n"
                f"预测: 高度={height_pred:.3f}, 宽度={width_pred:.3f} | "
                f"标准: 高度={height_true:.3f}, 宽度={width_true:.3f}",
                fontsize=12
            )
        else:
            plt.suptitle(
                f"{filename}\n"
                f"动脉瘤类别: {position_name}\n"
                f"预测: 高度={height_pred:.3f}, 宽度={width_pred:.3f}",
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
        """测试所有图像"""
        test_dir = Path(test_image_dir)

        image_files = []
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix == '' or file_path.suffix.lower() in ['.dcm', '.dicom', '.png', '.jpg', '.jpeg']:
                    image_files.append(file_path)

        print(f"\n开始测试 {len(image_files)} 张图像...")
        print("=" * 60)

        results = []

        for i, image_path in enumerate(tqdm(image_files, desc="测试进度")):
            result = self.test_single_image(image_path)
            if result:
                results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(image_files)} 张图像")

        if results:
            self._save_test_results(results)

        return results

    def _save_test_results(self, results):
        """保存测试结果"""
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'test_results.csv', index=False)

        if results_df['height_error'].notna().any() and results_df['width_error'].notna().any():
            height_errors = results_df['height_error'].dropna()
            width_errors = results_df['width_error'].dropna()

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

            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(self.output_dir / 'test_statistics.csv', index=False)

            with open(self.output_dir / 'test_summary.txt', 'w', encoding='utf-8') as f:
                f.write("测试结果摘要\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"测试图像总数: {stats['total_images']}\n")
                f.write(f"有真实值的图像数: {stats['images_with_ground_truth']}\n")
                if 'mean_pred_iou' in stats:
                    f.write(f"有动脉瘤掩膜的图像数: {len(results_df['pred_iou'].dropna())}\n\n")
                else:
                    f.write(f"有动脉瘤掩膜的图像数: 0\n\n")

                f.write("动脉瘤类别分布:\n")
                f.write("-" * 40 + "\n")
                for position, count in position_dist.items():
                    f.write(f"  {position}: {count} 张图像\n")
                f.write("\n")

                f.write("高度比例误差统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  均值: {stats['mean_height_error']:.6f}\n")
                f.write(f"  标准差: {stats['std_height_error']:.6f}\n")
                f.write(f"  最大值: {stats['max_height_error']:.6f}\n")
                f.write(f"  最小值: {stats['min_height_error']:.6f}\n\n")

                f.write("宽度比例误差统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  均值: {stats['mean_width_error']:.6f}\n")
                f.write(f"  标准差: {stats['std_width_error']:.6f}\n")
                f.write(f"  最大值: {stats['max_width_error']:.6f}\n")
                f.write(f"  最小值: {stats['min_width_error']:.6f}\n\n")

                if 'mean_pred_iou' in stats:
                    f.write("预测掩膜 - 动脉瘤重叠统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  平均IoU: {stats['mean_pred_iou']:.6f}\n")
                    f.write(f"  IoU标准差: {stats['std_pred_iou']:.6f}\n")
                    f.write(f"  最大IoU: {stats['max_pred_iou']:.6f}\n")
                    f.write(f"  最小IoU: {stats['min_pred_iou']:.6f}\n")
                    f.write(f"  平均动脉瘤在注意力区域内: {stats['mean_pred_aneurysm_in_attention']:.2%}\n\n")

                if 'mean_gt_iou' in stats:
                    f.write("标准掩膜 - 动脉瘤重叠统计:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  平均IoU: {stats['mean_gt_iou']:.6f}\n")
                    f.write(f"  IoU标准差: {stats['std_gt_iou']:.6f}\n")
                    f.write(f"  最大IoU: {stats['max_gt_iou']:.6f}\n")
                    f.write(f"  最小IoU: {stats['min_gt_iou']:.6f}\n")
                    f.write(f"  平均动脉瘤在注意力区域内: {stats['mean_gt_aneurysm_in_attention']:.2%}\n")

            print(f"\n测试结果统计:")
            print(f"  平均高度比例误差: {stats['mean_height_error']:.6f}")
            print(f"  平均宽度比例误差: {stats['mean_width_error']:.6f}")
            if 'mean_pred_iou' in stats:
                print(f"  预测掩膜 - 平均IoU: {stats['mean_pred_iou']:.6f}")
                print(f"  预测掩膜 - 平均动脉瘤在注意力区域内: {stats['mean_pred_aneurysm_in_attention']:.2%}")
            if 'mean_gt_iou' in stats:
                print(f"  标准掩膜 - 平均IoU: {stats['mean_gt_iou']:.6f}")
                print(f"  标准掩膜 - 平均动脉瘤在注意力区域内: {stats['mean_gt_aneurysm_in_attention']:.2%}")

            print(f"\n动脉瘤类别分布:")
            for position, count in position_dist.items():
                print(f"  {position}: {count} 张图像")

        print(f"\n测试结果保存到: {self.output_dir}")


# ==============================
# 7. 主程序
# ==============================

def main():
    """主函数"""
    print("CoarseInfoExtractor 训练与测试程序")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/translate/contrast",
        'val_image_dir': "D:/med_data/ai/translate/test1",
        'test_image_dir': "D:/med_data/ai/translate/test1",
        'location_excel_path': "D:/med_data/ai/translate/contrast/location_trans_updated.xlsx",
        'position_excel_path': "D:/med_data/ai/translate/contrast/classify_all_trans_updated.xlsx",

        # 模型参数
        'image_size': (512, 512),
        'base_channels': 32,
        'num_position_classes': 6,  # 6类动脉瘤
        'dropout_rate': 0.2,

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
        'cache_root': "D:/med_data/ai/stream_cache",  # 流式数据缓存根目录
        'model_save_root': "D:/med_data/ai/pre_loc",  # 模型保存根目录
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
            print(f"      动脉瘤类别: {result['position_name']}")
            print(f"      预测 - 高度: {result['height_pred']:.4f}, 宽度: {result['width_pred']:.4f}")
            if result['height_true']:
                print(f"      标准 - 高度: {result['height_true']:.4f}, 宽度: {result['width_true']:.4f}")
                print(f"      误差 - 高度: {result['height_error']:.4f}, 宽度: {result['width_error']:.4f}")
            if result['has_aneurysm_mask']:
                print(f"      预测掩膜 - IoU: {result['pred_iou']:.4f}")
                print(f"      预测掩膜 - 动脉瘤在注意力: {result['pred_aneurysm_in_attention']:.2%}")
                if result['gt_iou']:
                    print(f"      标准掩膜 - IoU: {result['gt_iou']:.4f}")
                    print(f"      标准掩膜 - 动脉瘤在注意力: {result['gt_aneurysm_in_attention']:.2%}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()