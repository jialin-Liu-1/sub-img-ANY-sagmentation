import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from tqdm import tqdm
import time
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import traceback
import gc
from datetime import datetime
import random

# 导入模型模块
from multi.locate_net import ArterySegmentationSystem, create_pretraining_model, create_complete_system

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EarlyStopping:
    """早停类"""

    def __init__(self, patience=10, min_delta=0, mode='min', verbose=True):

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.best_epoch = None
        self.early_stop = False
        self.improved = False

    def __call__(self, current_value, epoch):
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            self.improved = True
            return False, True

        self.improved = False

        if self.mode == 'min':
            # 监控损失，越小越好
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.best_epoch = epoch
                self.counter = 0
                self.improved = True
            else:
                self.counter += 1
        elif self.mode == 'max':
            # 监控指标，越大越好
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.best_epoch = epoch
                self.counter = 0
                self.improved = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f'🚨 早停触发！连续{self.patience}个epoch未改善')
                print(f'   最佳指标 ({self.mode}模式): {self.best_value:.4f} (epoch {self.best_epoch})')

        return self.early_stop, self.improved

    def reset(self):
        """重置早停器"""
        self.counter = 0
        self.best_value = None
        self.best_epoch = None
        self.early_stop = False
        self.improved = False


class MedicalRecordPositionLoader:
    """基于病历号的位置信息加载器"""

    def __init__(self, excel_path="D:\\med_data\\ai\\classify.xlsx"):
        self.excel_path = excel_path
        self.position_dict = {}  # 病历号 -> 位置
        self._load_by_medical_record()

    def _load_by_medical_record(self):
        """根据病历号加载位置信息"""
        try:
            # 使用pandas读取Excel
            df = pd.read_excel(self.excel_path, header=0)
            print(f"Excel列名: {df.columns.tolist()}")
            print(f"前5行数据:\n{df.head()}")

            for idx, row in df.iterrows():
                try:
                    # 第一列: 病历号
                    record_str = str(row.iloc[0]).strip()

                    # 提取数字病历号
                    record_num = None
                    if record_str:
                        # 尝试匹配数字
                        num_match = re.search(r'(\d+)', record_str)
                        if num_match:
                            record_num = int(num_match.group(1))

                    if record_num is None:
                        continue

                    # 第二列: 位置值
                    position_val = row.iloc[1]
                    if pd.isna(position_val):
                        position_num = 0
                    else:
                        try:
                            position_num = int(float(position_val))
                            position_num = max(0, min(7, position_num))
                        except:
                            position_num = 0

                    self.position_dict[record_num] = position_num
                    if idx < 5:  # 显示前5条记录
                        print(f"  病历号 {record_num}: 位置 {position_num}")

                except Exception as e:
                    continue

            print(f"位置信息加载完成: {len(self.position_dict)} 条病历记录")
            print(f"前10个病历号: {list(self.position_dict.keys())[:10]}")

        except Exception as e:
            print(f"加载位置信息失败: {e}")
            traceback.print_exc()
            self.position_dict = {}

    def extract_medical_record_from_filename(self, filename):
        """从文件名中提取病历号"""
        try:
            # 文件名格式: "ANY_450_0" 或 "ANY_450_0.dcm"
            basename = os.path.splitext(filename)[0]  # 去掉扩展名
            parts = basename.split('_')

            if len(parts) >= 2:
                # 格式: ANY_450_0 -> 取第二部分作为病历号
                try:
                    record_num = int(parts[1])
                    return record_num
                except:
                    pass

            # 尝试从字符串中提取数字
            num_match = re.search(r'(\d+)', basename)
            if num_match:
                try:
                    return int(num_match.group(1))
                except:
                    pass

            return None

        except Exception as e:
            print(f"从 {filename} 提取病历号失败: {e}")
            return None

    def get_position_for_image(self, filename):
        """根据图像文件名获取位置信息"""
        try:
            # 提取病历号
            record_num = self.extract_medical_record_from_filename(filename)

            if record_num is None:
                print(f"警告: 无法从文件名提取病历号: {filename}")
                position_num = 0
                case_id = f"unknown_{filename}"
            else:
                case_id = f"record_{record_num}"

                # 查找位置信息
                if record_num in self.position_dict:
                    position_num = self.position_dict[record_num]
                else:
                    # 检查是否为翻转病例（病历号 > 500）
                    if record_num > 500:
                        original_num = record_num - 500
                        if original_num in self.position_dict:
                            position_num = self.position_dict[original_num]
                        else:
                            position_num = 0
                    else:
                        position_num = 0

            # 创建位置张量（one-hot编码）
            position_tensor = torch.zeros(8, dtype=torch.float32)
            if 0 <= position_num < 8:
                position_tensor[position_num] = 1.0

            return position_tensor, case_id

        except Exception as e:
            print(f"为 {filename} 获取位置信息失败: {e}")
            return torch.zeros(8, dtype=torch.float32), f"error_{filename}"


class MedicalRecordLocationLoader:
    """基于完整文件名的动脉瘤位置信息加载器（用于预训练）
    直接使用完整的文件名（如ANY_001_0）来匹配位置信息
    """

    def __init__(self, excel_path="D:\\med_data\\ai\\location.xlsx"):
        self.excel_path = excel_path
        self.location_dict = {}  # 完整文件名 -> (高度比例, 半径比例)
        self._load_location_info()

    def _load_location_info(self):
        """加载动脉瘤位置信息"""
        try:
            # 使用pandas读取Excel
            df = pd.read_excel(self.excel_path, header=0)
            print(f"位置信息Excel列名: {df.columns.tolist()}")
            print(f"位置信息前5行:\n{df.head()}")

            # 检查Excel格式
            # 需要三列：文件名、高度比例、半径比例
            if len(df.columns) < 3:
                print(f"警告: Excel文件需要至少3列")
                print(f"当前列数: {len(df.columns)}")
                print("请确保Excel格式为: 文件名 | 高度比例 | 半径比例")
                return

            for idx, row in df.iterrows():
                try:
                    # 第一列: 文件名（如ANY_001_0）
                    filename_str = str(row.iloc[0]).strip()

                    if not filename_str:
                        continue

                    # 清理文件名：移除扩展名，统一格式
                    clean_filename = self._clean_filename(filename_str)

                    # 第二列: 高度比例 (0-1)
                    height_ratio = row.iloc[1]
                    # 第三列: 半径比例 (0-1)
                    radius_ratio = row.iloc[2]

                    # 处理缺失值
                    if pd.isna(height_ratio) or pd.isna(radius_ratio):
                        continue

                    # 转换为浮点数
                    try:
                        height_val = float(height_ratio)
                        radius_val = float(radius_ratio)

                        # 限制在0-1范围内
                        height_val = max(0.0, min(1.0, height_val))
                        radius_val = max(0.0, min(1.0, radius_val))

                        # 存储到字典
                        self.location_dict[clean_filename] = (height_val, radius_val)

                        if idx < 10:  # 显示前10条记录
                            print(f"  文件名 {clean_filename}: 高度={height_val:.3f}, 半径={radius_val:.3f}")

                    except Exception as e:
                        print(f"  第{idx + 1}行数据格式错误: {e}")
                        continue

                except Exception as e:
                    print(f"  处理第{idx + 1}行时出错: {e}")
                    continue

            print(f"动脉瘤位置信息加载完成: {len(self.location_dict)} 条记录")

        except Exception as e:
            print(f"加载动脉瘤位置信息失败: {e}")
            traceback.print_exc()
            self.location_dict = {}

    def _clean_filename(self, filename_str):
        """清理文件名，统一格式"""
        try:
            # 移除扩展名
            filename = os.path.splitext(filename_str)[0]

            # 移除路径
            filename = os.path.basename(filename)

            # 统一大小写
            filename = filename.upper()

            return filename

        except Exception as e:
            print(f"清理文件名 {filename_str} 失败: {e}")
            return filename_str

    def _extract_record_number(self, filename):
        """从文件名中提取病历号（用于统计）"""
        try:
            # 格式: ANY_001_0 或 001_0
            parts = filename.split('_')

            if len(parts) >= 2:
                # 如果是ANY_001_0格式，第二部分是病历号
                if parts[0].isalpha() and len(parts) >= 3:
                    return parts[1]
                # 如果是001_0格式，第一部分是病历号
                else:
                    return parts[0]

            return None

        except Exception as e:
            return None

    def get_location_for_image(self, filename):
        """根据图像文件名获取动脉瘤位置信息"""
        try:
            # 清理文件名用于匹配
            clean_filename = self._clean_filename(filename)

            # 直接查找
            if clean_filename in self.location_dict:
                height, radius = self.location_dict[clean_filename]
                return torch.tensor([height, radius], dtype=torch.float32), True

            # 如果没有找到，尝试查找对应的翻转病例
            # 从文件名中提取病历号
            record_num = self._extract_record_number(clean_filename)

            if record_num:
                try:
                    record_int = int(record_num)
                    if record_int > 500:
                        # 如果是翻转病例，查找原始病例
                        original_num = record_int - 500

                        # 构建原始文件名（假设前缀和格式相同）
                        for key in self.location_dict.keys():
                            if f"_{record_int}_" in key:
                                # 找到翻转文件名，构建原始文件名
                                original_key = key.replace(f"_{record_int}_", f"_{original_num}_")
                                if original_key in self.location_dict:
                                    height, radius = self.location_dict[original_key]
                                    print(f"翻转病例 {clean_filename} -> {original_key}")
                                    return torch.tensor([height, radius], dtype=torch.float32), True
                except:
                    pass

            # 没找到对应的位置信息
            return torch.tensor([0.5, 0.3], dtype=torch.float32), False

        except Exception as e:
            print(f"为 {filename} 获取动脉瘤位置信息失败: {e}")
            return torch.tensor([0.5, 0.3], dtype=torch.float32), False


class ArteryDataset(Dataset):
    """动脉瘤分割数据集 - 专为无后缀DICOM文件优化"""

    def __init__(self, image_dir, mask_dir, position_loader=None,
                 location_loader=None, pretrain_mode=False, max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.position_loader = position_loader
        self.location_loader = location_loader
        self.pretrain_mode = pretrain_mode
        self.max_samples = max_samples

        # 获取文件列表
        self.file_pairs = self._get_file_pairs()

        if self.file_pairs:
            if max_samples and len(self.file_pairs) > max_samples:
                self.file_pairs = self.file_pairs[:max_samples]

            print(f"找到 {len(self.file_pairs)} 个样本")

        else:
            print("错误: 未找到文件对")

    def _get_file_base_name(self, filename):
        """获取文件基础名（用于匹配）"""
        try:
            # 移除扩展名（如果有）
            basename = os.path.splitext(filename)[0] if '.' in filename else filename

            # 格式: "ANY_450_0" -> 取前两部分 "ANY_450"
            parts = basename.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
            else:
                return basename

        except:
            return filename

    def _get_file_pairs(self):
        """获取图像和掩码文件对"""
        file_pairs = []

        try:
            # 获取所有图像文件（假设都是无后缀DICOM文件）
            image_files = []
            for f in os.listdir(self.image_dir):
                # 假设所有文件都是无后缀的DICOM文件
                if os.path.isfile(os.path.join(self.image_dir, f)):
                    image_files.append(f)

            print(f"找到 {len(image_files)} 个DICOM图像文件")

            # 获取所有掩码文件
            mask_files = []
            for f in os.listdir(self.mask_dir):
                if os.path.isfile(os.path.join(self.mask_dir, f)):
                    mask_files.append(f)

            print(f"找到 {len(mask_files)} 个掩码文件")

            if not image_files or not mask_files:
                return file_pairs

            # 创建基础名映射
            image_dict = {}
            for img_file in image_files:
                base_name = self._get_file_base_name(img_file)
                if base_name:
                    image_dict[base_name] = img_file

            # 匹配文件
            matched_count = 0
            for mask_file in mask_files:
                mask_base = self._get_file_base_name(mask_file)
                if mask_base in image_dict:
                    file_pairs.append((image_dict[mask_base], mask_file))
                    matched_count += 1

            print(f"成功匹配 {matched_count} 个文件对")

            # 显示前几个匹配对用于验证
            if file_pairs:
                print("前5个匹配的文件对:")
                for i, (img, mask) in enumerate(file_pairs[:5]):
                    print(f"  {i + 1}. 图像: {img} -> 掩码: {mask}")

        except Exception as e:
            print(f"获取文件对错误: {e}")
            traceback.print_exc()

        return file_pairs

    def _load_dicom_image(self, file_path):
        """加载DICOM图像文件"""
        try:
            import pydicom
            dicom_data = pydicom.dcmread(file_path, force=True)
            image = dicom_data.pixel_array.astype(np.float32)
            return image

        except Exception as e:
            print(f"加载DICOM文件 {file_path} 失败: {e}")
            return None

    def _load_image(self, file_path):
        """加载图像文件 - 专为无后缀DICOM优化"""
        try:
            # 假设所有图像都是无后缀的DICOM文件
            image = self._load_dicom_image(file_path)

            if image is None:
                # 如果加载失败，尝试其他格式
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"无法加载图像: {file_path}")
                image = img.astype(np.float32)
                print(f"使用CV2加载: {os.path.basename(file_path)}")

            # 归一化到0-1范围
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            else:
                image = np.zeros_like(image)

            # 调整尺寸到512x512
            if image.shape != (512, 512):
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

            return image

        except Exception as e:
            print(f"加载图像 {file_path} 失败: {e}")
            return np.zeros((512, 512), dtype=np.float32)

    def _load_mask(self, file_path):
        """加载掩码文件"""
        try:
            # 首先尝试作为DICOM加载
            try:
                import pydicom
                dicom_data = pydicom.dcmread(file_path, force=True)
                mask = dicom_data.pixel_array.astype(np.float32)
            except:
                # 如果不是DICOM，尝试常规图像格式
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # 尝试作为numpy数组加载
                    if file_path.lower().endswith('.npy'):
                        mask = np.load(file_path).astype(np.float32)
                    else:
                        raise ValueError(f"无法加载掩码: {file_path}")
                else:
                    mask = img.astype(np.float32)

            # 归一化和二值化
            if mask.max() > 1.0:
                mask = mask / mask.max()  # 归一化到0-1

            # 二值化（阈值0.5）
            mask = (mask > 0.5).astype(np.float32)

            # 调整尺寸到512x512
            if mask.shape != (512, 512):
                mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            return mask

        except Exception as e:
            print(f"加载掩码 {file_path} 失败: {e}")
            return np.zeros((512, 512), dtype=np.float32)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_file, mask_file = self.file_pairs[idx]

        try:
            # 加载图像
            image_path = os.path.join(self.image_dir, image_file)
            image = self._load_image(image_path)

            # 加载掩码
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = self._load_mask(mask_path)

            # 获取位置信息
            if self.position_loader:
                position_tensor, case_id = self.position_loader.get_position_for_image(image_file)
            else:
                position_tensor = torch.zeros(8, dtype=torch.float32)
                case_id = image_file

            # 如果是预训练模式，获取动脉瘤位置信息
            if self.pretrain_mode and self.location_loader:
                location_tensor, has_location = self.location_loader.get_location_for_image(image_file)
            else:
                location_tensor = torch.zeros(2, dtype=torch.float32)  # 默认值
                has_location = False

            # 添加通道维度
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # 转换为张量
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask).float()

            # 清理内存
            del image, mask

            if self.pretrain_mode:
                return image_tensor, mask_tensor, position_tensor, location_tensor, case_id
            else:
                return image_tensor, mask_tensor, position_tensor, case_id

        except Exception as e:
            print(f"处理样本 {image_file} 失败: {e}")
            traceback.print_exc()
            # 返回默认张量
            dummy_image = torch.zeros((1, 512, 512), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            dummy_position = torch.zeros(8, dtype=torch.float32)

            if self.pretrain_mode:
                dummy_location = torch.zeros(2, dtype=torch.float32)
                return dummy_image, dummy_mask, dummy_position, dummy_location, "error"
            else:
                return dummy_image, dummy_mask, dummy_position, "error"


class PretrainingLoss(nn.Module):
    """预训练损失函数"""

    def __init__(self, height_weight=1.0, radius_weight=1.0):
        super().__init__()
        self.height_weight = height_weight
        self.radius_weight = radius_weight
        self.huber_loss = nn.HuberLoss(delta=0.1)

    def forward(self, predictions, targets):
        """计算预训练损失"""
        # 高度比例损失
        height_loss = self.huber_loss(predictions['height_ratio'], targets['height_ratio'])

        # 半径比例损失
        radius_loss = self.huber_loss(predictions['radius_ratio'], targets['radius_ratio'])

        # 总损失
        total_loss = self.height_weight * height_loss + self.radius_weight * radius_loss

        return {
            'total_loss': total_loss,
            'height_loss': height_loss,
            'radius_loss': radius_loss
        }


class SegmentationLoss(nn.Module):
    """分割损失函数"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()

    def dice_loss(self, pred, target):
        """Dice损失"""
        smooth = 1e-5
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def forward(self, pred, target):
        """计算分割损失"""
        # Dice损失
        dice = self.dice_loss(pred, target)
        # 二元交叉熵损失
        bce = self.bce_loss(pred, target)
        # 总损失
        total_loss = self.dice_weight * dice + self.bce_weight * bce

        return {
            'total_loss': total_loss,
            'dice_loss': dice,
            'bce_loss': bce
        }


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 创建主目录
        base_dir = Path("D:/med_data/ai/model")
        date_str = datetime.now().strftime('%Y%m%d')
        random_num = f"{random.randint(100, 999):03d}"
        self.experiment_name = f"{date_str}_{random_num}"
        self.experiment_dir = base_dir / self.experiment_name

        # 创建实验目录结构
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.chp_dir = self.experiment_dir / "CHP"  # 检查点目录
        self.log_dir = self.experiment_dir / "logs"
        self.plots_dir = self.log_dir / "plots"  # 图表目录
        self.metrics_dir = self.log_dir / "metrics"  # 指标目录

        # 创建所有目录
        for dir_path in [self.experiment_dir, self.checkpoint_dir, self.chp_dir,
                         self.log_dir, self.plots_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Checkpoints directory: {self.checkpoint_dir}")
        print(f"CHP directory: {self.chp_dir}")
        print(f"Logs directory: {self.log_dir}")
        print(f"Plots directory: {self.plots_dir}")
        print(f"Metrics directory: {self.metrics_dir}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': []
        }

        # 加载位置信息
        self.position_loader = MedicalRecordPositionLoader(config['position_excel_path'])

        # 如果需要预训练，加载动脉瘤位置信息
        if config.get('pretrain', False):
            self.location_loader = MedicalRecordLocationLoader(config['location_excel_path'])
        else:
            self.location_loader = None

    def create_dataloaders(self, pretrain_mode=False):
        """创建数据加载器"""
        print(f"\nCreating data loaders ({'pretrain mode' if pretrain_mode else 'train mode'})...")

        # 训练集
        train_dataset = ArteryDataset(
            image_dir=self.config['train_image_dir'],
            mask_dir=self.config['train_mask_dir'],
            position_loader=self.position_loader,
            location_loader=self.location_loader,
            pretrain_mode=pretrain_mode,
            max_samples=self.config.get('max_train_samples', None)
        )

        # 验证集
        val_dataset = ArteryDataset(
            image_dir=self.config['val_image_dir'],
            mask_dir=self.config['val_mask_dir'],
            position_loader=self.position_loader,
            location_loader=self.location_loader,
            pretrain_mode=pretrain_mode,
            max_samples=self.config.get('max_val_samples', None)
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Batch size: {self.config['batch_size']}")

        return train_loader, val_loader

    def calculate_dice_score(self, pred, target):
        """计算Dice分数"""
        smooth = 1e-5
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def pretrain(self):
        """预训练粗信息提取模块"""
        print("\n" + "=" * 60)
        print("Starting Pretraining of Coarse Information Extraction Module")
        print("=" * 60)

        # 创建预训练模型
        pretrain_model = create_pretraining_model(
            image_size=(512, 512),
            base_channels=self.config['base_channels'],
            num_position_classes=8,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders(pretrain_mode=True)

        # 创建优化器
        optimizer = torch.optim.AdamW(
            pretrain_model.parameters(),
            lr=self.config['pretrain_lr'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # 创建学习率调度器和早停器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=self.config.get('early_stopping_min_delta', 0.001),
            mode='min',
            verbose=True
        )

        # 创建损失函数
        criterion = PretrainingLoss(
            height_weight=1.0,
            radius_weight=1.0
        )

        # 训练循环
        best_val_loss = float('inf')
        best_model_path = None
        metrics_history = []

        for epoch in range(self.config['pretrain_epochs']):
            # 训练阶段
            pretrain_model.train()
            train_losses = {'total': 0, 'height': 0, 'radius': 0}

            train_bar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch + 1}/{self.config["pretrain_epochs"]}')
            for batch_idx, batch_data in enumerate(train_bar):
                images, masks, positions, locations, _ = batch_data

                images = images.to(self.device)
                positions = positions.to(self.device)
                locations = locations.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                outputs = pretrain_model(images, positions)

                # 准备目标
                targets = {
                    'height_ratio': locations[:, 0],
                    'radius_ratio': locations[:, 1]
                }

                # 计算损失
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
                optimizer.step()

                # 记录损失
                train_losses['total'] += loss.item()
                train_losses['height'] += loss_dict['height_loss'].item()
                train_losses['radius'] += loss_dict['radius_loss'].item()

                # 更新进度条
                avg_loss = train_losses['total'] / (batch_idx + 1)
                train_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'height_loss': f'{loss_dict["height_loss"].item():.4f}',
                    'radius_loss': f'{loss_dict["radius_loss"].item():.4f}'
                })

            # 验证阶段
            pretrain_model.eval()
            val_losses = {'total': 0, 'height': 0, 'radius': 0}

            with torch.no_grad():
                for batch_data in val_loader:
                    images, masks, positions, locations, _ = batch_data

                    images = images.to(self.device)
                    positions = positions.to(self.device)
                    locations = locations.to(self.device)

                    # 前向传播
                    outputs = pretrain_model(images, positions)

                    # 准备目标
                    targets = {
                        'height_ratio': locations[:, 0],
                        'radius_ratio': locations[:, 1]
                    }

                    # 计算损失
                    loss_dict = criterion(outputs, targets)

                    # 记录损失
                    val_losses['total'] += loss_dict['total_loss'].item()
                    val_losses['height'] += loss_dict['height_loss'].item()
                    val_losses['radius'] += loss_dict['radius_loss'].item()

            # 计算平均损失
            train_loss_avg = train_losses['total'] / len(train_loader)
            val_loss_avg = val_losses['total'] / len(val_loader)

            # 更新学习率
            scheduler.step(val_loss_avg)

            # 早停检查
            should_stop, improved = early_stopping(val_loss_avg, epoch + 1)

            # 记录指标历史
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'train_height_loss': train_losses['height'] / len(train_loader),
                'val_height_loss': val_losses['height'] / len(val_loader),
                'train_radius_loss': train_losses['radius'] / len(train_loader),
                'val_radius_loss': val_losses['radius'] / len(val_loader),
                'improved': improved,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # 记录学习率
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 打印统计信息
            print(f'\nEpoch {epoch + 1}/{self.config["pretrain_epochs"]}:')
            print(f'  Train Loss: {train_loss_avg:.4f} (Height: {train_losses["height"] / len(train_loader):.4f}, '
                  f'Radius: {train_losses["radius"] / len(train_loader):.4f})')
            print(f'  Val Loss: {val_loss_avg:.4f} (Height: {val_losses["height"] / len(val_loader):.4f}, '
                  f'Radius: {val_losses["radius"] / len(val_loader):.4f})')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            if improved:
                print('  ✓ Model improved')
            else:
                print(f'  ⚠️ No improvement for {early_stopping.counter} epochs')

            # 保存最佳模型（如果有改善）
            if improved:
                best_val_loss = val_loss_avg
                best_model_path = self.checkpoint_dir / f'best_pretrain_model.pth'

                # 保存预训练权重
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': pretrain_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'config': self.config,
                    'metrics_history': metrics_history
                }, best_model_path)

                print(f'  ✓ Saved best pretrain model to: {best_model_path}')

            # 每5个epoch保存检查点到CHP目录
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.chp_dir / f'pretrain_checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': pretrain_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'early_stopping_counter': early_stopping.counter,
                    'metrics_history': metrics_history
                }, checkpoint_path)
                print(f'  ✓ Saved checkpoint to CHP directory: {checkpoint_path}')

            # 保存训练历史
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss_avg)

            # 如果触发早停，跳出循环
            if should_stop:
                print(f"\n🚨 Early stopping triggered! Stopping pretraining")
                break

        # 保存最终模型
        final_model_path = self.checkpoint_dir / 'final_pretrain_model.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': pretrain_model.state_dict(),
            'config': self.config,
            'history': self.history,
            'metrics_history': metrics_history,
            'early_stopping': {
                'best_value': early_stopping.best_value,
                'best_epoch': early_stopping.best_epoch,
                'stopped_early': early_stopping.early_stop
            }
        }, final_model_path)

        print(f"\nPretraining completed! (Trained {epoch + 1} epochs)")
        print(f"Best validation loss: {best_val_loss:.4f} (epoch {early_stopping.best_epoch})")
        print(f"Best model: {best_model_path}")
        print(f"Final model saved to: {final_model_path}")

        # 保存指标历史为CSV和JSON文件
        self.save_pretrain_metrics(metrics_history, epoch + 1)

        # 绘制预训练图表
        self.plot_pretrain_training_history(metrics_history)

        # 清理内存
        del pretrain_model
        torch.cuda.empty_cache()

        return best_model_path

    def train(self, pretrained_path=None):
        """训练完整的分割模型"""
        print("\n" + "=" * 60)
        print("Starting Training of Complete Segmentation Model")
        print("=" * 60)

        # 创建完整模型
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights: {pretrained_path}")
            model = create_complete_system(
                image_size=(512, 512),
                base_channels=self.config['base_channels'],
                num_position_classes=8,
                dropout_rate=self.config['dropout_rate'],
                max_radius=self.config.get('max_radius', 50),
                pretrained_path=pretrained_path
            ).to(self.device)

            # 冻结粗信息提取器（可选）
            if self.config.get('freeze_coarse', True):
                model.freeze_coarse_extractor()
                print("Coarse extractor frozen")
        else:
            print("Training from scratch")
            model = create_complete_system(
                image_size=(512, 512),
                base_channels=self.config['base_channels'],
                num_position_classes=8,
                dropout_rate=self.config['dropout_rate'],
                max_radius=self.config.get('max_radius', 50),
                pretrained_path=None
            ).to(self.device)

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders(pretrain_mode=False)

        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # 创建学习率调度器和早停器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        early_stopping = EarlyStopping(
            patience=self.config.get('train_early_stopping_patience', 20),
            min_delta=self.config.get('early_stopping_min_delta', 0.001),
            mode='max',  # 监控Dice分数，越大越好
            verbose=True
        )

        # 创建损失函数
        criterion = SegmentationLoss(
            dice_weight=0.5,
            bce_weight=0.5
        )

        # 训练循环
        best_val_dice = 0.0
        best_model_path = None
        metrics_history = []

        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            model.train()
            train_losses = {'total': 0, 'dice': 0, 'bce': 0}
            train_dice_scores = []

            train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{self.config["num_epochs"]}')
            for batch_idx, batch_data in enumerate(train_bar):
                images, masks, positions, _ = batch_data

                images = images.to(self.device)
                masks = masks.to(self.device)
                positions = positions.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(images, positions)
                segmentation = outputs['segmentation']

                # 计算损失
                loss_dict = criterion(segmentation, masks)
                loss = loss_dict['total_loss']

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 记录损失
                train_losses['total'] += loss.item()
                train_losses['dice'] += loss_dict['dice_loss'].item()
                train_losses['bce'] += loss_dict['bce_loss'].item()

                # 计算Dice分数
                with torch.no_grad():
                    dice_score = self.calculate_dice_score(segmentation, masks)
                    train_dice_scores.append(dice_score)

                # 更新进度条
                avg_loss = train_losses['total'] / (batch_idx + 1)
                avg_dice = np.mean(train_dice_scores)
                train_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'dice': f'{avg_dice:.4f}'
                })

            # 验证阶段
            model.eval()
            val_losses = {'total': 0, 'dice': 0, 'bce': 0}
            val_dice_scores = []

            with torch.no_grad():
                for batch_data in val_loader:
                    images, masks, positions, _ = batch_data

                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    positions = positions.to(self.device)

                    # 前向传播
                    outputs = model(images, positions)
                    segmentation = outputs['segmentation']

                    # 计算损失
                    loss_dict = criterion(segmentation, masks)

                    # 记录损失
                    val_losses['total'] += loss_dict['total_loss'].item()
                    val_losses['dice'] += loss_dict['dice_loss'].item()
                    val_losses['bce'] += loss_dict['bce_loss'].item()

                    # 计算Dice分数
                    dice_score = self.calculate_dice_score(segmentation, masks)
                    val_dice_scores.append(dice_score)

            # 计算平均指标
            train_loss_avg = train_losses['total'] / len(train_loader)
            val_loss_avg = val_losses['total'] / len(val_loader)
            train_dice_avg = np.mean(train_dice_scores)
            val_dice_avg = np.mean(val_dice_scores)

            # 更新学习率
            scheduler.step(val_dice_avg)

            # 早停检查（监控Dice分数）
            should_stop, improved = early_stopping(val_dice_avg, epoch + 1)

            # 记录指标历史
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'train_dice': train_dice_avg,
                'val_dice': val_dice_avg,
                'train_dice_loss': train_losses['dice'] / len(train_loader),
                'val_dice_loss': val_losses['dice'] / len(val_loader),
                'train_bce_loss': train_losses['bce'] / len(train_loader),
                'val_bce_loss': val_losses['bce'] / len(val_loader),
                'improved': improved,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # 记录学习率
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 打印统计信息
            print(f'\nEpoch {epoch + 1}/{self.config["num_epochs"]}:')
            print(f'  Train Loss: {train_loss_avg:.4f} (Dice: {train_dice_avg:.4f})')
            print(f'  Val Loss: {val_loss_avg:.4f} (Dice: {val_dice_avg:.4f})')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            if improved:
                print('  ✓ Model improved')
            else:
                print(f'  ⚠️ No improvement for {early_stopping.counter} epochs')

            # 保存最佳模型（如果有改善）
            if improved:
                best_val_dice = val_dice_avg
                best_model_path = self.checkpoint_dir / f'best_model.pth'

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'train_dice': train_dice_avg,
                    'val_dice': val_dice_avg,
                    'config': self.config,
                    'metrics_history': metrics_history
                }, best_model_path)

                print(f'  ✓ Saved best model to: {best_model_path} (Dice: {val_dice_avg:.4f})')

            # 每5个epoch保存检查点到CHP目录
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.chp_dir / f'train_checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'train_dice': train_dice_avg,
                    'val_dice': val_dice_avg,
                    'early_stopping_counter': early_stopping.counter,
                    'metrics_history': metrics_history
                }, checkpoint_path)
                print(f'  ✓ Saved checkpoint to CHP directory: {checkpoint_path}')

            # 保存训练历史
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss_avg)
            self.history['train_dice'].append(train_dice_avg)
            self.history['val_dice'].append(val_dice_avg)

            # 如果验证指标不再提升，考虑解冻粗信息提取器
            if epoch >= 30 and val_dice_avg < max(self.history['val_dice'][-5:]):
                if hasattr(model, 'coarse_extractor'):
                    frozen = not all(p.requires_grad for p in model.coarse_extractor.parameters())
                    if frozen:
                        print("  Validation metrics stagnated, unfreezing coarse extractor for fine-tuning...")
                        model.unfreeze_all()

            # 如果触发早停，跳出循环
            if should_stop:
                print(f"\n🚨 Early stopping triggered! Stopping training")
                break

        # 保存最终模型
        final_model_path = self.checkpoint_dir / 'final_model.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': self.history,
            'metrics_history': metrics_history,
            'early_stopping': {
                'best_value': early_stopping.best_value,
                'best_epoch': early_stopping.best_epoch,
                'stopped_early': early_stopping.early_stop
            }
        }, final_model_path)

        print(f"\nTraining completed! (Trained {epoch + 1} epochs)")
        print(f"Best validation Dice: {best_val_dice:.4f} (epoch {early_stopping.best_epoch})")
        print(f"Best model: {best_model_path}")
        print(f"Final model saved to: {final_model_path}")

        # 保存指标历史为CSV和JSON文件
        self.save_train_metrics(metrics_history, epoch + 1)

        # 绘制训练图表
        self.plot_training_history(metrics_history)

        return best_model_path

    def save_pretrain_metrics(self, metrics_history, total_epochs):
        """保存预训练指标数据"""
        try:
            # 保存为CSV
            csv_path = self.metrics_dir / 'pretrain_metrics.csv'
            metrics_df = pd.DataFrame(metrics_history)
            metrics_df.to_csv(csv_path, index=False)

            # 保存为JSON（更易读）
            json_path = self.metrics_dir / 'pretrain_metrics.json'
            with open(json_path, 'w') as f:
                json.dump({
                    'experiment_name': self.experiment_name,
                    'total_epochs': total_epochs,
                    'config': self.config,
                    'metrics': metrics_history,
                    'summary': {
                        'best_val_loss': min([m['val_loss'] for m in metrics_history]),
                        'best_epoch': min(range(len(metrics_history)),
                                          key=lambda i: metrics_history[i]['val_loss']) + 1,
                        'final_val_loss': metrics_history[-1]['val_loss'],
                        'final_learning_rate': metrics_history[-1]['learning_rate']
                    }
                }, f, indent=4, default=str)

            # 保存总结统计
            summary_path = self.metrics_dir / 'pretrain_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("PRETRAINING METRICS SUMMARY\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Experiment: {self.experiment_name}\n")
                f.write(f"Total Epochs: {total_epochs}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("Best Performance:\n")
                f.write("-" * 40 + "\n")
                best_idx = min(range(len(metrics_history)),
                               key=lambda i: metrics_history[i]['val_loss'])
                best_metrics = metrics_history[best_idx]
                f.write(f"Epoch: {best_metrics['epoch']}\n")
                f.write(f"Validation Loss: {best_metrics['val_loss']:.6f}\n")
                f.write(f"Training Loss: {best_metrics['train_loss']:.6f}\n")
                f.write(f"Height Loss: {best_metrics['val_height_loss']:.6f}\n")
                f.write(f"Radius Loss: {best_metrics['val_radius_loss']:.6f}\n")
                f.write(f"Learning Rate: {best_metrics['learning_rate']:.6f}\n\n")

                f.write("Final Performance:\n")
                f.write("-" * 40 + "\n")
                final_metrics = metrics_history[-1]
                f.write(f"Epoch: {final_metrics['epoch']}\n")
                f.write(f"Validation Loss: {final_metrics['val_loss']:.6f}\n")
                f.write(f"Training Loss: {final_metrics['train_loss']:.6f}\n")
                f.write(f"Height Loss: {final_metrics['val_height_loss']:.6f}\n")
                f.write(f"Radius Loss: {final_metrics['val_radius_loss']:.6f}\n")
                f.write(f"Learning Rate: {final_metrics['learning_rate']:.6f}\n")

            print(f"Pretraining metrics saved to: {csv_path}")
            print(f"Pretraining JSON metrics saved to: {json_path}")
            print(f"Pretraining summary saved to: {summary_path}")

        except Exception as e:
            print(f"Error saving pretrain metrics: {e}")

    def save_train_metrics(self, metrics_history, total_epochs):
        """保存主训练指标数据"""
        try:
            # 保存为CSV
            csv_path = self.metrics_dir / 'train_metrics.csv'
            metrics_df = pd.DataFrame(metrics_history)
            metrics_df.to_csv(csv_path, index=False)

            # 保存为JSON（更易读）
            json_path = self.metrics_dir / 'train_metrics.json'
            with open(json_path, 'w') as f:
                json.dump({
                    'experiment_name': self.experiment_name,
                    'total_epochs': total_epochs,
                    'config': self.config,
                    'metrics': metrics_history,
                    'summary': {
                        'best_val_dice': max([m['val_dice'] for m in metrics_history]),
                        'best_epoch': max(range(len(metrics_history)),
                                          key=lambda i: metrics_history[i]['val_dice']) + 1,
                        'final_val_dice': metrics_history[-1]['val_dice'],
                        'final_learning_rate': metrics_history[-1]['learning_rate']
                    }
                }, f, indent=4, default=str)

            # 保存总结统计
            summary_path = self.metrics_dir / 'train_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("TRAINING METRICS SUMMARY\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Experiment: {self.experiment_name}\n")
                f.write(f"Total Epochs: {total_epochs}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("Best Performance:\n")
                f.write("-" * 40 + "\n")
                best_idx = max(range(len(metrics_history)),
                               key=lambda i: metrics_history[i]['val_dice'])
                best_metrics = metrics_history[best_idx]
                f.write(f"Epoch: {best_metrics['epoch']}\n")
                f.write(f"Validation Dice: {best_metrics['val_dice']:.6f}\n")
                f.write(f"Training Dice: {best_metrics['train_dice']:.6f}\n")
                f.write(f"Validation Loss: {best_metrics['val_loss']:.6f}\n")
                f.write(f"Training Loss: {best_metrics['train_loss']:.6f}\n")
                f.write(f"Learning Rate: {best_metrics['learning_rate']:.6f}\n\n")

                f.write("Final Performance:\n")
                f.write("-" * 40 + "\n")
                final_metrics = metrics_history[-1]
                f.write(f"Epoch: {final_metrics['epoch']}\n")
                f.write(f"Validation Dice: {final_metrics['val_dice']:.6f}\n")
                f.write(f"Training Dice: {final_metrics['train_dice']:.6f}\n")
                f.write(f"Validation Loss: {final_metrics['val_loss']:.6f}\n")
                f.write(f"Training Loss: {final_metrics['train_loss']:.6f}\n")
                f.write(f"Learning Rate: {final_metrics['learning_rate']:.6f}\n")

            print(f"Training metrics saved to: {csv_path}")
            print(f"Training JSON metrics saved to: {json_path}")
            print(f"Training summary saved to: {summary_path}")

        except Exception as e:
            print(f"Error saving train metrics: {e}")

    def plot_pretrain_training_history(self, metrics_history):
        """绘制预训练历史图表（英文注释）"""
        try:
            if not metrics_history:
                print("No metrics history to plot")
                return

            epochs = [m['epoch'] for m in metrics_history]

            # 1. 训练和验证损失图
            plt.figure(figsize=(12, 8))

            # 主损失图
            plt.subplot(2, 2, 1)
            plt.plot(epochs, [m['train_loss'] for m in metrics_history],
                     'b-', linewidth=2, label='Train Loss')
            plt.plot(epochs, [m['val_loss'] for m in metrics_history],
                     'r-', linewidth=2, label='Validation Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

            # 2. 高度和半径损失图
            plt.subplot(2, 2, 2)
            plt.plot(epochs, [m['val_height_loss'] for m in metrics_history],
                     'g-', linewidth=2, label='Height Loss')
            plt.plot(epochs, [m['val_radius_loss'] for m in metrics_history],
                     'm-', linewidth=2, label='Radius Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Height and Radius Losses', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

            # 3. 学习率变化图
            plt.subplot(2, 2, 3)
            plt.plot(epochs, [m['learning_rate'] for m in metrics_history],
                     'c-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            # 4. 损失对比图
            plt.subplot(2, 2, 4)
            train_height = [m['train_height_loss'] for m in metrics_history]
            val_height = [m['val_height_loss'] for m in metrics_history]
            train_radius = [m['train_radius_loss'] for m in metrics_history]
            val_radius = [m['val_radius_loss'] for m in metrics_history]

            width = 0.2
            x = np.arange(len(epochs))

            plt.bar(x - 1.5 * width, train_height, width, label='Train Height', color='lightgreen', alpha=0.8)
            plt.bar(x - 0.5 * width, val_height, width, label='Val Height', color='darkgreen', alpha=0.8)
            plt.bar(x + 0.5 * width, train_radius, width, label='Train Radius', color='lightblue', alpha=0.8)
            plt.bar(x + 1.5 * width, val_radius, width, label='Val Radius', color='darkblue', alpha=0.8)

            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Height vs Radius Loss Comparison', fontsize=14, fontweight='bold')
            plt.legend(fontsize=9, loc='upper right')
            plt.xticks(x[::max(1, len(epochs) // 10)], epochs[::max(1, len(epochs) // 10)], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'pretrain_training_history.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.plots_dir / 'pretrain_training_history.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # 单独保存学习率图
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [m['learning_rate'] for m in metrics_history],
                     'c-', linewidth=3, marker='o', markersize=6)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Learning Rate', fontsize=14)
            plt.title('Learning Rate Schedule During Pretraining', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'pretrain_learning_rate.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Pretraining plots saved to: {self.plots_dir}")

        except Exception as e:
            print(f"Error plotting pretrain history: {e}")

    def plot_training_history(self, metrics_history):
        """绘制主训练历史图表（英文注释）"""
        try:
            if not metrics_history:
                print("No metrics history to plot")
                return

            epochs = [m['epoch'] for m in metrics_history]

            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 训练和验证损失图
            axes[0, 0].plot(epochs, [m['train_loss'] for m in metrics_history],
                            'b-', linewidth=2.5, label='Train Loss')
            axes[0, 0].plot(epochs, [m['val_loss'] for m in metrics_history],
                            'r-', linewidth=2.5, label='Validation Loss')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].legend(fontsize=11)
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 训练和验证Dice分数图
            axes[0, 1].plot(epochs, [m['train_dice'] for m in metrics_history],
                            'g-', linewidth=2.5, label='Train Dice')
            axes[0, 1].plot(epochs, [m['val_dice'] for m in metrics_history],
                            'm-', linewidth=2.5, label='Validation Dice')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
            axes[0, 1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])  # Dice分数在0-1之间

            # 3. 训练损失与Dice的联合图
            ax3 = axes[1, 0]
            line1 = ax3.plot(epochs, [m['train_loss'] for m in metrics_history],
                             'b-', linewidth=2.5, label='Train Loss')
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Training Loss', color='b', fontsize=12)
            ax3.tick_params(axis='y', labelcolor='b')

            ax3_2 = ax3.twinx()
            line2 = ax3_2.plot(epochs, [m['train_dice'] for m in metrics_history],
                               'g-', linewidth=2.5, label='Train Dice')
            ax3_2.set_ylabel('Training Dice', color='g', fontsize=12)
            ax3_2.tick_params(axis='y', labelcolor='g')
            ax3_2.set_ylim([0, 1])

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left', fontsize=11)
            ax3.set_title('Training Loss vs Dice Score', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # 4. 验证损失与Dice的联合图
            ax4 = axes[1, 1]
            line3 = ax4.plot(epochs, [m['val_loss'] for m in metrics_history],
                             'r-', linewidth=2.5, label='Validation Loss')
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Validation Loss', color='r', fontsize=12)
            ax4.tick_params(axis='y', labelcolor='r')

            ax4_2 = ax4.twinx()
            line4 = ax4_2.plot(epochs, [m['val_dice'] for m in metrics_history],
                               'm-', linewidth=2.5, label='Validation Dice')
            ax4_2.set_ylabel('Validation Dice', color='m', fontsize=12)
            ax4_2.tick_params(axis='y', labelcolor='m')
            ax4_2.set_ylim([0, 1])

            lines = line3 + line4
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left', fontsize=11)
            ax4.set_title('Validation Loss vs Dice Score', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.plots_dir / 'training_history.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # 单独保存学习率图
            self.plot_learning_rate_history(metrics_history)

            # 保存额外的详细图表
            self.plot_detailed_metrics(metrics_history)

            print(f"Training plots saved to: {self.plots_dir}")

        except Exception as e:
            print(f"Error plotting training history: {e}")

    def plot_learning_rate_history(self, metrics_history):
        """绘制学习率历史图（英文注释）"""
        try:
            epochs = [m['epoch'] for m in metrics_history]
            learning_rates = [m['learning_rate'] for m in metrics_history]

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, learning_rates, 'c-', linewidth=3, marker='o', markersize=6)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Learning Rate', fontsize=14)
            plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(epochs, learning_rates, 'c-', linewidth=3, marker='o', markersize=6)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Learning Rate (log scale)', fontsize=14)
            plt.title('Learning Rate Schedule (Log Scale)', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'learning_rate_history.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting learning rate history: {e}")

    def plot_detailed_metrics(self, metrics_history):
        """绘制详细指标图（英文注释）"""
        try:
            epochs = [m['epoch'] for m in metrics_history]

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Dice损失对比
            axes[0, 0].plot(epochs, [m['train_dice_loss'] for m in metrics_history],
                            'b-', linewidth=2, label='Train Dice Loss')
            axes[0, 0].plot(epochs, [m['val_dice_loss'] for m in metrics_history],
                            'r-', linewidth=2, label='Validation Dice Loss')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Dice Loss', fontsize=12)
            axes[0, 0].set_title('Dice Loss Comparison', fontsize=14, fontweight='bold')
            axes[0, 0].legend(fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)

            # 2. BCE损失对比
            axes[0, 1].plot(epochs, [m['train_bce_loss'] for m in metrics_history],
                            'g-', linewidth=2, label='Train BCE Loss')
            axes[0, 1].plot(epochs, [m['val_bce_loss'] for m in metrics_history],
                            'm-', linewidth=2, label='Validation BCE Loss')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('BCE Loss', fontsize=12)
            axes[0, 1].set_title('Binary Cross-Entropy Loss Comparison', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 训练集指标对比
            axes[1, 0].plot(epochs, [m['train_dice'] for m in metrics_history],
                            'b-', linewidth=2, label='Train Dice')
            axes[1, 0].plot(epochs, [m['train_dice_loss'] for m in metrics_history],
                            'r-', linewidth=2, label='Train Dice Loss')
            axes[1, 0].plot(epochs, [m['train_bce_loss'] for m in metrics_history],
                            'g-', linewidth=2, label='Train BCE Loss')
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Score/Loss', fontsize=12)
            axes[1, 0].set_title('Training Set Metrics', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 验证集指标对比
            axes[1, 1].plot(epochs, [m['val_dice'] for m in metrics_history],
                            'b-', linewidth=2, label='Validation Dice')
            axes[1, 1].plot(epochs, [m['val_dice_loss'] for m in metrics_history],
                            'r-', linewidth=2, label='Validation Dice Loss')
            axes[1, 1].plot(epochs, [m['val_bce_loss'] for m in metrics_history],
                            'g-', linewidth=2, label='Validation BCE Loss')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Score/Loss', fontsize=12)
            axes[1, 1].set_title('Validation Set Metrics', fontsize=14, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.plots_dir / 'detailed_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting detailed metrics: {e}")


def create_experiment_summary(config, experiment_dir):
    """创建实验摘要文件"""
    try:
        summary_path = experiment_dir / 'experiment_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("动脉瘤分割模型训练实验摘要\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"实验名称: {experiment_dir.name}\n")
            f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("配置参数:\n")
            f.write("-" * 40 + "\n")
            for key, value in config.items():
                f.write(f"{key:25s}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("目录结构:\n")
            f.write("-" * 40 + "\n")
            f.write(f"实验根目录: {experiment_dir}\n")
            f.write(f"预训练模型: {experiment_dir}/checkpoints/best_pretrain_model.pth\n")
            f.write(f"分割模型: {experiment_dir}/checkpoints/best_model.pth\n")
            f.write(f"检查点目录: {experiment_dir}/CHP/\n")
            f.write(f"日志目录: {experiment_dir}/logs/\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("训练策略:\n")
            f.write("-" * 40 + "\n")
            f.write("1. 预训练: 粗信息提取模块\n")
            f.write("   - 早停策略: 连续15个epoch未改善则停止\n")
            f.write("   - 检查点: 每5个epoch保存一次到CHP目录\n")
            f.write("   - 监控指标: 验证损失\n\n")

            f.write("2. 主训练: 完整分割模型\n")
            f.write("   - 早停策略: 连续20个epoch未改善则停止\n")
            f.write("   - 检查点: 每5个epoch保存一次到CHP目录\n")
            f.write("   - 监控指标: 验证Dice分数\n")
            f.write("   - 可选: 冻结粗信息提取器\n")

        print(f"实验摘要已保存到: {summary_path}")

    except Exception as e:
        print(f"创建实验摘要失败: {e}")


def main():
    """主函数"""
    print("动脉瘤分割模型训练程序")
    print("=" * 60)

    # 配置参数
    config = {
        # 数据路径
        'train_image_dir': "D:/med_data/ai/train1",
        'train_mask_dir': "D:/med_data/ai/train2",
        'val_image_dir': "D:/med_data/ai/test1",
        'val_mask_dir': "D:/med_data/ai/test2",

        # 位置信息文件
        'position_excel_path': "D:/med_data/ai/classify.xlsx",
        'location_excel_path': "D:/med_data/ai/location.xlsx",  # 预训练用

        # 模型参数
        'base_channels': 32,
        'dropout_rate': 0.2,
        'max_radius': 60,

        # 训练参数
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,

        # 预训练参数
        'pretrain': True,  # 是否进行预训练
        'pretrain_epochs': 40,  # 预训练轮数
        'pretrain_lr': 1e-3,  # 预训练学习率

        # 早停参数
        'early_stopping_patience': 5,  # 预训练早停耐心值
        'early_stopping_min_delta': 0.0005,  # 最小改善阈值
        'train_early_stopping_patience': 8,  # 主训练早停耐心值

        # 训练策略
        'freeze_coarse': True,  # 是否冻结粗信息提取器

        # 可选：限制样本数（用于调试）
        'max_train_samples': None,
        'max_val_samples': None,

        # 可选：工作线程数
        'num_workers': 2,
    }

    # 显示配置
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # 创建训练器
    trainer = ModelTrainer(config)

    # 创建实验摘要
    create_experiment_summary(config, trainer.experiment_dir)

    # 训练流程
    try:
        # 步骤1: 预训练（如果需要）
        pretrained_path = None
        if config['pretrain']:
            pretrained_path = trainer.pretrain()
            print(f"\n预训练完成，最佳模型: {pretrained_path}")

        # 步骤2: 训练完整模型
        best_model_path = trainer.train(pretrained_path)

        print(f"\n训练完成!")
        print(f"最佳模型: {best_model_path}")

        # 显示训练总结
        if trainer.history['val_dice']:
            final_dice = trainer.history['val_dice'][-1]
            best_dice = max(trainer.history['val_dice'])
            print(f"最终验证Dice: {final_dice:.4f}")
            print(f"最佳验证Dice: {best_dice:.4f}")
            print(f"实验目录: {trainer.experiment_dir}")
            print(f"所有模型和日志已保存到实验目录")

    except Exception as e:
        print(f"训练过程错误: {e}")
        traceback.print_exc()

    print("\n程序结束")


if __name__ == "__main__":
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 运行主函数
    main()