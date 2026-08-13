import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import datetime
import torch.nn.functional as F
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# 修复OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class BrainMRIDataset3DWithFeatures(Dataset):
    """3D脑部MRI数据集，支持图像切块和特征数据加载"""

    def __init__(self, t1w_dir, t2w_dir, features_csv, patch_size=(64, 64, 64), overlap=(16, 16, 16)):
        self.t1w_dir = t1w_dir
        self.t2w_dir = t2w_dir
        self.patch_size = patch_size
        self.overlap = overlap

        # 加载特征数据
        self.features_df = pd.read_csv(features_csv)
        print(f"加载了 {len(self.features_df)} 个病例的特征数据")

        # 选择的8个特征
        self.selected_features = [
            'mean_intensity', 'std_intensity', 'skewness_norm',
            'kurtosis_norm', 'iqr', 'mad', 'energy_norm', 'contrast'
        ]

        # 验证所有需要的特征都存在
        missing_features = [feat for feat in self.selected_features if feat not in self.features_df.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")

        # 创建病例特征映射
        self.case_features = {}
        for idx, row in self.features_df.iterrows():
            case_id = row['case_id']
            features = row[self.selected_features].values.astype(np.float32)
            self.case_features[case_id] = features

        # 获取匹配的文件对
        self.file_pairs = self.get_matching_file_pairs(t1w_dir, t2w_dir)
        self.patches_info = []  # 存储 (文件名, 块索引, 位置, 特征) 信息
        self.image_shapes = {}  # 存储图像形状以便快速计算

        print(f"找到 {len(self.file_pairs)} 个匹配的3D图像对")
        print("正在预处理图像并生成块信息...")

        # 预处理所有图像并生成块信息
        for t1w_file, t2w_file in tqdm(self.file_pairs):
            # 提取病例ID
            case_id = t1w_file.split('_')[0]

            # 获取特征数据
            if case_id not in self.case_features:
                print(f"警告: 病例 {case_id} 的特征数据不存在，跳过")
                continue

            features = self.case_features[case_id]

            # 加载T1w图像以确定块数量
            t1w_path = os.path.join(self.t1w_dir, t1w_file)
            t1w_img = nib.load(t1w_path)
            t1w_data = t1w_img.get_fdata().astype(np.float32)

            # 存储图像形状
            self.image_shapes[t1w_file] = t1w_data.shape

            # 计算块的数量
            d, h, w = t1w_data.shape
            patch_d, patch_h, patch_w = patch_size
            overlap_d, overlap_h, overlap_w = overlap

            stride_d = patch_d - overlap_d
            stride_h = patch_h - overlap_h
            stride_w = patch_w - overlap_w

            num_patches_d = max(1, (d - overlap_d) // stride_d) if d > patch_d else 1
            num_patches_h = max(1, (h - overlap_h) // stride_h) if h > patch_h else 1
            num_patches_w = max(1, (w - overlap_w) // stride_w) if w > patch_w else 1

            num_patches = num_patches_d * num_patches_h * num_patches_w

            # 为每个块存储信息
            for patch_idx in range(num_patches):
                self.patches_info.append({
                    't1w_file': t1w_file,
                    't2w_file': t2w_file,
                    'case_id': case_id,
                    'features': features.copy(),  # 特征副本
                    'patch_idx': patch_idx,
                    'total_patches': num_patches,  # 记录这个文件有多少个块
                    'shape': t1w_data.shape
                })

        print(f"共生成 {len(self.patches_info)} 个图像块")
        print(f"每个块共享同一个病例的特征数据")

    def get_matching_file_pairs(self, t1w_dir, t2w_dir):
        """获取匹配的T1和T2文件对"""
        t1w_files = [f for f in os.listdir(t1w_dir) if f.endswith('_T1.nii.gz')]
        file_pairs = []

        for t1w_file in t1w_files:
            # 提取subject ID，如s001
            subject_id = t1w_file.split('_')[0]
            t2w_file = f"{subject_id}_T2.nii.gz"
            t2w_path = os.path.join(t2w_dir, t2w_file)

            if os.path.exists(t2w_path):
                file_pairs.append((t1w_file, t2w_file))
            else:
                print(f"警告: 找不到对应的T2文件 {t2w_file} 对于 {t1w_file}")

        return file_pairs

    def calculate_patch_position(self, patch_idx, total_patches, image_shape):
        """计算块在图像中的位置"""
        d, h, w = image_shape
        patch_d, patch_h, patch_w = self.patch_size
        overlap_d, overlap_h, overlap_w = self.overlap

        # 计算步长
        stride_d = patch_d - overlap_d
        stride_h = patch_h - overlap_h
        stride_w = patch_w - overlap_w

        # 计算每个维度上的块数量
        num_patches_d = max(1, (d - overlap_d) // stride_d) if d > patch_d else 1
        num_patches_h = max(1, (h - overlap_h) // stride_h) if h > patch_h else 1
        num_patches_w = max(1, (w - overlap_w) // stride_w) if w > patch_w else 1

        # 根据patch_idx计算3D位置
        idx = patch_idx
        k = idx % num_patches_w
        idx = idx // num_patches_w
        j = idx % num_patches_h
        i = idx // num_patches_h

        # 计算块起始位置
        d_start = min(i * stride_d, d - patch_d)
        h_start = min(j * stride_h, h - patch_h)
        w_start = min(k * stride_w, w - patch_w)

        # 确保不超出边界
        d_start = max(0, d_start)
        h_start = max(0, h_start)
        w_start = max(0, w_start)

        return d_start, h_start, w_start

    def create_patch_from_image(self, image_path, patch_idx, image_shape):
        """从图像中提取指定的块"""
        # 加载图像
        img = nib.load(image_path)
        data = img.get_fdata().astype(np.float32)

        # 计算块位置
        d_start, h_start, w_start = self.calculate_patch_position(patch_idx, 0, image_shape)
        patch_d, patch_h, patch_w = self.patch_size

        # 提取块
        patch = data[d_start:d_start + patch_d, h_start:h_start + patch_h, w_start:w_start + patch_w]

        # 如果块尺寸不足，进行填充
        if patch.shape != self.patch_size:
            pad_d = patch_d - patch.shape[0]
            pad_h = patch_h - patch.shape[1]
            pad_w = patch_w - patch.shape[2]

            patch = np.pad(patch,
                           ((0, pad_d), (0, pad_h), (0, pad_w)),
                           mode='constant', constant_values=0)

        return patch

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        patch_info = self.patches_info[idx]

        # 按需加载图像并提取对应的块
        t1w_path = os.path.join(self.t1w_dir, patch_info['t1w_file'])
        t2w_path = os.path.join(self.t2w_dir, patch_info['t2w_file'])

        # 提取T1w和T2w块
        t1w_patch = self.create_patch_from_image(t1w_path, patch_info['patch_idx'], patch_info['shape'])
        t2w_patch = self.create_patch_from_image(t2w_path, patch_info['patch_idx'], patch_info['shape'])

        # 转换为tensor并添加通道维度
        t1w_patch = torch.from_numpy(t1w_patch).float().unsqueeze(0)  # [1, D, H, W]
        t2w_patch = torch.from_numpy(t2w_patch).float().unsqueeze(0)

        # 特征数据
        features = torch.from_numpy(patch_info['features']).float()

        # 标识信息
        identifier = f"{patch_info['case_id']}_patch{patch_info['patch_idx']}"

        return t1w_patch, t2w_patch, features, identifier, patch_info['case_id'], patch_info['total_patches']

class CombinedLoss(nn.Module):
    """MSE + SSIM 加权综合损失函数"""

    def __init__(self, mse_weight=0.7, ssim_weight=0.3, data_range=1.0):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = nn.MSELoss()
        self.data_range = data_range

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)

        # 简化的SSIM计算
        mu_x = torch.mean(pred, dim=[2, 3, 4], keepdim=True)
        mu_y = torch.mean(target, dim=[2, 3, 4], keepdim=True)
        sigma_x = torch.var(pred, dim=[2, 3, 4], keepdim=True, unbiased=False)
        sigma_y = torch.var(target, dim=[2, 3, 4], keepdim=True, unbiased=False)
        sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=[2, 3, 4], keepdim=True)

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim_value = torch.mean(ssim_numerator / ssim_denominator)
        ssim_loss = 1 - ssim_value

        combined_loss = self.mse_weight * mse + self.ssim_weight * ssim_loss
        return combined_loss, mse, ssim_loss


class OptimizerManager:
    """管理单个优化器，同时优化所有参数"""

    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(),  # 优化所有参数
            lr=lr,
            weight_decay=1e-4
        )
        print(f"优化器参数数: {sum(p.numel() for p in model.parameters()):,}")

    def zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()

    def step(self):
        """更新所有参数"""
        self.optimizer.step()


class DualTrainingDataloader:
    """简化版数据加载器"""

    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 按病例组织数据
        self.case_to_indices = {}
        for idx in range(len(dataset)):
            patch_info = dataset.patches_info[idx]
            case_id = patch_info['case_id']
            if case_id not in self.case_to_indices:
                self.case_to_indices[case_id] = []
            self.case_to_indices[case_id].append(idx)

        self.case_ids = list(self.case_to_indices.keys())
        self.current_case_idx = 0
        self.current_patch_idx = 0

        # 打乱病例顺序
        if self.shuffle:
            random.shuffle(self.case_ids)

        print(f"数据加载器初始化完成，共 {len(self.case_ids)} 个病例")

    def shuffle_cases(self):
        """打乱病例顺序"""
        if self.shuffle:
            random.shuffle(self.case_ids)
        self.current_case_idx = 0
        self.current_patch_idx = 0

    def shuffle_patches_in_current_case(self):
        """打乱当前病例内的块顺序"""
        current_case = self.case_ids[self.current_case_idx]
        if self.shuffle:
            random.shuffle(self.case_to_indices[current_case])
        self.current_patch_idx = 0

    def get_training_batch(self):
        """获取训练批次"""
        current_case = self.case_ids[self.current_case_idx]
        patch_indices = self.case_to_indices[current_case]

        # 获取当前病例的批次
        batch_size = min(self.batch_size, len(patch_indices))
        if batch_size == 0:
            # 如果没有数据，转到下一个病例
            self.next_case()
            return self.get_training_batch()

        # 随机选择批次索引
        batch_indices = random.sample(patch_indices, batch_size)
        batch_data = [self.dataset[idx] for idx in batch_indices]

        # 移动到下一个病例（每次处理完一个病例）
        self.current_patch_idx += batch_size
        if self.current_patch_idx >= len(patch_indices):
            self.next_case()

        return self.collate_batch(batch_data)

    def get_main_training_batch(self):
        """兼容验证函数的方法名（直接调用get_training_batch）"""
        return self.get_training_batch()

    def next_case(self):
        """移动到下一个病例"""
        self.current_case_idx = (self.current_case_idx + 1) % len(self.case_ids)
        self.current_patch_idx = 0
        return self.case_ids[self.current_case_idx]

    def collate_batch(self, batch_data):
        """整理批次数据"""
        t1w_batch = []
        t2w_batch = []
        features_batch = []
        identifiers = []
        case_ids = []

        for t1w, t2w, features, identifier, case_id, total_patches in batch_data:
            t1w_batch.append(t1w)
            t2w_batch.append(t2w)
            features_batch.append(features)
            identifiers.append(identifier)
            case_ids.append(case_id)

        return (
            torch.stack(t1w_batch),
            torch.stack(t2w_batch),
            torch.stack(features_batch),
            identifiers,
            case_ids
        )

    def __len__(self):
        # 估计总批次数
        total_patches = len(self.dataset)
        return (total_patches + self.batch_size - 1) // self.batch_size


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """标准训练epoch：同时优化所有参数"""
    model.train()
    total_loss = 0
    total_batches = 0

    # 使用完整数据集
    num_batches = len(dataloader)
    pbar = tqdm(range(num_batches), desc=f'Epoch {epoch + 1} - Training')

    for batch_idx in pbar:
        try:
            # 获取训练批次数据
            t1w, t2w, features, _, _ = dataloader.get_training_batch()
            t1w = t1w.to(device)
            t2w = t2w.to(device)
            features = features.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(t1w, features, modulation_mode='all')
            loss, mse, ssim = criterion(outputs, t2w)

            # 反向传播
            loss.backward()

            # 更新所有参数
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MSE': f'{mse.item():.4f}',
                'SSIM': f'{ssim.item():.4f}',
                'Case': dataloader.case_ids[dataloader.current_case_idx]
            })

        except Exception as e:
            print(f"训练批次 {batch_idx} 出错: {e}")
            continue

    # 计算平均损失
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


def validate_epoch(model, dataloader, criterion, device):
    """验证epoch"""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_ssim = 0
    num_batches = 0

    with torch.no_grad():
        num_val_batches = min(20, len(dataloader))
        pbar = tqdm(range(num_val_batches), desc='Validation')

        for _ in pbar:
            try:
                # 修改这里：使用get_training_batch()而不是get_main_training_batch()
                t1w, t2w, features, _, _ = dataloader.get_training_batch()
                t1w = t1w.to(device)
                t2w = t2w.to(device)
                features = features.to(device)

                # 前向传播
                outputs = model(t1w, features, modulation_mode='all')
                loss, mse, ssim = criterion(outputs, t2w)

                total_loss += loss.item()
                total_mse += mse.item()
                total_ssim += ssim.item()
                num_batches += 1

                pbar.set_postfix({
                    'Val_Loss': f'{loss.item():.4f}',
                    'Val_MSE': f'{mse.item():.4f}'
                })
            except Exception as e:
                print(f"验证批次出错: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)
    avg_mse = total_mse / max(num_batches, 1)
    avg_ssim = total_ssim / max(num_batches, 1)

    return avg_loss, avg_mse, avg_ssim


class EarlyStopping:
    """早停策略（简化版）"""

    def __init__(self, patience=10, min_delta=0, checkpoint_dir='checkpoints'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, optimizer, epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model, optimizer, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))
        print(f"保存最佳模型，验证损失: {self.best_loss:.6f}")

class InterleavedTrainingMonitor:
    """监控交织式网络的训练过程"""

    def __init__(self, model, save_dir=None):
        self.model = model
        self.save_dir = Path(save_dir) if save_dir else Path("training_monitor")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'modulation_freq': {stage: [] for stage in model.modulation_history.keys()},
            'top_channels_history': {stage: [] for stage in model.modulation_history.keys()}
        }

        print(f"训练监控器初始化完成")

    def analyze_modulation_patterns(self, epoch):
        """分析调制模式 - 修复版本"""
        insights = {}

        for stage in self.model.modulation_history.keys():
            records = self.model.modulation_history[stage]

            if not records:
                continue

            # 分析该阶段最常被调制的通道
            channel_counts = np.zeros(self._get_stage_channels(stage))

            # 使用最近的记录
            recent_records = records[-100:] if len(records) > 100 else records

            for record in recent_records:
                selected_channels = record['selected_channels']
                # 确保selected_channels是numpy数组
                if isinstance(selected_channels, list):
                    selected_channels = np.array(selected_channels)
                elif isinstance(selected_channels, torch.Tensor):
                    selected_channels = selected_channels.cpu().numpy()

                for ch in selected_channels.flatten():
                    channel_counts[int(ch)] += 1

            # 找出最活跃的通道
            n_top = min(10, len(channel_counts))
            top_indices = np.argsort(channel_counts)[-n_top:][::-1]

            insights[stage] = {
                'top_channels': top_indices.tolist(),
                'modulation_frequency': (channel_counts[top_indices] / len(recent_records)).tolist(),
                'stage_semantic': self._get_stage_semantic(stage),
                'total_modulations': len(recent_records),
                'unique_channels_modulated': np.sum(channel_counts > 0)
            }

            # 记录频率数据
            self.metrics['modulation_freq'][stage].append({
                'epoch': epoch,
                'top_channels': top_indices.tolist(),
                'top_frequencies': (channel_counts[top_indices] / len(recent_records)).tolist(),
                'avg_modulation_rate': channel_counts.mean() / len(recent_records),
                'channel_diversity': np.sum(channel_counts > 0) / len(channel_counts)
            })

        return insights

    def _get_stage_channels(self, stage):
        """获取各阶段的通道数"""
        channel_map = {
            'enc1': self.model.base_channels,
            'enc3': self.model.base_channels * 4,
            'dec2': self.model.base_channels * 4,
            'dec4': self.model.base_channels
        }
        return channel_map.get(stage, 0)

    def _get_stage_semantic(self, stage):
        """推断各阶段的语义意义"""
        semantics = {
            'enc1': '早期特征：边缘/简单结构',
            'enc3': '中期特征：器官部件/局部结构',
            'dec2': '解码中期：结构重建',
            'dec4': '解码后期：细节精修'
        }
        return semantics.get(stage, '未知')


def main():
    # 配置参数
    config = {
        't1w_dir': "D:\\med_data\\MR\\train_1",  # T1w加权图像
        't2w_dir': "D:\\med_data\\MR\\train_2",  # T2w加权图像（作为标签）
        'features_csv': "D:\\med_data\\MR\\train_1\\mri_features_precision4.csv",  # 特征数据
        'base_model_dir': "D:\\med_data\\MR\\interleaved_model",  # 基础模型目录
        'batch_size': 4,  # 减小batch_size以确保内存足够
        'lr': 1e-4,  # 统一学习率
        'num_epochs': 50,  # epoch数
        'patience': 5,  # 早停耐心值
        'train_ratio': 0.8,
        'patch_size': (64, 64, 64),
        'overlap': (32, 32, 32),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_dim': 8,  # 特征维度
        'base_channels': 32,
        'modulation_sparsity': 0.25
    }

    # 创建基于日期的子文件夹
    today = datetime.now().strftime("%Y%m%d")
    config['model_dir'] = os.path.join(config['base_model_dir'], today)

    # 创建模型目录和子目录
    os.makedirs(config['model_dir'], exist_ok=True)
    checkpoint_dir = os.path.join(config['model_dir'], 'checkpoints')
    log_dir = os.path.join(config['model_dir'], 'logs')
    monitor_dir = os.path.join(config['model_dir'], 'monitor')

    for subdir in [checkpoint_dir, log_dir, monitor_dir]:
        os.makedirs(subdir, exist_ok=True)

    print("=" * 60)
    print("交织式跨模态3D U-Net T1w到T2w转换训练（简化版）")
    print(f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型保存目录: {config['model_dir']}")
    print("=" * 60)
    print("配置信息:")
    for key, value in config.items():
        if key not in ['t1w_dir', 't2w_dir', 'features_csv', 'model_dir']:  # 隐藏长路径
            print(f"  {key}: {value}")

    # 保存配置到文件
    config_save_path = os.path.join(config['model_dir'], 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        # 将numpy数组转换为列表以便JSON序列化
        config_serializable = {k: v for k, v in config.items() if not isinstance(v, tuple)}
        config_serializable['patch_size'] = list(config['patch_size'])
        config_serializable['overlap'] = list(config['overlap'])
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)
    print(f"配置文件已保存: {config_save_path}")

    # 创建数据集
    print("\n创建数据集...")
    full_dataset = BrainMRIDataset3DWithFeatures(
        config['t1w_dir'],
        config['t2w_dir'],
        config['features_csv'],
        patch_size=config['patch_size'],
        overlap=config['overlap']
    )

    # 分割训练集和验证集（按病例分割）
    case_ids = list(set([info['case_id'] for info in full_dataset.patches_info]))
    random.shuffle(case_ids)
    split_idx = int(len(case_ids) * config['train_ratio'])
    train_case_ids = set(case_ids[:split_idx])
    val_case_ids = set(case_ids[split_idx:])

    # 创建训练和验证的patch索引
    train_indices = [i for i, info in enumerate(full_dataset.patches_info) if info['case_id'] in train_case_ids]
    val_indices = [i for i, info in enumerate(full_dataset.patches_info) if info['case_id'] in val_case_ids]

    print(f"数据集分割: 训练集 {len(train_indices)} 块, 验证集 {len(val_indices)} 块")
    print(f"训练病例: {len(train_case_ids)} 个, 验证病例: {len(val_case_ids)} 个")

    # 创建训练和验证数据集
    class SubsetDataset(Dataset):
        def __init__(self, full_dataset, indices):
            self.full_dataset = full_dataset
            self.indices = indices
            self.patches_info = [full_dataset.patches_info[i] for i in indices]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.full_dataset[self.indices[idx]]

    train_dataset = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(full_dataset, val_indices)

    # 创建数据加载器
    train_loader = DualTrainingDataloader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_loader = DualTrainingDataloader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # 导入模型
    print("\n初始化模型...")
    from multi.network import InterleavedCrossModalUNet3D

    model = InterleavedCrossModalUNet3D(
        image_in_channels=1,
        data_dim=config['data_dim'],
        base_channels=config['base_channels'],
        modulation_sparsity=config['modulation_sparsity'],
        dropout_rate=0.2
    )
    model = model.to(config['device'])

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

    # 保存模型结构信息
    model_info_path = os.path.join(config['model_dir'], 'model_info.txt')
    with open(model_info_path, 'w', encoding='utf-8') as f:
        f.write(f"模型名称: InterleavedCrossModalUNet3D\n")
        f.write(f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write("\n模型结构:\n")
        f.write(str(model))

    print(f"模型信息已保存: {model_info_path}")

    # 初始化优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-4
    )
    print(f"优化器: Adam(lr={config['lr']})")

    # 初始化损失函数
    criterion = CombinedLoss(mse_weight=0.7, ssim_weight=0.3)

    # 初始化监控器
    monitor = InterleavedTrainingMonitor(model, monitor_dir)

    # 早停策略
    early_stopping = EarlyStopping(patience=config['patience'], checkpoint_dir=checkpoint_dir)

    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_ssim': []
    }

    print("\n开始训练...")
    print("-" * 80)

    # 训练循环
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 50)

        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], epoch
        )

        # 验证
        val_loss, val_mse, val_ssim = validate_epoch(
            model, val_loader, criterion, config['device']
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_ssim'].append(val_ssim)

        # 打印epoch结果
        print(f"训练损失: {train_loss:.6f}")
        print(f"验证 - 总损失: {val_loss:.6f}, MSE: {val_mse:.6f}, SSIM: {val_ssim:.6f}")

        # 分析调制模式
        if epoch % 5 == 0 or epoch == config['num_epochs'] - 1:
            insights = monitor.analyze_modulation_patterns(epoch)
            if insights:
                print(f"\n调制模式分析 (Epoch {epoch}):")
                for stage, data in insights.items():
                    print(f"  {stage}: 最活跃通道 {data['top_channels'][:3]}")

        # 早停检查
        if early_stopping(val_loss, model, optimizer, epoch):
            print(f"\n早停触发! 最佳验证损失: {early_stopping.best_loss:.6f}")
            break

        # 每个epoch保存训练历史（用于恢复训练）
        epoch_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1:03d}.pth')
        torch.save(epoch_checkpoint, epoch_checkpoint_path)

    print("\n训练完成!")

    # 保存最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(config['model_dir'], f'final_model_{timestamp}.pth')

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config,
        'final_epoch': epoch + 1,
        'best_val_loss': early_stopping.best_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(final_checkpoint, final_model_path)
    print(f"最终模型已保存: {final_model_path}")

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config,
        'final_epoch': epoch + 1,
        'best_val_loss': early_stopping.best_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(final_checkpoint, final_model_path)
    print(f"最终模型已保存: {final_model_path}")

    # 保存训练历史为JSON
    history_path = os.path.join(config['model_dir'], f'training_history_{timestamp}.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json_ready_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump({
            'history': json_ready_history,
            'config': config_serializable,
            'best_val_loss': float(early_stopping.best_loss),
            'final_epoch': epoch + 1
        }, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存: {history_path}")

    # 保存训练历史为CSV（便于Excel分析）
    csv_history_path = os.path.join(config['model_dir'], f'training_history_{timestamp}.csv')
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_mse': history['val_mse'],
        'val_ssim': history['val_ssim']
    })
    history_df.to_csv(csv_history_path, index=False, encoding='utf-8-sig')
    print(f"训练历史(CSV)已保存: {csv_history_path}")

    # 绘制训练曲线
    plot_training_curves(history, config['model_dir'], timestamp)

    # 打印训练总结
    print(f"\n{'=' * 60}")
    print("训练总结:")
    print(f"{'=' * 60}")
    print(f"  总训练轮数: {epoch + 1}")
    print(f"  最佳验证损失: {early_stopping.best_loss:.6f}")
    print(f"  最终训练损失: {train_loss:.6f}")
    print(f"  最终验证损失: {val_loss:.6f}")
    print(f"  最终验证MSE: {val_mse:.6f}")
    print(f"  最终验证SSIM: {val_ssim:.6f}")
    print(f"  总图像块数量: {len(full_dataset)}")
    print(f"  训练图像块: {len(train_indices)}")
    print(f"  验证图像块: {len(val_indices)}")
    print(f"  模型保存目录: {config['model_dir']}")
    print(f"  最佳模型: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    print(f"  最终模型: {final_model_path}")
    print(f"{'=' * 60}")


def plot_training_curves(history, output_dir, timestamp):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))

    # 训练和验证损失
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失', linewidth=2)
    plt.plot(history['val_loss'], label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 验证MSE
    plt.subplot(1, 3, 2)
    plt.plot(history['val_mse'], label='验证MSE', linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('验证MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 验证SSIM
    plt.subplot(1, 3, 3)
    plt.plot(history['val_ssim'], label='验证SSIM', linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('验证SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(output_dir, f'training_curves_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线图已保存: {plot_path}")


if __name__ == "__main__":
    main()
