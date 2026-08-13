import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
from unet import LightMultiScale_UNet3D_3
import pydicom
from PIL import Image
import torch.nn.functional as F
import time


# 自定义数据集类 - 按需读取文件
class AneurysmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # 获取所有匹配的文件对
        if file_list is None:
            self.samples = self._find_matching_files()
        else:
            self.samples = file_list

        print(f"find {len(self.samples)} samples")

    def _find_matching_files(self):

        samples = []

        # 获取DICOM文件列表
        dicom_files = [f for f in os.listdir(self.image_dir) if f.endswith('.dcm')]

        for dicom_file in dicom_files:
            base_name = os.path.splitext(dicom_file)[0]
            mask_file = base_name + '.tif'
            mask_path = os.path.join(self.mask_dir, mask_file)

            if os.path.exists(mask_path):
                samples.append((dicom_file, mask_file))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dicom_file, mask_file = self.samples[idx]

        try:
            # 读取DICOM图像
            dicom_path = os.path.join(self.image_dir, dicom_file)
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)

            # 读取TIF掩码
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = np.array(Image.open(mask_path)).astype(np.float32)

            # 归一化
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            mask = (mask > 0).astype(np.float32)  # 二值化掩码

            # 添加通道维度 (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # 转换为Tensor
            image_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask)

            return image_tensor, mask_tensor, dicom_file

        except Exception as e:
            print(f"Err reading: {dicom_file}, {mask_file}, err: {e}")
            # 返回空数据
            dummy_data = torch.zeros((1, 512, 512))
            return dummy_data, dummy_data, "error_file"

# 简单的数据增强（保留定义，主函数中不使用）
class SimpleTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # 随机水平翻转
        if torch.rand(1) > 0.5:
            x = torch.flip(x, [2])
        # 随机垂直翻转
        if torch.rand(1) > 0.5:
            x = torch.flip(x, [1])
        return x

# 计算准确率
def calculate_accuracy(preds, targets):
    preds_binary = (preds > 0.5).float()
    correct = (preds_binary == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

# 计算Dice系数
def calculate_dice(preds, targets):
    preds_binary = (preds > 0.5).float()
    intersection = (preds_binary * targets).sum()
    union = preds_binary.sum() + targets.sum()
    dice = (2. * intersection) / (union + 1e-8)
    return dice.item()

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc='Training')

    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, masks)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算指标
        accuracy = calculate_accuracy(outputs, masks)
        dice = calculate_dice(outputs, masks)

        # 更新统计
        running_loss += loss.item()
        running_accuracy += accuracy
        running_dice += dice

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.4f}',
            'Dice': f'{dice:.4f}'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_accuracy, epoch_dice


# 验证函数
def validate_epoch(model, dataloader, criterion, device, phase="Validation"):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_dice = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=phase)
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            accuracy = calculate_accuracy(outputs, masks)
            dice = calculate_dice(outputs, masks)

            running_loss += loss.item()
            running_accuracy += accuracy
            running_dice += dice

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'Dice': f'{dice:.4f}'
            })

    val_loss = running_loss / len(dataloader)
    val_accuracy = running_accuracy / len(dataloader)
    val_dice = running_dice / len(dataloader)

    return val_loss, val_accuracy, val_dice


# 早停类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0.0
        self.best_loss = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f'验证损失改善，保存模型到: {self.checkpoint_path}')


def main():
    # 设置参数
    batch_size = 4
    num_epochs = 100
    learning_rate = 1e-4
    patience = 5  # 早停耐心值

    # 路径设置
    train_image_dir = "D:/ai/train_0"  # 训练集DICOM
    train_mask_dir = "D:/ai/train_1"  # 训练集掩码
    val_image_dir = "D:/ai/val_0"  # 验证集DICOM
    val_mask_dir = "D:/ai/val_1"  # 验证集掩码
    test_image_dir = "D:/ai/test_0"  # 测试集DICOM
    test_mask_dir = "D:/ai/test_1"  # 测试集掩码

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集
    print("加载训练集...")
    train_dataset = AneurysmDataset(train_image_dir, train_mask_dir)

    print("加载验证集...")
    val_dataset = AneurysmDataset(val_image_dir, val_mask_dir)

    print("加载测试集...")
    test_dataset = AneurysmDataset(test_image_dir, test_mask_dir)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 创建模型
    model = LightMultiScale_UNet3D_3(in_channels=1, num_filters_start=32, dropout_rate=0.4)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # 早停器
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        checkpoint_path='best_aneurysm_model.pth'
    )

    print(f"\n开始训练，共 {num_epochs} 个epoch")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"早停耐心值: {patience}")

    # 记录训练历史
    train_history = {
        'loss': [], 'accuracy': [], 'dice': [],
        'val_loss': [], 'val_accuracy': [], 'val_dice': []
    }

    # 训练循环
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # 训练阶段
        train_loss, train_acc, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证阶段
        val_loss, val_acc, val_dice = validate_epoch(
            model, val_loader, criterion, device, "Validation"
        )

        # 记录历史
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)
        train_history['dice'].append(train_dice)
        train_history['val_loss'].append(val_loss)
        train_history['val_accuracy'].append(val_acc)
        train_history['val_dice'].append(val_dice)

        # 打印epoch结果
        print(f"\nEpoch {epoch} 结果:")
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Dice: {train_dice:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Dice: {val_dice:.4f}")

        # 更新学习率
        scheduler.step(val_loss)

        # 早停检查
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"\n早停触发! 在 epoch {epoch} 停止训练")
            break

        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")

    # 训练结束，加载最佳模型进行测试
    print("\n训练完成，加载最佳模型进行测试...")
    model.load_state_dict(torch.load('best_aneurysm_model.pth'))

    # 在测试集上评估
    test_loss, test_acc, test_dice = validate_epoch(
        model, test_loader, criterion, device, "Testing"
    )

    print(f"\n{'=' * 50}")
    print("最终测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试Dice系数: {test_dice:.4f}")
    print(f"{'=' * 50}")

    # 保存最终模型和训练历史
    final_model_path = 'final_aneurysm_model.pth'
    torch.save(model.state_dict(), final_model_path)

    # 保存训练历史
    history_path = 'training_history.npy'
    np.save(history_path, train_history)

    total_time = time.time() - start_time
    print(f"\n总训练时间: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")
    print(f"最佳模型已保存: best_aneurysm_model.pth")
    print(f"最终模型已保存: {final_model_path}")
    print(f"训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
