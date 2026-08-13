import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
import pydicom
from PIL import Image
import time
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import traceback
from sklearn.metrics import roc_curve, auc
import gc

# 导入你的模型
from multi.ves_U import EnhancedAttentionAwareUNet

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加安全的全局变量以支持PyTorch 2.6+
import torch.serialization

try:
    import numpy as np

    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    torch.serialization.add_safe_globals([np.ndarray])
except Exception as e:
    print(f"添加安全全局变量时警告: {e}")


class MedicalRecordPositionLoader:
    """基于病历号的位置信息加载器"""

    def __init__(self, excel_path="D:\\med_data\\ai\\classify.xlsx"):
        self.excel_path = excel_path
        self.position_dict = {}  # 病历号 -> 位置
        self._load_by_medical_record()

    def _load_by_medical_record(self):
        """按病历号加载位置信息"""
        try:
            # 使用pandas读取Excel
            df = pd.read_excel(self.excel_path, header=0)
            print(f"Excel列名: {df.columns.tolist()}")
            print(f"Excel前5行:\n{df.head()}")

            for idx, row in df.iterrows():
                try:
                    # 第一列：病历号
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

                    # 第二列：位置值
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
                    if idx < 5:  # 显示前5个记录
                        print(f"  病历号 {record_num}: 位置 {position_num}")

                except Exception as e:
                    continue

            print(f"位置信息加载完成: 共{len(self.position_dict)}个病历号")
            print(f"前10个病历号: {list(self.position_dict.keys())[:10]}")

        except Exception as e:
            print(f"加载位置信息失败: {e}")
            traceback.print_exc()
            self.position_dict = {}

    def extract_medical_record_from_filename(self, filename):
        """从文件名提取病历号"""
        try:
            # 文件名格式: "ANY_450_0" 或 "ANY_450_0.dcm"
            basename = os.path.splitext(filename)[0]  # 移除扩展名
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
            print(f"提取病历号失败 {filename}: {e}")
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
                    print(f"找到病历号 {record_num}: 位置 {position_num}")
                else:
                    # 检查是否翻转病例 (病历号 > 500)
                    if record_num > 500:
                        original_num = record_num - 500
                        if original_num in self.position_dict:
                            position_num = self.position_dict[original_num]
                            print(f"翻转病例 {record_num} -> {original_num}: 位置 {position_num}")
                        else:
                            position_num = 0
                            print(f"警告: 病历号 {record_num} 未找到位置信息")
                    else:
                        position_num = 0
                        print(f"警告: 病历号 {record_num} 未找到位置信息")

            # 创建位置张量 (one-hot编码)
            position_tensor = torch.zeros(8, dtype=torch.float32)
            if 0 <= position_num < 8:
                position_tensor[position_num] = 1.0

            return position_tensor, case_id

        except Exception as e:
            print(f"获取位置信息失败 {filename}: {e}")
            return torch.zeros(8, dtype=torch.float32), f"error_{filename}"


class DicomTestDataset(Dataset):
    """DICOM测试数据集，支持无后缀文件"""

    def __init__(self, image_dir, mask_dir, position_loader=None, max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.position_loader = position_loader
        self.max_samples = max_samples

        # 获取文件列表
        self.file_pairs = self._get_dicom_file_pairs()

        if self.file_pairs:
            if max_samples and len(self.file_pairs) > max_samples:
                self.file_pairs = self.file_pairs[:max_samples]

            print(f"找到 {len(self.file_pairs)} 个DICOM测试样本")
            for i, (img_file, mask_file) in enumerate(self.file_pairs[:5]):
                print(f"  样本{i + 1}: {img_file} -> {mask_file}")
        else:
            print("错误: 未找到任何DICOM文件对")
            # 显示目录内容
            print(f"\n图像目录内容 ({image_dir}):")
            img_files = os.listdir(image_dir)[:10]
            for f in img_files:
                print(f"  {f}")

            print(f"\nMask目录内容 ({mask_dir}):")
            mask_files = os.listdir(mask_dir)[:10]
            for f in mask_files:
                print(f"  {f}")

    def _get_dicom_file_pairs(self):
        """获取DICOM图像和mask文件对"""
        file_pairs = []

        try:
            # 获取所有图像文件（支持无后缀的DICOM文件）
            image_files = []
            for f in os.listdir(self.image_dir):
                file_path = os.path.join(self.image_dir, f)

                # 检查是否是DICOM文件
                if self._is_dicom_file(file_path):
                    image_files.append(f)
                # 或者如果文件名看起来像DICOM（无后缀且有数字）
                elif '_' in f and not '.' in f:
                    image_files.append(f)

            print(f"找到 {len(image_files)} 个可能的DICOM图像文件")

            # 获取所有mask文件
            mask_files = []
            for f in os.listdir(self.mask_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.dcm')):
                    mask_files.append(f)
                # 也检查无后缀文件
                elif '_' in f and not '.' in f:
                    mask_files.append(f)

            print(f"找到 {len(mask_files)} 个mask文件")

            if not image_files or not mask_files:
                return file_pairs

            # 创建基础名映射
            image_dict = {}
            for img_file in image_files:
                base_name = self._get_file_base_name(img_file)
                if base_name:
                    image_dict[base_name] = img_file
                    if len(image_dict) <= 5:  # 显示前5个映射
                        print(f"  图像映射: {img_file} -> {base_name}")

            # 匹配文件
            matched_count = 0
            for mask_file in mask_files:
                mask_base = self._get_file_base_name(mask_file)
                if mask_base in image_dict:
                    file_pairs.append((image_dict[mask_base], mask_file))
                    matched_count += 1
                    if matched_count <= 5:
                        print(f"  匹配成功: {image_dict[mask_base]} <-> {mask_file}")

            print(f"成功匹配 {matched_count} 个文件对")

        except Exception as e:
            print(f"获取文件对时出错: {e}")
            traceback.print_exc()

        return file_pairs

    def _get_file_base_name(self, filename):
        """获取文件的基础名（用于匹配）"""
        try:
            # 移除扩展名
            basename = os.path.splitext(filename)[0]

            # 格式: "ANY_450_0" -> 取前两部分 "ANY_450"
            parts = basename.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
            else:
                return basename

        except:
            return filename

    def _is_dicom_file(self, file_path):
        """检查文件是否是DICOM文件"""
        try:
            # 尝试读取为DICOM
            pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            return True
        except:
            # 检查文件头
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(132)  # DICOM文件应该有"DICM"在128字节处
                    if len(header) >= 132 and header[128:132] == b'DICM':
                        return True
            except:
                pass
            return False

    def _load_dicom_file(self, file_path):
        """加载DICOM文件"""
        try:
            dicom_data = pydicom.dcmread(file_path, force=True)
            image = dicom_data.pixel_array.astype(np.float32)

            # 处理可能的颜色通道
            if len(image.shape) == 3:
                # 如果是RGB，转换为灰度
                if image.shape[2] == 3:
                    image = np.mean(image, axis=2)
                elif image.shape[2] == 4:
                    image = np.mean(image[:, :, :3], axis=2)

            # 归一化到0-1
            img_min = np.min(image)
            img_max = np.max(image)
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min + 1e-8)
            else:
                image = np.zeros_like(image)

            return image

        except Exception as e:
            print(f"加载DICOM失败 {file_path}: {e}")
            # 返回默认图像
            return np.zeros((512, 512), dtype=np.float32)

    def _load_mask_file(self, file_path):
        """加载mask文件"""
        try:
            if file_path.lower().endswith('.dcm'):
                # 如果是DICOM格式的mask
                return self._load_dicom_file(file_path)
            else:
                # 普通图像文件
                img = Image.open(file_path)
                mask = np.array(img).astype(np.float32)

                # 转换为灰度
                if len(mask.shape) == 3:
                    mask = np.mean(mask, axis=2)

                # 二值化
                mask = (mask > 0.5).astype(np.float32)

                return mask

        except Exception as e:
            print(f"加载mask失败 {file_path}: {e}")
            return np.zeros((512, 512), dtype=np.float32)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_file, mask_file = self.file_pairs[idx]

        try:
            # 加载DICOM图像
            image_path = os.path.join(self.image_dir, image_file)
            image = self._load_dicom_file(image_path)

            # 加载mask
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = self._load_mask_file(mask_path)

            # 确保尺寸匹配
            if image.shape != mask.shape:
                # 调整mask尺寸
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            # 获取位置信息
            if self.position_loader:
                position_tensor, case_id = self.position_loader.get_position_for_image(image_file)
            else:
                position_tensor = torch.zeros(8, dtype=torch.float32)
                case_id = image_file

            # 添加通道维度
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # 转换为张量
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask).float()

            # 清理内存
            del image, mask
            if idx % 10 == 0:
                gc.collect()

            return image_tensor, mask_tensor, position_tensor, case_id, image_file

        except Exception as e:
            print(f"处理样本 {image_file} 失败: {e}")
            traceback.print_exc()
            # 返回默认张量
            dummy_image = torch.zeros((1, 512, 512), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            dummy_position = torch.zeros(8, dtype=torch.float32)
            return dummy_image, dummy_mask, dummy_position, "error", image_file


# 辅助函数保持不变
def calculate_dice_safe(preds, targets):
    try:
        preds_binary = (preds > 0.5).float()
        intersection = (preds_binary * targets).sum()
        union = preds_binary.sum() + targets.sum()
        dice = (2. * intersection) / (union + 1e-8)
        return dice.item()
    except:
        return 0.0


def calculate_iou_safe(preds, targets):
    try:
        preds_binary = (preds > 0.5).float()
        intersection = (preds_binary * targets).sum()
        union = preds_binary.sum() + targets.sum() - intersection
        iou = intersection / (union + 1e-8)
        return iou.item()
    except:
        return 0.0


def calculate_sensitivity_specificity_safe(preds, targets):
    try:
        preds_binary = (preds > 0.5).float()
        tp = (preds_binary * targets).sum()
        fp = (preds_binary * (1 - targets)).sum()
        tn = ((1 - preds_binary) * (1 - targets)).sum()
        fn = ((1 - preds_binary) * targets).sum()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        return sensitivity.item(), specificity.item()
    except:
        return 0.0, 0.0


def load_model_safely(model_path, device):
    print(f"加载模型: {os.path.basename(model_path)}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ 模型加载成功")

        model = EnhancedAttentionAwareUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            dropout_rate=0.1,
            use_attention=True,
            attention_strength=0.3,
            num_position_classes=8
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return None


class TestResultSaver:
    """测试结果保存器"""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建子文件夹
        (self.save_dir / "predictions").mkdir(exist_ok=True)
        (self.save_dir / "attention_maps").mkdir(exist_ok=True)
        (self.save_dir / "comparisons").mkdir(exist_ok=True)

        self.results = []

    def save_sample_result(self, filename, image, true_mask, pred_mask, attention_map,
                           dice, iou, sensitivity, specificity, case_id, position):
        """保存单个样本结果"""
        try:
            # 生成安全的文件名
            safe_name = re.sub(r'[^\w\-_.]', '_', filename)
            safe_name = safe_name[:50]  # 限制长度

            # 转换为numpy
            img_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image.squeeze()
            true_np = true_mask.squeeze().cpu().numpy() if isinstance(true_mask, torch.Tensor) else true_mask.squeeze()
            pred_np = pred_mask.squeeze().cpu().numpy() if isinstance(pred_mask, torch.Tensor) else pred_mask.squeeze()
            att_np = attention_map.squeeze().cpu().numpy() if isinstance(attention_map,
                                                                         torch.Tensor) else attention_map.squeeze()

            # 1. 保存预测结果
            plt.figure(figsize=(8, 6))
            plt.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title(f'预测结果\n病历号: {case_id}, 位置: {position}\nDice: {dice:.3f}, IoU: {iou:.3f}')
            plt.colorbar(label='预测概率')
            plt.savefig(self.save_dir / "predictions" / f'{safe_name}_pred.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # 2. 保存注意力图
            plt.figure(figsize=(8, 6))
            plt.imshow(att_np, cmap='hot')
            plt.axis('off')
            plt.title(f'注意力图\n病历号: {case_id}')
            plt.colorbar(label='注意力权重')
            plt.savefig(self.save_dir / "attention_maps" / f'{safe_name}_att.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # 3. 保存对比图
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # 原始图像
            axes[0, 0].imshow(img_np, cmap='gray')
            axes[0, 0].set_title('原始DICOM图像')
            axes[0, 0].axis('off')

            # 真实mask
            axes[0, 1].imshow(true_np, cmap='gray')
            axes[0, 1].set_title('真实分割')
            axes[0, 1].axis('off')

            # 预测结果
            axes[1, 0].imshow(pred_np, cmap='gray')
            axes[1, 0].set_title(f'预测结果 (Dice: {dice:.3f})')
            axes[1, 0].axis('off')

            # 叠加显示
            axes[1, 1].imshow(img_np, cmap='gray')
            overlay = axes[1, 1].imshow(pred_np > 0.5, cmap='Reds', alpha=0.5)
            axes[1, 1].set_title('预测叠加')
            axes[1, 1].axis('off')

            plt.suptitle(f'病历号: {case_id}, 文件: {filename}', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.save_dir / "comparisons" / f'{safe_name}_compare.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # 记录结果
            self.results.append({
                'filename': filename,
                'medical_record': case_id,
                'position': position,
                'dice': float(dice),
                'iou': float(iou),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'prediction_file': f'{safe_name}_pred.png',
                'attention_file': f'{safe_name}_att.png',
                'comparison_file': f'{safe_name}_compare.png'
            })

            print(f"  保存: {filename} -> Dice: {dice:.3f}, IoU: {iou:.3f}")

            # 清理内存
            del img_np, true_np, pred_np, att_np
            gc.collect()

        except Exception as e:
            print(f"保存样本 {filename} 结果失败: {e}")

    def save_final_report(self, avg_dice, avg_iou, avg_sens, avg_spec, auc_score):
        """保存最终报告"""
        try:
            # 保存CSV结果
            if self.results:
                df = pd.DataFrame(self.results)
                csv_path = self.save_dir / "test_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"✓ CSV结果保存到: {csv_path}")

            # 保存文本总结
            summary_path = self.save_dir / "test_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("脑动脉瘤分割模型测试结果总结\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总测试样本数: {len(self.results)}\n\n")

                f.write("总体性能指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均 Dice系数: {avg_dice:.4f}\n")
                f.write(f"  平均 IoU系数:  {avg_iou:.4f}\n")
                f.write(f"  平均 敏感性:    {avg_sens:.4f}\n")
                f.write(f"  平均 特异性:    {avg_spec:.4f}\n")
                f.write(f"  AUC:           {auc_score:.4f}\n\n")

                if self.results:
                    f.write("各样本详细结果:\n")
                    f.write("-" * 40 + "\n")
                    for result in self.results:
                        f.write(f"\n文件: {result['filename']}\n")
                        f.write(f"  病历号: {result['medical_record']}\n")
                        f.write(f"  位置:   {result['position']}\n")
                        f.write(f"  Dice:   {result['dice']:.4f}\n")
                        f.write(f"  IoU:    {result['iou']:.4f}\n")
                        f.write(f"  敏感性: {result['sensitivity']:.4f}\n")
                        f.write(f"  特异性: {result['specificity']:.4f}\n")

            print(f"✓ 文本总结保存到: {summary_path}")

        except Exception as e:
            print(f"保存最终报告失败: {e}")


def test_dicom_model(model_path, test_image_dir, test_mask_dir,
                     position_excel_path, batch_size=4,
                     result_base_dir="D:/med_data/ai/result"):
    """测试DICOM模型"""
    print("=" * 80)
    print("DICOM模型测试程序")
    print("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. 从模型路径提取结果文件夹名称
    try:
        model_path_obj = Path(model_path)
        experiment_name = model_path_obj.parent.parent.parent.name
        fold_name = model_path_obj.parent.parent.name
        result_folder = f"{experiment_name}_{fold_name}"
        result_dir = Path(result_base_dir) / result_folder

        print(f"模型: {model_path_obj.name}")
        print(f"实验: {experiment_name}")
        print(f"Fold: {fold_name}")
        print(f"结果目录: {result_dir}")

    except Exception as e:
        print(f"解析路径失败: {e}")
        result_dir = Path(result_base_dir) / "dicom_test_results"

    # 2. 加载模型
    print("\n1. 加载模型...")
    model = load_model_safely(model_path, device)
    if model is None:
        print("错误: 无法加载模型")
        return

    # 3. 加载位置信息
    print("\n2. 加载位置信息...")
    position_loader = MedicalRecordPositionLoader(position_excel_path)

    # 4. 创建DICOM数据集
    print("\n3. 创建DICOM测试数据集...")
    try:
        dataset = DicomTestDataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            position_loader=position_loader,
            max_samples=None  # 使用所有样本
        )

        if len(dataset) == 0:
            print("错误: DICOM数据集为空")
            return

        print(f"准备测试 {len(dataset)} 个DICOM样本")

    except Exception as e:
        print(f"创建DICOM数据集失败: {e}")
        traceback.print_exc()
        return

    # 5. 创建数据加载器
    test_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, 4),  # 限制批次大小
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"批次大小: {test_loader.batch_size}")
    print(f"总批次数: {len(test_loader)}")

    # 6. 创建结果保存器
    result_saver = TestResultSaver(result_dir)

    # 7. 开始测试
    print("\n4. 开始测试...")
    total_dice = 0.0
    total_iou = 0.0
    total_sensitivity = 0.0
    total_specificity = 0.0

    all_preds = []
    all_targets = []

    model.eval()

    try:
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="测试进度", unit="批次")
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    images, masks, positions, case_ids, filenames = batch_data

                    # 移动到设备
                    images = images.to(device)
                    masks = masks.to(device)
                    positions = positions.to(device)

                    # 前向传播
                    outputs, attention_maps = model(images, positions)

                    # 计算批次指标
                    batch_dices = []
                    batch_ious = []
                    batch_sensitivities = []
                    batch_specificities = []

                    for i in range(len(images)):
                        output_i = outputs[i:i + 1]
                        mask_i = masks[i:i + 1]

                        dice = calculate_dice_safe(output_i, mask_i)
                        iou = calculate_iou_safe(output_i, mask_i)
                        sensitivity, specificity = calculate_sensitivity_specificity_safe(output_i, mask_i)

                        batch_dices.append(dice)
                        batch_ious.append(iou)
                        batch_sensitivities.append(sensitivity)
                        batch_specificities.append(specificity)

                        total_dice += dice
                        total_iou += iou
                        total_sensitivity += sensitivity
                        total_specificity += specificity

                        # 收集预测
                        all_preds.extend(output_i.cpu().numpy().flatten())
                        all_targets.extend(mask_i.cpu().numpy().flatten())

                    # 保存批次结果
                    for i in range(len(images)):
                        result_saver.save_sample_result(
                            filename=filenames[i],
                            image=images[i],
                            true_mask=masks[i],
                            pred_mask=outputs[i],
                            attention_map=attention_maps[i] if attention_maps is not None else torch.zeros_like(
                                images[i]),
                            dice=batch_dices[i],
                            iou=batch_ious[i],
                            sensitivity=batch_sensitivities[i],
                            specificity=batch_specificities[i],
                            case_id=case_ids[i],
                            position=torch.argmax(positions[i]).item() if positions[i].dim() > 0 else positions[
                                i].item()
                        )

                    # 更新进度条
                    current_avg_dice = total_dice / ((batch_idx + 1) * len(images))
                    current_avg_iou = total_iou / ((batch_idx + 1) * len(images))
                    pbar.set_postfix({
                        '平均Dice': f'{current_avg_dice:.3f}',
                        '平均IoU': f'{current_avg_iou:.3f}'
                    })

                    # 清理内存
                    del images, masks, positions, outputs, attention_maps
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if batch_idx % 5 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"\n批次 {batch_idx} 处理失败: {e}")
                    traceback.print_exc()
                    continue

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程出错: {e}")
        traceback.print_exc()

    # 8. 计算最终指标
    num_samples = len(dataset)
    if num_samples > 0:
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
        avg_sensitivity = total_sensitivity / num_samples
        avg_specificity = total_specificity / num_samples
    else:
        avg_dice = avg_iou = avg_sensitivity = avg_specificity = 0.0

    # 9. 计算AUC
    auc_score = 0.0
    if len(all_preds) > 0 and len(np.unique(all_targets)) > 1:
        try:
            fpr, tpr, _ = roc_curve(all_targets, all_preds)
            auc_score = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率 (False Positive Rate)')
            plt.ylabel('真阳性率 (True Positive Rate)')
            plt.title('ROC曲线')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "roc_curve.png", dpi=120, bbox_inches='tight')
            plt.close()
            print(f"✓ ROC曲线保存到: {result_dir}/roc_curve.png")

        except Exception as e:
            print(f"计算AUC失败: {e}")

    # 10. 保存最终报告
    print("\n5. 保存测试结果...")
    result_saver.save_final_report(
        avg_dice=avg_dice,
        avg_iou=avg_iou,
        avg_sens=avg_sensitivity,
        avg_spec=avg_specificity,
        auc_score=auc_score
    )

    # 11. 打印总结
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    print(f"测试样本数: {num_samples}")
    print(f"平均 Dice系数: {avg_dice:.4f}")
    print(f"平均 IoU系数:  {avg_iou:.4f}")
    print(f"平均 敏感性:    {avg_sensitivity:.4f}")
    print(f"平均 特异性:    {avg_specificity:.4f}")
    print(f"AUC:           {auc_score:.4f}")
    print(f"\n结果目录: {result_dir}")

    # 显示文件夹结构
    if result_dir.exists():
        print("\n生成的文件结构:")
        for item in sorted(result_dir.glob("*")):
            if item.is_file():
                size_kb = item.stat().st_size / 1024
                print(f"  📄 {item.name} ({size_kb:.1f} KB)")
            elif item.is_dir():
                file_count = len(list(item.glob("*")))
                print(f"  📁 {item.name}/ ({file_count} 个文件)")


def main():
    """主函数"""
    print("DICOM脑动脉瘤分割模型测试程序")
    print("-" * 60)

    # 配置参数
    config = {
        'model_path': "D:/med_data/ai/model/20260201_795/folds/fold3/models/model_fold_3.pth",
        'test_image_dir': "D:/med_data/ai/test1",
        'test_mask_dir': "D:/med_data/ai/test2",
        'position_excel_path': "D:/med_data/ai/classify.xlsx",
        'batch_size': 4,
        'result_base_dir': "D:/med_data/ai/result"
    }

    # 显示配置
    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # 运行测试
    try:
        test_dicom_model(**config)
    except Exception as e:
        print(f"测试过程发生错误: {e}")
        traceback.print_exc()

    print("\n程序结束")


if __name__ == "__main__":
    # 添加内存清理
    import gc

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 运行主函数
    main()