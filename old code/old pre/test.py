import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json

# 修复OpenMP冲突


# 导入模型
import sys

sys.path.append(os.path.dirname(__file__))
from network.unet3 import MultiScale_UNet3D


def create_overlapping_patches_3d(image, patch_size=(64, 64, 64), overlap=(32, 32, 32)):
    """
    创建有重叠的3D图像块（与训练时相同的切块函数）
    """
    d, h, w = image.shape
    patch_d, patch_h, patch_w = patch_size
    overlap_d, overlap_h, overlap_w = overlap

    # 计算步长
    stride_d = patch_d - overlap_d
    stride_h = patch_h - overlap_h
    stride_w = patch_w - overlap_w

    # 计算每个维度上的块数量
    num_patches_d = max(1, (d - overlap_d) // stride_d) if d > patch_d else 1
    num_patches_h = max(1, (h - overlap_h) // stride_h) if h > patch_h else 1
    num_patches_w = max(1, (w - overlap_w) // stride_w) if w > patch_w else 1

    # 调整步长以确保覆盖整个图像
    if num_patches_d > 1:
        stride_d = (d - patch_d) // (num_patches_d - 1) if num_patches_d > 1 else patch_d
    if num_patches_h > 1:
        stride_h = (h - patch_h) // (num_patches_h - 1) if num_patches_h > 1 else patch_h
    if num_patches_w > 1:
        stride_w = (w - patch_w) // (num_patches_w - 1) if num_patches_w > 1 else patch_w

    patches = []
    positions = []

    # 生成所有可能的块位置
    for i in range(num_patches_d):
        for j in range(num_patches_h):
            for k in range(num_patches_w):
                # 计算块起始位置
                d_start = min(i * stride_d, d - patch_d)
                h_start = min(j * stride_h, h - patch_h)
                w_start = min(k * stride_w, w - patch_w)

                # 确保不超出边界
                d_start = max(0, d_start)
                h_start = max(0, h_start)
                w_start = max(0, w_start)
                d_end = min(d_start + patch_d, d)
                h_end = min(h_start + patch_h, h)
                w_end = min(w_start + patch_w, w)

                # 如果块尺寸不足，从边缘开始填充
                if d_end - d_start < patch_d:
                    d_start = max(0, d - patch_d)
                    d_end = d
                if h_end - h_start < patch_h:
                    h_start = max(0, h - patch_h)
                    h_end = h
                if w_end - w_start < patch_w:
                    w_start = max(0, w - patch_w)
                    w_end = w

                # 提取块
                patch = image[d_start:d_end, h_start:h_end, w_start:w_end]

                # 如果块尺寸小于目标尺寸，进行填充
                if patch.shape != patch_size:
                    pad_d = patch_d - patch.shape[0]
                    pad_h = patch_h - patch.shape[1]
                    pad_w = patch_w - patch.shape[2]

                    patch = np.pad(patch,
                                   ((0, pad_d), (0, pad_h), (0, pad_w)),
                                   mode='constant', constant_values=0)

                patches.append(patch)
                positions.append((d_start, h_start, w_start, d_end, h_end, w_end))

    return patches, positions


def reconstruct_from_patches(patches, positions, original_shape):
    """
    从图像块重建完整图像
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_map = np.zeros(original_shape, dtype=np.float32)

    for patch, pos in zip(patches, positions):
        d_start, h_start, w_start, d_end, h_end, w_end = pos
        # 确保patch尺寸与目标区域匹配
        patch_d, patch_h, patch_w = patch.shape
        actual_d = min(patch_d, d_end - d_start)
        actual_h = min(patch_h, h_end - h_start)
        actual_w = min(patch_w, w_end - w_start)

        reconstructed[d_start:d_start + actual_d, h_start:h_start + actual_h, w_start:w_start + actual_w] += patch[
                                                                                                             :actual_d,
                                                                                                             :actual_h,
                                                                                                             :actual_w]
        count_map[d_start:d_start + actual_d, h_start:h_start + actual_h, w_start:w_start + actual_w] += 1

    # 避免除以0
    count_map[count_map == 0] = 1
    reconstructed = reconstructed / count_map

    return reconstructed


def calculate_metrics(pred, target):
    """
    计算PSNR, SSIM, MSE指标
    """
    # 确保数据在合理范围内
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)

    # 计算MSE
    mse = np.mean((pred - target) ** 2)

    # 计算PSNR
    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)

    # 计算SSIM - 对于3D图像，我们计算每个切片的SSIM然后取平均
    ssim_scores = []
    for slice_idx in range(pred.shape[0]):
        try:
            ssim = structural_similarity(
                pred[slice_idx], target[slice_idx],
                data_range=1.0, win_size=7  # 减小win_size以适应小图像
            )
            ssim_scores.append(ssim)
        except:
            # 如果SSIM计算失败，使用默认值
            ssim_scores.append(0.0)

    ssim_mean = np.mean(ssim_scores) if ssim_scores else 0.0

    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim_mean)
    }


def load_model(model_path, device):
    """
    加载训练好的模型
    """
    checkpoint = torch.load(model_path, map_location=device)

    # 初始化模型
    model = MultiScale_UNet3D(in_channels=1, num_filters_start=32, dropout_rate=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型加载成功: {model_path}")
    print(f"训练时的最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")

    return model


def get_matching_file_pairs(t1w_dir, t2w_dir):
    """
    获取匹配的T1和T2文件对
    """
    t1w_files = [f for f in os.listdir(t1w_dir) if f.endswith('_T1.nii.gz')]
    file_pairs = []

    for t1w_file in t1w_files:
        subject_id = t1w_file.split('_')[0]
        t2w_file = f"{subject_id}_T2.nii.gz"
        t2w_path = os.path.join(t2w_dir, t2w_file)

        if os.path.exists(t2w_path):
            file_pairs.append((t1w_file, t2w_file))
        else:
            print(f"警告: 找不到对应的T2文件 {t2w_file} 对于 {t1w_file}")

    return file_pairs


def process_single_image(model, t1w_path, t2w_path, device, patch_size=(64, 64, 64), overlap=(32, 32, 32),
                         batch_size=8):
    """
    处理单个图像对：预测并计算指标
    """
    # 加载T1w图像（输入）
    t1w_img = nib.load(t1w_path)
    t1w_data = t1w_img.get_fdata().astype(np.float32)
    original_shape = t1w_data.shape

    # 加载T2w图像（真实值）
    t2w_img = nib.load(t2w_path)
    t2w_data = t2w_img.get_fdata().astype(np.float32)

    # 创建图像块
    patches, positions = create_overlapping_patches_3d(t1w_data, patch_size, overlap)
    print(f"  生成 {len(patches)} 个图像块")

    # 分批处理图像块
    predicted_patches = []

    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="预测图像块"):
            batch_patches = patches[i:i + batch_size]

            # 准备batch数据
            batch_tensors = []
            for patch in batch_patches:
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                batch_tensors.append(patch_tensor)

            batch_data = torch.cat(batch_tensors, dim=0).to(device)

            # 模型预测
            batch_pred = model(batch_data)

            # 收集预测结果
            for j in range(batch_pred.shape[0]):
                pred_patch = batch_pred[j, 0].cpu().numpy()  # 移除通道维度
                predicted_patches.append(pred_patch)

    # 重建完整图像
    print("  重建完整图像...")
    reconstructed = reconstruct_from_patches(predicted_patches, positions, original_shape)

    # 计算评价指标
    print("  计算评价指标...")
    metrics = calculate_metrics(reconstructed, t2w_data)

    return reconstructed, metrics, original_shape


def main():
    # 配置参数
    config = {
        'model_path': "D:\\med_data\\MR\\old_data\\model\\1113(0)\\best_model.pth",  # 训练好的模型路径
        'test_t1w_dir': "D:\\med_data\\MR\\TEST1",  # 测试T1加权图像（归一化后）
        'test_t2w_dir': "D:\\med_data\\MR\\TEST2",  # 测试T2加权图像（归一化后）
        'output_dir': "D:\\med_data\\MR\\old_data\\results_multi\\1113",  # 输出目录
        'batch_size': 8,
        'patch_size': (64, 64, 64),
        'overlap': (4, 4, 4),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    print("=" * 60)
    print("3D U-Net MRI图像生成测试程序")
    print("=" * 60)
    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 检查输入
    if not os.path.exists(config['model_path']):
        print(f"错误: 模型文件不存在 {config['model_path']}")
        return

    if not os.path.exists(config['test_t1w_dir']):
        print(f"错误: 测试T1目录不存在 {config['test_t1w_dir']}")
        return

    if not os.path.exists(config['test_t2w_dir']):
        print(f"错误: 测试T2目录不存在 {config['test_t2w_dir']}")
        return

    # 加载模型
    print("\n加载模型中...")
    model = load_model(config['model_path'], config['device'])

    # 获取匹配的文件对
    file_pairs = get_matching_file_pairs(config['test_t1w_dir'], config['test_t2w_dir'])
    print(f"\n找到 {len(file_pairs)} 个测试图像对")

    if len(file_pairs) == 0:
        print("错误: 没有找到匹配的图像对")
        return

    # 存储所有结果
    all_results = []

    # 处理每个图像对
    for t1w_file, t2w_file in file_pairs:
        print(f"\n处理图像对: {t1w_file} -> {t2w_file}")

        t1w_path = os.path.join(config['test_t1w_dir'], t1w_file)
        t2w_path = os.path.join(config['test_t2w_dir'], t2w_file)

        try:
            # 处理单个图像对
            reconstructed, metrics, original_shape = process_single_image(
                model, t1w_path, t2w_path, config['device'],
                config['patch_size'], config['overlap'], config['batch_size']
            )

            # 保存生成的图像
            output_filename = f"generated_{t2w_file}"
            output_path = os.path.join(config['output_dir'], output_filename)

            # 使用原始T2图像的affine和header信息
            original_t2w_img = nib.load(t2w_path)
            generated_img = nib.Nifti1Image(reconstructed, original_t2w_img.affine, original_t2w_img.header)
            nib.save(generated_img, output_path)

            # 记录结果
            result = {
                'case_name': t2w_file.replace('.nii.gz', ''),
                't1w_file': t1w_file,
                't2w_file': t2w_file,
                'generated_file': output_filename,
                'image_shape': f"{original_shape[0]}x{original_shape[1]}x{original_shape[2]}",
                'mse': metrics['mse'],
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            }
            all_results.append(result)

            print(f"  生成图像已保存: {output_filename}")
            print(
                f"  评价指标 - MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")

        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果到Excel
    if all_results:
        print(f"\n保存结果到Excel...")

        # 创建DataFrame
        df = pd.DataFrame(all_results)

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(config['output_dir'], f'test_results_{timestamp}.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 详细结果表
            df.to_excel(writer, sheet_name='详细结果', index=False)

            # 统计摘要表
            summary_data = {
                '指标': ['MSE', 'PSNR (dB)', 'SSIM'],
                '平均值': [
                    df['mse'].mean(),
                    df['psnr'].mean(),
                    df['ssim'].mean()
                ],
                '标准差': [
                    df['mse'].std(),
                    df['psnr'].std(),
                    df['ssim'].std()
                ],
                '最小值': [
                    df['mse'].min(),
                    df['psnr'].min(),
                    df['ssim'].min()
                ],
                '最大值': [
                    df['mse'].max(),
                    df['psnr'].max(),
                    df['ssim'].max()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)

        # 保存为CSV格式
        csv_path = os.path.join(config['output_dir'], f'test_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"结果已保存:")
        print(f"  Excel文件: {excel_path}")
        print(f"  CSV文件: {csv_path}")

        # 打印总体统计
        print(f"\n总体统计:")
        print(f"  平均MSE: {df['mse'].mean():.6f} ± {df['mse'].std():.6f}")
        print(f"  平均PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
        print(f"  平均SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")

    print(f"\n测试完成! 共处理 {len(all_results)} 个图像对")


if __name__ == "__main__":
    main()