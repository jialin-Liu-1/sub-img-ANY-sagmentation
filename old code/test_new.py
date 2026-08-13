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
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# 导入模型
sys.path.append(os.path.dirname(__file__))
# 确保可以导入InterleavedCrossModalUNet3D模型
try:
    from multi.network import InterleavedCrossModalUNet3D
except ImportError:
    print("警告: 无法导入InterleavedCrossModalUNet3D，请确保模型定义可用")


def create_overlapping_patches_3d(image, patch_size=(64, 64, 64), overlap=(16, 16, 16)):
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


def load_model(model_path, device, config=None):
    """
    加载训练好的交织式跨模态3D U-Net模型
    """
    checkpoint = torch.load(model_path, map_location=device)

    # 从检查点获取配置或使用传入的配置
    if config is None and 'config' in checkpoint:
        config = checkpoint['config']
    elif config is None:
        # 默认配置
        config = {
            'data_dim': 8,
            'base_channels': 32,
            'modulation_sparsity': 0.25
        }

    print(f"模型配置: {config}")

    # 初始化模型
    model = InterleavedCrossModalUNet3D(
        image_in_channels=1,
        data_dim=config['data_dim'],
        base_channels=config['base_channels'],
        modulation_sparsity=config['modulation_sparsity'],
        dropout_rate=0.2
    )

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"模型加载成功: {model_path}")

    # 显示训练信息
    if 'best_val_loss' in checkpoint:
        print(f"训练时的最佳验证损失: {checkpoint['best_val_loss']:.6f}")
    if 'final_epoch' in checkpoint:
        print(f"训练轮数: {checkpoint['final_epoch']}")

    return model


def load_features_from_csv(features_csv, selected_features=None):
    """
    从CSV文件加载特征数据
    """
    if selected_features is None:
        selected_features = [
            'mean_intensity', 'std_intensity', 'skewness_norm',
            'kurtosis_norm', 'iqr', 'mad', 'energy_norm', 'contrast'
        ]

    features_df = pd.read_csv(features_csv)

    # 创建病例ID到特征的映射
    case_features = {}
    for idx, row in features_df.iterrows():
        case_id = row['case_id']
        features = row[selected_features].values.astype(np.float32)
        case_features[case_id] = features

    print(f"从 {features_csv} 加载了 {len(case_features)} 个病例的特征数据")
    return case_features


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


def process_single_image(model, t1w_path, features_data, case_id, device,
                         patch_size=(64, 64, 64), overlap=(16, 16, 16),
                         batch_size=4):
    """
    处理单个图像：预测T2图像
    """
    # 加载T1w图像（输入）
    t1w_img = nib.load(t1w_path)
    t1w_data = t1w_img.get_fdata().astype(np.float32)
    original_shape = t1w_data.shape

    # 获取该病例的特征数据
    if case_id not in features_data:
        print(f"警告: 病例 {case_id} 的特征数据不存在，使用零特征")
        features = np.zeros(len(features_data[next(iter(features_data))]), dtype=np.float32)
    else:
        features = features_data[case_id]

    # 创建图像块
    patches, positions = create_overlapping_patches_3d(t1w_data, patch_size, overlap)
    print(f"  生成 {len(patches)} 个图像块")

    # 准备特征张量（为每个块重复）
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # [1, feature_dim]
    features_batch = features_tensor.repeat(batch_size, 1).to(device)

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

            # 准备特征（如果最后一个批次大小不同）
            current_batch_size = batch_data.shape[0]
            if current_batch_size < batch_size:
                current_features = features_tensor.repeat(current_batch_size, 1).to(device)
            else:
                current_features = features_batch

            # 模型预测
            batch_pred = model(batch_data, current_features, modulation_mode='all')

            # 收集预测结果
            for j in range(batch_pred.shape[0]):
                pred_patch = batch_pred[j, 0].cpu().numpy()  # 移除通道维度
                predicted_patches.append(pred_patch)

    # 重建完整图像
    print("  重建完整图像...")
    reconstructed = reconstruct_from_patches(predicted_patches, positions, original_shape)

    return reconstructed, original_shape


def compare_with_ground_truth(generated, t2w_path):
    """
    比较生成图像与真实T2图像
    """
    # 加载T2w图像（真实值）
    t2w_img = nib.load(t2w_path)
    t2w_data = t2w_img.get_fdata().astype(np.float32)

    # 确保尺寸匹配
    if generated.shape != t2w_data.shape:
        print(f"  警告: 生成图像形状 {generated.shape} 与真实图像形状 {t2w_data.shape} 不匹配")
        # 裁剪或填充以匹配尺寸
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(generated.shape, t2w_data.shape))
        generated = generated[:min_shape[0], :min_shape[1], :min_shape[2]]
        t2w_data = t2w_data[:min_shape[0], :min_shape[1], :min_shape[2]]

    # 计算评价指标
    print("  计算评价指标...")
    metrics = calculate_metrics(generated, t2w_data)

    return metrics, t2w_data


def save_visualization_slices(generated, ground_truth, case_id, output_dir):
    """
    保存可视化切片对比图
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 选择几个代表性切片
    slice_indices = [generated.shape[0] // 4, generated.shape[0] // 2, 3 * generated.shape[0] // 4]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Case: {case_id}', fontsize=16)

    for i, slice_idx in enumerate(slice_indices):
        if slice_idx >= generated.shape[0]:
            continue

        # 生成图像切片
        ax1 = axes[0, i]
        im1 = ax1.imshow(generated[slice_idx], cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Generated Slice {slice_idx}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 真实图像切片
        ax2 = axes[1, i]
        im2 = ax2.imshow(ground_truth[slice_idx], cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Ground Truth Slice {slice_idx}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 保存图像
    vis_path = os.path.join(vis_dir, f'{case_id}_comparison.png')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  可视化图像已保存: {vis_path}")


def main():
    # 配置参数
    config = {
        'model_path': "D:\\med_data\\MR\\interleaved_model\\20251216\\final_model_20251216_180854.pth",  # 模型路径
        'test_t1w_dir': "D:\\med_data\\MR\\TEST1",  # 测试T1加权图像
        'test_t2w_dir': "D:\\med_data\\MR\\TEST2",  # 测试T2加权图像（标签）
        'features_csv': "D:\\med_data\\MR\\TEST1\\test4.csv",  # 测试特征数据
        'output_dir': "D:\\med_data\\MR\\result_multi",  # 输出目录
        'batch_size': 4,  # 减小batch_size以适应内存
        'patch_size': (64, 64, 64),
        'overlap': (16, 16, 16),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_visualizations': True,  # 是否保存可视化图像
        'selected_features': [
            'mean_intensity', 'std_intensity', 'skewness_norm',
            'kurtosis_norm', 'iqr', 'mad', 'energy_norm', 'contrast'
        ]
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 创建子目录
    subdirs = ['generated_images', 'evaluation_results']
    if config['save_visualizations']:
        subdirs.append('visualizations')

    for subdir in subdirs:
        os.makedirs(os.path.join(config['output_dir'], subdir), exist_ok=True)

    print("=" * 60)
    print("交织式跨模态3D U-Net MRI图像生成测试程序")
    print("=" * 60)
    print("配置信息:")
    for key, value in config.items():
        if key != 'selected_features':
            print(f"  {key}: {value}")

    # 检查输入
    if not os.path.exists(config['model_path']):
        print(f"错误: 模型文件不存在 {config['model_path']}")
        return

    if not os.path.exists(config['test_t1w_dir']):
        print(f"错误: 测试T1目录不存在 {config['test_t1w_dir']}")
        return

    if not os.path.exists(config['features_csv']):
        print(f"错误: 特征CSV文件不存在 {config['features_csv']}")
        return

    # 加载特征数据
    print("\n加载特征数据...")
    features_data = load_features_from_csv(config['features_csv'], config['selected_features'])

    # 加载模型
    print("\n加载模型中...")
    try:
        model = load_model(config['model_path'], config['device'])
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

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
        print(f"\n{'=' * 40}")
        print(f"处理图像对: {t1w_file} -> {t2w_file}")
        print(f"{'=' * 40}")

        t1w_path = os.path.join(config['test_t1w_dir'], t1w_file)
        t2w_path = os.path.join(config['test_t2w_dir'], t2w_file)

        # 提取病例ID
        case_id = t1w_file.split('_')[0]

        try:
            # 处理单个图像：生成T2图像
            generated_image, original_shape = process_single_image(
                model, t1w_path, features_data, case_id, config['device'],
                config['patch_size'], config['overlap'], config['batch_size']
            )

            # 比较生成图像与真实T2图像
            metrics, ground_truth = compare_with_ground_truth(generated_image, t2w_path)

            # 保存生成的图像
            output_filename = f"generated_{t2w_file}"
            output_path = os.path.join(config['output_dir'], 'generated_images', output_filename)

            # 使用原始T2图像的affine和header信息
            original_t2w_img = nib.load(t2w_path)
            generated_img = nib.Nifti1Image(generated_image, original_t2w_img.affine, original_t2w_img.header)
            nib.save(generated_img, output_path)

            # 保存可视化对比图
            if config['save_visualizations']:
                save_visualization_slices(generated_image, ground_truth, case_id, config['output_dir'])

            # 记录结果
            result = {
                'case_id': case_id,
                't1w_file': t1w_file,
                't2w_file': t2w_file,
                'generated_file': output_filename,
                'image_shape': f"{original_shape[0]}x{original_shape[1]}x{original_shape[2]}",
                'mse': metrics['mse'],
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'num_features_used': len(config['selected_features'])
            }
            all_results.append(result)

            print(f"  生成图像已保存: {output_filename}")
            print(f"  评价指标:")
            print(f"    MSE: {metrics['mse']:.6f}")
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
            print(f"    SSIM: {metrics['ssim']:.4f}")

        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果到Excel和CSV
    if all_results:
        print(f"\n{'=' * 60}")
        print(f"保存结果到文件...")

        # 创建DataFrame
        df = pd.DataFrame(all_results)

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(config['output_dir'], 'evaluation_results', f'test_results_{timestamp}.xlsx')
        csv_path = os.path.join(config['output_dir'], 'evaluation_results', f'test_results_{timestamp}.csv')

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
                ],
                '中位数': [
                    df['mse'].median(),
                    df['psnr'].median(),
                    df['ssim'].median()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)

            # 病例排名表
            df_sorted = df.sort_values('psnr', ascending=False)
            df_sorted['rank'] = range(1, len(df_sorted) + 1)
            df_sorted.to_excel(writer, sheet_name='病例排名', index=False)

        # 保存为CSV格式
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存为JSON格式（便于后续分析）
        json_path = os.path.join(config['output_dir'], 'evaluation_results', f'test_results_{timestamp}.json')
        results_dict = {
            'timestamp': timestamp,
            'config': config,
            'results': df.to_dict('records'),
            'summary': {
                'mean_mse': float(df['mse'].mean()),
                'std_mse': float(df['mse'].std()),
                'mean_psnr': float(df['psnr'].mean()),
                'std_psnr': float(df['psnr'].std()),
                'mean_ssim': float(df['ssim'].mean()),
                'std_ssim': float(df['ssim'].std())
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"结果已保存:")
        print(f"  Excel文件: {excel_path}")
        print(f"  CSV文件: {csv_path}")
        print(f"  JSON文件: {json_path}")

        # 打印总体统计
        print(f"\n{'=' * 60}")
        print(f"总体统计结果:")
        print(f"{'=' * 60}")
        print(f"  处理病例数: {len(df)}")
        print(f"  平均MSE: {df['mse'].mean():.6f} ± {df['mse'].std():.6f}")
        print(f"  平均PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
        print(f"  平均SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")

        # 最佳和最差病例
        best_case = df.loc[df['psnr'].idxmax()]
        worst_case = df.loc[df['psnr'].idxmin()]

        print(f"\n  最佳病例: {best_case['case_id']} (PSNR: {best_case['psnr']:.2f} dB)")
        print(f"  最差病例: {worst_case['case_id']} (PSNR: {worst_case['psnr']:.2f} dB)")

    print(f"\n{'=' * 60}")
    print(f"测试完成! 共处理 {len(all_results)} 个图像对")
    print(f"生成图像保存在: {os.path.join(config['output_dir'], 'generated_images')}")
    print(f"评估结果保存在: {os.path.join(config['output_dir'], 'evaluation_results')}")

    if config['save_visualizations']:
        print(f"可视化图像保存在: {os.path.join(config['output_dir'], 'visualizations')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()