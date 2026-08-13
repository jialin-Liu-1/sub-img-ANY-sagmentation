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
import warnings

warnings.filterwarnings('ignore')

# 导入模型
sys.path.append(os.path.dirname(__file__))
try:
    from multi.network import InterleavedCrossModalUNet3D
except ImportError:
    print("警告: 无法导入InterleavedCrossModalUNet3D，请确保模型定义可用")


class DataFeatureAnalyzer:
    """数据特征重要性分析器，使用LIME方法"""

    def __init__(self, model, selected_features, device='cuda'):
        self.model = model
        self.selected_features = selected_features
        self.device = device
        self.model.eval()
        print(f"特征分析器初始化完成，设备: {device}")

    def predict_with_perturbed_features(self, image_patch, features, feature_idx, perturbation_scale=0.3):
        """
        使用扰动后的特征进行预测
        """
        batch_size = 16  # 进一步减小批次大小以避免内存问题

        # 创建扰动特征
        perturbed_features_list = []
        for i in range(batch_size):
            # 创建扰动：在原始特征的基础上添加高斯噪声
            perturbed = features.clone()
            # 对指定特征添加随机扰动
            if feature_idx is not None:
                # 确保是标量值
                noise = torch.randn(1, device=self.device).item() * perturbation_scale
                perturbed[feature_idx] += noise
            else:
                # 对所有特征添加随机扰动
                perturbed += torch.randn_like(perturbed) * perturbation_scale * 0.1

            # 确保特征在合理范围内
            perturbed = torch.clamp(perturbed, 0, 1)
            perturbed_features_list.append(perturbed)

        # 堆叠成批次
        perturbed_features = torch.stack(perturbed_features_list)

        # 重复图像批次 - 正确的维度处理
        # image_patch形状应该是[1, D, H, W]（通道维度为1）
        # 需要扩展到[batch_size, 1, D, H, W]
        if len(image_patch.shape) == 3:  # [D, H, W]
            image_patch = image_patch.unsqueeze(0)  # [1, D, H, W]

        # 现在image_patch是[1, D, H, W]或[1, 1, D, H, W]
        if len(image_patch.shape) == 4:  # [1, D, H, W]
            image_patch = image_patch.unsqueeze(0)  # [1, 1, D, H, W]

        # 扩展到批次大小
        image_batch = image_patch.repeat(batch_size, 1, 1, 1, 1)

        with torch.no_grad():
            predictions = self.model(image_batch, perturbed_features, modulation_mode='all')

        return predictions.mean(dim=0, keepdim=True)

    def compute_feature_importance(self, image_patch, features, num_samples=200):
        """
        计算每个特征的重要性分数
        """
        print(f"  计算特征重要性，样本数: {num_samples}")

        # 确保image_patch有正确的维度
        if len(image_patch.shape) == 3:  # [D, H, W]
            image_patch = image_patch.unsqueeze(0)  # [1, D, H, W]

        # 确保图像在设备上
        image_patch = image_patch.to(self.device)
        features = features.to(self.device)

        # 添加批次维度
        image_input = image_patch.unsqueeze(0)  # [1, 1, D, H, W] 或 [1, D, H, W]
        features_input = features.unsqueeze(0)  # [1, feature_dim]

        # 获取基准预测（使用原始特征）
        with torch.no_grad():
            baseline_pred = self.model(
                image_input,
                features_input,
                modulation_mode='all'
            )

        importance_scores = []

        print(f"分析 {len(self.selected_features)} 个特征的重要性...")
        pbar = tqdm(range(len(self.selected_features)), desc="特征重要性分析")

        for i in pbar:
            # 计算当该特征被扰动时的预测
            perturbed_preds = []

            # 分批处理，减少内存使用
            num_batches = max(1, num_samples // 10)  # 更小的批次
            for batch_idx in range(num_batches):
                try:
                    perturbed_pred = self.predict_with_perturbed_features(
                        image_patch, features, i, perturbation_scale=0.2  # 减小扰动程度
                    )
                    perturbed_preds.append(perturbed_pred)
                except Exception as e:
                    print(f"    批次 {batch_idx} 失败: {e}")
                    continue

            # 计算平均扰动预测
            if perturbed_preds:
                avg_perturbed_pred = torch.cat(perturbed_preds, dim=0).mean(dim=0, keepdim=True)
            else:
                print(f"    特征 {self.selected_features[i]} 无有效预测，使用基准预测")
                avg_perturbed_pred = baseline_pred.clone()

            # 计算该特征的重要性（预测变化）
            importance = torch.abs(baseline_pred - avg_perturbed_pred).mean().item()
            importance_scores.append(importance)

            pbar.set_postfix({
                '特征': self.selected_features[i],
                '重要性': f'{importance:.6f}'
            })

        return importance_scores

    def compute_global_importance(self, image_patch, features, num_samples=100):
        """
        计算全局特征重要性（所有特征同时扰动）
        """
        # 确保image_patch有正确的维度
        if len(image_patch.shape) == 3:  # [D, H, W]
            image_patch = image_patch.unsqueeze(0)  # [1, D, H, W]

        # 确保数据在设备上
        image_patch = image_patch.to(self.device)
        features = features.to(self.device)

        # 添加批次维度
        image_input = image_patch.unsqueeze(0)  # [1, 1, D, H, W] 或 [1, D, H, W]
        features_input = features.unsqueeze(0)  # [1, feature_dim]

        with torch.no_grad():
            baseline_pred = self.model(
                image_input,
                features_input,
                modulation_mode='all'
            )

        # 全局扰动
        perturbed_preds = []
        for _ in range(num_samples // 5):
            try:
                perturbed_pred = self.predict_with_perturbed_features(
                    image_patch, features, None, perturbation_scale=0.3
                )
                perturbed_preds.append(perturbed_pred)
            except Exception as e:
                print(f"    全局扰动失败: {e}")
                continue

        if perturbed_preds:
            avg_perturbed_pred = torch.cat(perturbed_preds, dim=0).mean(dim=0, keepdim=True)
            global_importance = torch.abs(baseline_pred - avg_perturbed_pred).mean().item()
        else:
            global_importance = 0.0

        return global_importance

    def analyze_single_case(self, image_patch, features, case_id, output_dir):
        """
        分析单个病例的特征重要性
        """
        print(f"  开始分析病例 {case_id}...")

        # 确保数据在设备上
        image_patch = image_patch.to(self.device)
        features = features.to(self.device)

        print(f"  图像块形状: {image_patch.shape}")
        print(f"  特征形状: {features.shape}")

        # 计算特征重要性
        importance_scores = self.compute_feature_importance(image_patch, features)

        if not importance_scores:
            print(f"  警告: 无法计算特征重要性，跳过病例 {case_id}")
            return None

        # 计算全局重要性
        global_importance = self.compute_global_importance(image_patch, features)

        # 可视化结果
        self.visualize_feature_importance(
            importance_scores,
            global_importance,
            case_id,
            output_dir
        )

        # 保存分析结果
        self.save_analysis_results(
            importance_scores,
            global_importance,
            case_id,
            output_dir
        )

        return importance_scores

    def visualize_feature_importance(self, importance_scores, global_importance, case_id, output_dir):
        """
        可视化特征重要性
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'病例 {case_id} 特征重要性分析', fontsize=16)

        # 1. 条形图
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.selected_features)))
        bars = ax1.bar(range(len(self.selected_features)), importance_scores, color=colors)
        ax1.set_xlabel('特征')
        ax1.set_ylabel('重要性分数')
        ax1.set_title(f'特征重要性分数 (全局重要性: {global_importance:.6f})')
        ax1.set_xticks(range(len(self.selected_features)))
        ax1.set_xticklabels(self.selected_features, rotation=45, ha='right')

        # 在条形上添加数值
        for bar, score in zip(bars, importance_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{score:.6f}', ha='center', va='bottom', fontsize=8)

        # 2. 饼图（归一化后的重要性）
        ax2 = axes[0, 1]
        total_score = sum(importance_scores)
        if total_score > 0:
            normalized_scores = np.array(importance_scores) / total_score
            explode = [0.1 if score == max(normalized_scores) else 0 for score in normalized_scores]
            wedges, texts, autotexts = ax2.pie(normalized_scores, labels=self.selected_features,
                                               autopct='%1.1f%%', startangle=90, explode=explode)
            ax2.set_title('特征重要性分布')
        else:
            ax2.text(0.5, 0.5, '无重要性分数', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('特征重要性分布')

        # 3. 热力图
        ax3 = axes[1, 0]
        if len(importance_scores) > 0:
            im = ax3.imshow(np.array(importance_scores).reshape(1, -1), cmap='YlOrRd', aspect='auto')
            ax3.set_title('特征重要性热力图')
            ax3.set_yticks([])
            ax3.set_xticks(range(len(self.selected_features)))
            ax3.set_xticklabels(self.selected_features, rotation=45, ha='right')
            plt.colorbar(im, ax=ax3)
        else:
            ax3.text(0.5, 0.5, '无重要性分数', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('特征重要性热力图')

        # 4. 特征重要性随扰动程度的变化
        ax4 = axes[1, 1]
        # 选择最重要的3个特征
        if len(importance_scores) >= 3:
            top_indices = np.argsort(importance_scores)[-3:][::-1]
            top_features = [self.selected_features[i] for i in top_indices]
            top_scores = [importance_scores[i] for i in top_indices]

            ax4.bar(range(3), top_scores, color=['red', 'orange', 'yellow'])
            ax4.set_xlabel('特征')
            ax4.set_ylabel('重要性分数')
            ax4.set_title('Top 3 最重要特征')
            ax4.set_xticks(range(3))
            ax4.set_xticklabels(top_features, rotation=45, ha='right')

            # 添加数值标签
            for i, (feature, score) in enumerate(zip(top_features, top_scores)):
                ax4.text(i, score + 0.001, f'{score:.6f}', ha='center', va='bottom', fontsize=10)
        else:
            ax4.text(0.5, 0.5, '无足够特征进行分析',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('重要特征分析')

        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(output_dir, f'{case_id}_feature_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  特征重要性图已保存: {output_path}")

    def save_analysis_results(self, importance_scores, global_importance, case_id, output_dir):
        """
        保存分析结果
        """
        # 创建结果字典
        results = {
            'case_id': case_id,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'global_importance': float(global_importance),
            'feature_importance': {},
            'feature_descriptions': {
                'mean_intensity': '平均强度',
                'std_intensity': '强度标准差',
                'skewness_norm': '归一化偏度',
                'kurtosis_norm': '归一化峰度',
                'iqr': '四分位距',
                'mad': '绝对中位差',
                'energy_norm': '归一化能量',
                'contrast': '对比度'
            }
        }

        total_score = sum(importance_scores) if importance_scores else 1e-10

        for i, (feature, score) in enumerate(zip(self.selected_features, importance_scores)):
            results['feature_importance'][feature] = {
                'score': float(score),
                'normalized_score': float(score / total_score),
                'description': results['feature_descriptions'].get(feature, '未知')
            }

        # 按分数排序
        sorted_features = sorted(zip(self.selected_features, importance_scores),
                                 key=lambda x: x[1], reverse=True)

        results['ranking'] = [
            {
                'feature': feat,
                'score': float(score),
                'normalized_score': float(score / total_score),
                'description': results['feature_descriptions'].get(feat, '未知')
            }
            for feat, score in sorted_features
        ]

        # 保存为JSON
        json_path = os.path.join(output_dir, f'{case_id}_feature_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存为CSV
        csv_path = os.path.join(output_dir, f'{case_id}_feature_analysis.csv')
        df = pd.DataFrame({
            'feature': self.selected_features,
            'importance_score': importance_scores,
            'normalized_score': np.array(importance_scores) / total_score,
            'description': [results['feature_descriptions'].get(f, '未知') for f in self.selected_features]
        })
        df = df.sort_values('importance_score', ascending=False)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"  特征分析结果已保存: {json_path}")
        print(f"  特征分析CSV已保存: {csv_path}")

        # 打印总结
        print(f"\n  {case_id} 特征重要性排名:")
        for i, (feature, score) in enumerate(sorted_features[:5]):
            desc = results['feature_descriptions'].get(feature, '未知')
            print(f"    第{i + 1}名: {feature} ({desc}) - 分数: {score:.6f}")


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


def analyze_model_feature_sensitivity(config):
    """
    分析模型对数据特征的敏感性
    """
    print("\n" + "=" * 60)
    print("模型特征敏感性分析")
    print("=" * 60)

    # 创建输出目录
    analysis_dir = os.path.join(config['output_dir'], 'feature_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # 加载特征数据
    features_data = load_features_from_csv(config['features_csv'], config['selected_features'])

    # 加载模型
    model = load_model(config['model_path'], config['device'])

    # 初始化分析器
    analyzer = DataFeatureAnalyzer(
        model,
        config['selected_features'],
        device=config['device']
    )

    # 获取测试文件
    file_pairs = get_matching_file_pairs(config['test_t1w_dir'], config['test_t2w_dir'])

    # 分析每个病例
    all_results = []

    for t1w_file, _ in file_pairs[:3]:  # 只分析前3个病例作为示例
        print(f"\n分析病例: {t1w_file}")

        # 提取病例ID
        case_id = t1w_file.split('_')[0]

        # 获取特征
        if case_id not in features_data:
            print(f"  警告: 病例 {case_id} 的特征数据不存在，跳过")
            continue

        features = features_data[case_id]

        # 加载图像并提取中心块
        t1w_path = os.path.join(config['test_t1w_dir'], t1w_file)
        t1w_img = nib.load(t1w_path)
        t1w_data = t1w_img.get_fdata().astype(np.float32)

        # 提取一个代表性的图像块（中心区域）
        d, h, w = t1w_data.shape
        patch_size = config['patch_size']

        # 确保图像足够大
        if d < patch_size[0] or h < patch_size[1] or w < patch_size[2]:
            print(f"  警告: 图像尺寸 {t1w_data.shape} 小于块尺寸 {patch_size}，跳过")
            continue

        # 提取中心块
        d_start = (d - patch_size[0]) // 2
        h_start = (h - patch_size[1]) // 2
        w_start = (w - patch_size[2]) // 2

        image_patch = t1w_data[
                      d_start:d_start + patch_size[0],
                      h_start:h_start + patch_size[1],
                      w_start:w_start + patch_size[2]
                      ]

        # 标准化图像块到[0, 1]范围
        image_min = image_patch.min()
        image_max = image_patch.max()
        if image_max > image_min:
            image_patch = (image_patch - image_min) / (image_max - image_min)
        else:
            image_patch = np.zeros_like(image_patch)

        print(f"  图像块形状: {image_patch.shape}")
        print(f"  特征向量: {features}")

        # 转换为tensor - 正确的维度: [channels, depth, height, width]
        # 注意：我们假设通道维度为1
        image_tensor = torch.from_numpy(image_patch).float().unsqueeze(0)  # [D, H, W] -> [1, D, H, W]

        features_tensor = torch.from_numpy(features).float()

        # 分析特征重要性
        try:
            importance_scores = analyzer.analyze_single_case(
                image_tensor,
                features_tensor,
                case_id,
                analysis_dir
            )

            if importance_scores is not None:
                # 保存结果
                all_results.append({
                    'case_id': case_id,
                    'importance_scores': importance_scores,
                    'top_feature': config['selected_features'][
                        np.argmax(importance_scores)] if importance_scores else None
                })

        except Exception as e:
            print(f"  分析失败: {e}")
            import traceback
            traceback.print_exc()

    # 综合分析所有病例
    if all_results:
        analyze_cross_case_features(all_results, config['selected_features'], analysis_dir)

    print(f"\n特征敏感性分析完成!")
    print(f"结果保存在: {analysis_dir}")


def analyze_cross_case_features(all_results, selected_features, output_dir):
    """
    综合分析所有病例的特征重要性
    """
    print(f"\n综合分析 {len(all_results)} 个病例的特征重要性...")

    if not all_results:
        print("  无有效分析结果")
        return

    # 计算平均重要性分数
    num_features = len(selected_features)
    avg_importance = np.zeros(num_features)
    num_cases = len(all_results)

    for result in all_results:
        if result['importance_scores'] is not None:
            avg_importance += np.array(result['importance_scores'])

    avg_importance /= num_cases

    # 创建综合可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('跨病例特征重要性综合分析', fontsize=16)

    # 1. 平均重要性条形图
    ax1 = axes[0, 0]
    sorted_indices = np.argsort(avg_importance)[::-1]
    sorted_features = [selected_features[i] for i in sorted_indices]
    sorted_scores = [avg_importance[i] for i in sorted_indices]

    bars = ax1.bar(range(num_features), sorted_scores,
                   color=plt.cm.viridis(np.linspace(0, 1, num_features)))
    ax1.set_xlabel('特征')
    ax1.set_ylabel('平均重要性分数')
    ax1.set_title(f'特征平均重要性 (共 {num_cases} 个病例)')
    ax1.set_xticks(range(num_features))
    ax1.set_xticklabels(sorted_features, rotation=45, ha='right')

    # 添加误差条（标准差）
    if num_cases > 1:
        std_values = []
        for i in range(num_features):
            feature_scores = [r['importance_scores'][i] for r in all_results if r['importance_scores'] is not None]
            std_values.append(np.std(feature_scores) if feature_scores else 0)

        ax1.errorbar(range(num_features), sorted_scores, yerr=std_values,
                     fmt='none', ecolor='black', capsize=5)

    # 2. 重要性热力图（按病例）
    ax2 = axes[0, 1]
    if len(all_results) > 1:
        importance_matrix = np.array(
            [r['importance_scores'] for r in all_results if r['importance_scores'] is not None])

        # 按平均重要性排序特征
        if importance_matrix.shape[0] > 0:
            importance_matrix_sorted = importance_matrix[:, sorted_indices]

            im = ax2.imshow(importance_matrix_sorted, cmap='YlOrRd', aspect='auto')
            ax2.set_title('各病例特征重要性热力图')
            ax2.set_xlabel('特征')
            ax2.set_ylabel('病例')
            ax2.set_xticks(range(num_features))
            ax2.set_xticklabels(sorted_features, rotation=45, ha='right')
            ax2.set_yticks(range(len(all_results)))
            ax2.set_yticklabels([r['case_id'] for r in all_results])
            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('各病例特征重要性热力图')
    else:
        ax2.text(0.5, 0.5, '需要至少2个病例', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('各病例特征重要性热力图')

    # 3. 箱线图
    ax3 = axes[1, 0]
    if len(all_results) > 1:
        data_to_plot = []
        for i in range(num_features):
            feature_scores = [r['importance_scores'][i] for r in all_results if r['importance_scores'] is not None]
            if feature_scores:
                data_to_plot.append(feature_scores)

        if data_to_plot:
            bp = ax3.boxplot(data_to_plot, labels=selected_features[:len(data_to_plot)])
            ax3.set_title('特征重要性分布箱线图')
            ax3.set_xlabel('特征')
            ax3.set_ylabel('重要性分数')
            ax3.set_xticklabels(selected_features[:len(data_to_plot)], rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('特征重要性分布箱线图')
    else:
        ax3.text(0.5, 0.5, '需要至少2个病例', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('特征重要性分布箱线图')

    # 4. 重要特征稳定性分析
    ax4 = axes[1, 1]
    if len(all_results) >= 2:
        # 计算每个特征的稳定性（在不同病例间的一致性）
        stability_scores = []
        for i in range(num_features):
            feature_scores = [r['importance_scores'][i] for r in all_results if r['importance_scores'] is not None]
            if feature_scores:
                stability = 1 - (np.std(feature_scores) / (np.mean(feature_scores) + 1e-10))
                stability_scores.append(stability)
            else:
                stability_scores.append(0)

        # 选择最稳定的特征
        stable_indices = np.argsort(stability_scores)[-5:][::-1]
        stable_features = [selected_features[i] for i in stable_indices]
        stable_scores = [stability_scores[i] for i in stable_indices]

        bars_stable = ax4.bar(range(len(stable_features)), stable_scores,
                              color=plt.cm.PuRd(np.linspace(0.3, 0.9, len(stable_features))))
        ax4.set_xlabel('特征')
        ax4.set_ylabel('稳定性分数')
        ax4.set_title('Top 5 最稳定特征')
        ax4.set_xticks(range(len(stable_features)))
        ax4.set_xticklabels(stable_features, rotation=45, ha='right')

        # 添加数值标签
        for i, (feature, score) in enumerate(zip(stable_features, stable_scores)):
            ax4.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax4.text(0.5, 0.5, '需要至少2个病例进行稳定性分析',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('特征稳定性分析')

    plt.tight_layout()

    # 保存综合图表
    output_path = os.path.join(output_dir, 'cross_case_feature_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存综合分析结果
    cross_results = {
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_cases': num_cases,
        'feature_ranking': []
    }

    for i, idx in enumerate(sorted_indices):
        feature_scores = [r['importance_scores'][idx] for r in all_results if r['importance_scores'] is not None]
        if feature_scores:
            std_value = np.std(feature_scores)
        else:
            std_value = 0

        cross_results['feature_ranking'].append({
            'feature': selected_features[idx],
            'avg_importance': float(avg_importance[idx]),
            'std_importance': float(std_value),
            'rank': i + 1
        })

    # 保存为JSON
    json_path = os.path.join(output_dir, 'cross_case_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cross_results, f, indent=2, ensure_ascii=False)

    # 保存为CSV
    csv_path = os.path.join(output_dir, 'cross_case_analysis.csv')
    df_cross = pd.DataFrame(cross_results['feature_ranking'])
    df_cross.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n综合分析完成!")
    print(f"  综合图表: {output_path}")
    print(f"  综合结果JSON: {json_path}")
    print(f"  综合结果CSV: {csv_path}")

    # 打印总结
    print(f"\n  特征重要性总结:")
    for i, item in enumerate(cross_results['feature_ranking'][:5]):
        print(
            f"    第{i + 1}名: {item['feature']} - 平均分数: {item['avg_importance']:.6f} ± {item['std_importance']:.6f}")


def main():
    """
    主函数：进行特征敏感性分析
    """
    # 配置参数
    config = {
        'model_path': "D:\\med_data\\MR\\interleaved_model\\20251216\\checkpoints\\best_model.pth",  # 模型路径
        'test_t1w_dir': "D:\\med_data\\MR\\TEST1",  # 测试T1加权图像
        'test_t2w_dir': "D:\\med_data\\MR\\TEST2",  # 测试T2加权图像（标签）
        'features_csv': "D:\\med_data\\MR\\TEST1\\test4.csv",  # 测试特征数据
        'output_dir': "D:\\med_data\\MR\\result_multi\\LIME",  # 输出目录
        'patch_size': (64, 64, 64),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'selected_features': [
            'mean_intensity', 'std_intensity', 'skewness_norm',
            'kurtosis_norm', 'iqr', 'mad', 'energy_norm', 'contrast'
        ]
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    print("=" * 60)
    print("交织式跨模态3D U-Net 特征敏感性分析")
    print("使用LIME方法分析模型对输入数据的关注程度")
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

    # 进行特征敏感性分析
    analyze_model_feature_sensitivity(config)


if __name__ == "__main__":
    main()