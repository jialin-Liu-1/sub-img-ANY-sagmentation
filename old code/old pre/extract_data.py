import os
import numpy as np
import nibabel as nib
from scipy import stats
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class MRIFeatureExtractor:
    """MRI图像特征提取器"""

    def __init__(self, image_dir, output_file=None):
        """
        初始化特征提取器

        参数:
            image_dir: 图像目录路径
            output_file: 输出文件路径（可选）
        """
        self.image_dir = Path(image_dir)

        if output_file is None:
            # 默认输出到同一目录
            self.output_file = self.image_dir / "mri_statistical_features.csv"
        else:
            self.output_file = Path(output_file)

        # 特征列表
        self.feature_names = [
            'mean_intensity',  # 平均强度
            'std_intensity',  # 强度标准差
            'skewness',  # 偏度
            'kurtosis',  # 峰度
            'median_intensity',  # 强度中位数
            'iqr',  # 四分位距
            'entropy',  # 熵
            'energy',  # 能量
            'contrast',  # 对比度
            'uniformity',  # 均匀度
            'min_intensity',  # 最小强度（新增）
            'max_intensity',  # 最大强度（新增）
            'range_intensity',  # 强度范围（新增）
            'variance',  # 方差（新增）
            'coeff_variation',  # 变异系数（新增）
            'mad',  # 平均绝对偏差（新增）
            'volume_voxels',  # 脑组织体素数（新增）
            'volume_mm3',  # 脑组织体积mm³（新增）
        ]

    def extract_brain_data(self, image_data, threshold_factor=0.1):
        """
        提取脑组织数据（去除背景）

        参数:
            image_data: 3D图像数据
            threshold_factor: 阈值因子（mean * factor）

        返回:
            brain_data: 脑组织数据（1D数组）
            mask: 脑组织掩码
        """
        # 计算阈值
        threshold = np.mean(image_data) * threshold_factor

        # 创建脑组织掩码
        mask = image_data > threshold

        # 提取脑组织数据
        brain_data = image_data[mask]

        return brain_data, mask

    def calculate_features(self, brain_data, mask, voxel_volume_mm3=1.0):
        """
        计算统计特征

        参数:
            brain_data: 脑组织数据
            mask: 脑组织掩码
            voxel_volume_mm3: 单个体素的体积（mm³）

        返回:
            features: 特征字典
        """
        if len(brain_data) == 0:
            # 如果没有脑组织数据，返回NaN
            return {feature: np.nan for feature in self.feature_names}

        # 基本统计特征
        features = {
            'mean_intensity': float(np.mean(brain_data)),
            'std_intensity': float(np.std(brain_data)),
            'skewness': float(stats.skew(brain_data)),
            'kurtosis': float(stats.kurtosis(brain_data)),
            'median_intensity': float(np.median(brain_data)),
            'min_intensity': float(np.min(brain_data)),
            'max_intensity': float(np.max(brain_data)),
            'range_intensity': float(np.max(brain_data) - np.min(brain_data)),
            'variance': float(np.var(brain_data)),
        }

        # 四分位距
        q75, q25 = np.percentile(brain_data, [75, 25])
        features['iqr'] = float(q75 - q25)

        # 平均绝对偏差
        features['mad'] = float(np.mean(np.abs(brain_data - features['mean_intensity'])))

        # 变异系数（标准差/均值）
        if features['mean_intensity'] != 0:
            features['coeff_variation'] = float(features['std_intensity'] / features['mean_intensity'])
        else:
            features['coeff_variation'] = np.nan

        # 熵
        hist, _ = np.histogram(brain_data, bins=50)
        hist = hist.astype(float)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist_normalized = hist / hist_sum
            # 避免log(0)
            hist_normalized = hist_normalized[hist_normalized > 0]
            entropy_val = -np.sum(hist_normalized * np.log2(hist_normalized))
        else:
            entropy_val = 0
        features['entropy'] = float(entropy_val)

        # 能量
        features['energy'] = float(np.sum(brain_data ** 2))

        # 对比度（已经计算为range_intensity）
        features['contrast'] = features['range_intensity']

        # 均匀度
        if hist_sum > 0:
            features['uniformity'] = float(np.sum((hist / hist_sum) ** 2))
        else:
            features['uniformity'] = np.nan

        # 体积特征
        features['volume_voxels'] = int(np.sum(mask))
        features['volume_mm3'] = float(features['volume_voxels'] * voxel_volume_mm3)

        return features

    def process_single_image(self, image_path, case_id):
        """
        处理单个图像

        参数:
            image_path: 图像文件路径
            case_id: 病例ID

        返回:
            features: 特征字典，包含case_id
        """
        print(f"处理: {case_id}")

        try:
            # 加载图像
            img = nib.load(str(image_path))
            data = img.get_fdata()

            # 获取体素体积（mm³）
            affine = img.affine
            voxel_dims = np.abs(np.diag(affine[:3, :3]))
            voxel_volume_mm3 = np.prod(voxel_dims)

            # 提取脑组织
            brain_data, mask = self.extract_brain_data(data)

            # 计算特征
            features = self.calculate_features(brain_data, mask, voxel_volume_mm3)

            # 添加病例ID
            features['case_id'] = case_id

            print(f"  ✓ 完成: {len(brain_data)}个体素")
            return features

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            # 返回包含NaN的特征
            features = {feature: np.nan for feature in self.feature_names}
            features['case_id'] = case_id
            return features

    def process_all_images(self, batch_size=1, save_frequency=10):
        """
        处理所有图像

        参数:
            batch_size: 批处理大小（此处固定为1）
            save_frequency: 保存频率（每处理N个病例保存一次）

        返回:
            all_features: 所有特征列表
        """
        print("=" * 60)
        print("开始提取MRI图像特征")
        print("=" * 60)

        # 查找所有nii.gz文件
        image_files = list(self.image_dir.glob("*.nii.gz"))

        if not image_files:
            print(f"在目录中未找到.nii.gz文件: {self.image_dir}")
            return []

        print(f"找到 {len(image_files)} 个图像文件")

        # 检查是否已有部分结果
        existing_data = self.load_existing_features()
        existing_cases = set(existing_data['case_id'].tolist()) if not existing_data.empty else set()

        all_features = []
        processed_count = 0
        new_cases_count = 0

        # 逐个处理图像
        for i, image_file in enumerate(image_files):
            # 提取病例ID
            case_id = self.extract_case_id(image_file)

            # 检查是否已处理
            if case_id in existing_cases:
                print(f"跳过已处理: {case_id}")
                continue

            # 处理单个图像
            features = self.process_single_image(image_file, case_id)
            all_features.append(features)
            new_cases_count += 1
            processed_count += 1

            # 定期保存
            if new_cases_count % save_frequency == 0:
                self.save_features_intermediate(all_features, existing_data, i + 1, len(image_files))

        # 最终保存
        if all_features:
            self.save_features_final(all_features, existing_data)

        return all_features

    def extract_case_id(self, image_path):
        """
        从文件名提取病例ID

        参数:
            image_path: 图像文件路径

        返回:
            case_id: 病例ID
        """
        filename = image_path.stem.replace('.nii', '')

        # 尝试提取s001、s002等格式
        import re

        # 匹配 s001, s002, s003 等
        match = re.search(r'[sS](\d{3})', filename)
        if match:
            return f"s{match.group(1)}"

        # 匹配 sub-s001, sub-s002 等
        match = re.search(r'sub-[sS](\d{3})', filename)
        if match:
            return f"s{match.group(1)}"

        # 匹配其他常见格式
        patterns = [
            r'(\d{3})_T1',  # 001_T1
            r'(\d{3})-T1',  # 001-T1
            r'case_(\d+)',  # case_001
            r'patient_(\d+)',  # patient_001
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                num = match.group(1).zfill(3)  # 填充为3位
                return f"s{num}"

        # 如果都没匹配到，使用文件名
        return filename

    def load_existing_features(self):
        """
        加载已存在的特征文件

        返回:
            existing_df: 已存在的DataFrame，如果没有则返回空DataFrame
        """
        if self.output_file.exists():
            try:
                existing_df = pd.read_csv(self.output_file)
                print(f"加载已存在的特征文件: {self.output_file}")
                print(f"已包含 {len(existing_df)} 个病例")
                return existing_df
            except Exception as e:
                print(f"加载现有文件失败: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def save_features_intermediate(self, new_features, existing_data, current, total):
        """
        中间保存特征

        参数:
            new_features: 新特征列表
            existing_data: 已存在的数据
            current: 当前处理的序号
            total: 总文件数
        """
        try:
            # 合并数据
            new_df = pd.DataFrame(new_features)

            if not existing_data.empty:
                # 移除重复的病例
                existing_data = existing_data[~existing_data['case_id'].isin(new_df['case_id'])]
                combined_df = pd.concat([existing_data, new_df], ignore_index=True)
            else:
                combined_df = new_df

            # 保存
            combined_df.to_csv(self.output_file, index=False)

            print(f"\n[进度: {current}/{total}] 已保存中间结果")
            print(f"文件: {self.output_file}")
            print(f"总病例数: {len(combined_df)}")

        except Exception as e:
            print(f"中间保存失败: {e}")

    def save_features_final(self, new_features, existing_data):
        """
        最终保存特征

        参数:
            new_features: 新特征列表
            existing_data: 已存在的数据
        """
        try:
            # 创建新数据的DataFrame
            new_df = pd.DataFrame(new_features)

            # 设置列顺序：case_id在第一列，然后是特征
            columns_order = ['case_id'] + [col for col in new_df.columns if col != 'case_id']
            new_df = new_df[columns_order]

            if not existing_data.empty:
                # 合并数据，新数据覆盖旧数据
                combined_df = pd.concat([existing_data, new_df], ignore_index=True)
                # 去除重复，保留最后出现的（新数据）
                combined_df = combined_df.drop_duplicates(subset=['case_id'], keep='last')
            else:
                combined_df = new_df

            # 按case_id排序
            combined_df['case_num'] = combined_df['case_id'].str.extract(r'(\d+)').astype(int)
            combined_df = combined_df.sort_values('case_num').drop('case_num', axis=1)

            # 保存到CSV
            combined_df.to_csv(self.output_file, index=False)

            print("\n" + "=" * 60)
            print("特征提取完成!")
            print(f"输出文件: {self.output_file}")
            print(f"总病例数: {len(combined_df)}")
            print(f"特征数量: {len(combined_df.columns) - 1}")  # 减去case_id列

            # 显示数据预览
            print("\n数据预览:")
            print(combined_df.head())

            # 显示统计摘要
            print("\n统计摘要:")
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary = combined_df[numeric_cols].describe().round(4)
                print(summary)

            return combined_df

        except Exception as e:
            print(f"最终保存失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""

    # 配置路径
    image_dir = r"D:\med_data\MR\train_1"
    output_file = r"D:\med_data\MR\train_1\mri_statistical_features.csv"

    print("MRI图像特征提取程序")
    print("=" * 60)
    print(f"图像目录: {image_dir}")
    print(f"输出文件: {output_file}")

    # 验证目录
    if not os.path.exists(image_dir):
        print(f"错误: 目录不存在: {image_dir}")
        return

    # 创建提取器
    extractor = MRIFeatureExtractor(image_dir, output_file)

    # 处理所有图像
    features = extractor.process_all_images(save_frequency=20)

    if features:
        print("\n" + "=" * 60)
        print(f"成功提取了 {len(features)} 个病例的特征")
        print(f"数据已保存到: {output_file}")
    else:
        print("没有提取到特征")


# 快速处理单个图像的示例
def process_single_case():
    """处理单个病例的示例"""

    image_dir = r"D:\med_data\MR\train_1"
    image_file = "s001_T1.nii.gz"  # 示例文件名

    image_path = os.path.join(image_dir, image_file)

    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        # 查找第一个文件
        files = list(Path(image_dir).glob("*.nii.gz"))
        if files:
            image_path = files[0]
            print(f"使用第一个文件: {image_path.name}")

    extractor = MRIFeatureExtractor(image_dir)

    # 提取病例ID
    case_id = extractor.extract_case_id(Path(image_path))

    # 处理单个图像
    features = extractor.process_single_image(image_path, case_id)

    print("\n提取的特征:")
    for key, value in features.items():
        if key != 'case_id':
            print(f"  {key}: {value:.4f}")

    return features


# 批量处理函数（可导入使用）
def batch_extract_features(image_dir, output_file=None, resume=True):
    """
    批量提取特征

    参数:
        image_dir: 图像目录
        output_file: 输出文件（可选）
        resume: 是否从现有文件恢复

    返回:
        DataFrame: 包含所有特征的DataFrame
    """
    extractor = MRIFeatureExtractor(image_dir, output_file)

    if not resume:
        # 如果不恢复，删除现有文件
        if extractor.output_file.exists():
            extractor.output_file.unlink()

    features = extractor.process_all_images(save_frequency=20)

    if features:
        # 重新加载完整数据
        if extractor.output_file.exists():
            return pd.read_csv(extractor.output_file)

    return pd.DataFrame()


if __name__ == "__main__":
    # 安装必要的库
    try:
        import nibabel
        import scipy
        import pandas
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装:")
        print("  pip install nibabel scipy pandas numpy")
        exit(1)

    # 运行主函数
    main()

    # 可选：处理单个图像测试
    # process_single_case()