import os
import numpy as np
import nibabel as nib
from scipy import stats
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class MRIFeatureExtractorNormalized:
    """MRI图像特征提取器（包含归一化）"""

    def __init__(self, image_dir, output_file=None, precision=4):
        """
        初始化特征提取器

        参数:
            image_dir: 图像目录路径
            output_file: 输出文件路径（可选）
            precision: 保留小数位数（默认4位）
        """
        self.image_dir = Path(image_dir)
        self.precision = precision  # 精度控制

        if output_file is None:
            # 默认输出到同一目录
            self.output_file = self.image_dir / f"mri_features_precision{precision}.csv"
        else:
            self.output_file = Path(output_file)

        # 存储全局统计信息用于归一化
        self.global_stats = {
            'energy_min': float('inf'),
            'energy_max': float('-inf'),
            'skewness_min': float('inf'),
            'skewness_max': float('-inf'),
            'kurtosis_min': float('inf'),
            'kurtosis_max': float('-inf')
        }

        # 第一阶段：收集全局统计信息
        self.collect_global_statistics()

        # 特征列表（按您的要求）
        self.feature_names = [
            'mean_intensity',  # 平均强度
            'std_intensity',  # 强度标准差
            'skewness',  # 偏度（原始值）
            'skewness_norm',  # 偏度（归一化0-1）
            'kurtosis',  # 峰度（原始值）
            'kurtosis_norm',  # 峰度（归一化0-1）
            'median_intensity',  # 强度中位数
            'range_intensity',  # 强度范围（max-min）
            'iqr',  # 四分位距
            'mad',  # 平均绝对偏差
            'coeff_variation',  # 变异系数（std/mean）
            'energy',  # 能量（原始值）
            'energy_norm',  # 能量（归一化0-1）
            'contrast',  # 对比度（同range_intensity）
            'uniformity',  # 均匀度
        ]

        print(f"特征提取器已初始化，精度设置为{precision}位小数")

    def format_float(self, value):
        """格式化浮点数到指定精度"""
        if np.isnan(value) or pd.isna(value):
            return ''

        val = float(value)

        # 根据数值大小选择合适的格式
        if abs(val) < 1e-4 or abs(val) > 1e8:
            # 对于非常大或非常小的数，使用科学计数法
            return f"{val:.{self.precision}e}"
        else:
            # 对于普通数值，使用固定小数位数
            return f"{val:.{self.precision}f}"

    def collect_global_statistics(self):
        """第一阶段：收集全局统计信息用于归一化"""
        print("收集全局统计信息用于归一化...")

        image_files = list(self.image_dir.glob("*.nii.gz"))

        if not image_files:
            print("未找到图像文件")
            return

        energies = []
        skewnesses = []
        kurtoses = []

        for i, img_file in enumerate(image_files[:50]):  # 使用前50个样本估计范围
            try:
                img = nib.load(str(img_file))
                data = img.get_fdata()

                # 只使用非零值（根据您的要求）
                brain_mask = data > 0
                brain_data = data[brain_mask]

                if len(brain_data) > 100:  # 确保有足够的数据
                    # 计算原始值
                    energy = np.sum(brain_data ** 2)
                    skew = stats.skew(brain_data)
                    kurt = stats.kurtosis(brain_data)

                    energies.append(energy)
                    skewnesses.append(skew)
                    kurtoses.append(kurt)

            except Exception as e:
                print(f"  跳过 {img_file.name}: {e}")

        # 更新全局统计
        if energies:
            self.global_stats['energy_min'] = np.min(energies)
            self.global_stats['energy_max'] = np.max(energies)
            print(
                f"  能量范围: {self.format_float(self.global_stats['energy_min'])} - {self.format_float(self.global_stats['energy_max'])}")

        if skewnesses:
            self.global_stats['skewness_min'] = np.min(skewnesses)
            self.global_stats['skewness_max'] = np.max(skewnesses)
            print(
                f"  偏度范围: {self.format_float(self.global_stats['skewness_min'])} - {self.format_float(self.global_stats['skewness_max'])}")

        if kurtoses:
            self.global_stats['kurtosis_min'] = np.min(kurtoses)
            self.global_stats['kurtosis_max'] = np.max(kurtoses)
            print(
                f"  峰度范围: {self.format_float(self.global_stats['kurtosis_min'])} - {self.format_float(self.global_stats['kurtosis_max'])}")

        print("全局统计信息收集完成")

    def normalize_value(self, value, min_val, max_val, feature_name):
        """归一化值到0-1范围"""
        if np.isnan(value):
            return np.nan

        # 检查范围是否有效
        if min_val >= max_val:
            print(f"警告: {feature_name} 的范围无效 ({min_val} >= {max_val})，使用原始值")
            return value

        # 线性归一化到[0, 1]
        normalized = (value - min_val) / (max_val - min_val)

        # 确保在[0, 1]范围内（处理舍入误差）
        normalized = np.clip(normalized, 0.0, 1.0)

        return float(normalized)

    def extract_nonzero_data(self, image_data):
        """
        提取非零数据（根据您的要求，背景为0，非零为脑组织）

        参数:
            image_data: 3D图像数据

        返回:
            brain_data: 非零数据（1D数组）
            mask: 非零掩码
        """
        # 创建非零掩码
        mask = image_data > 0

        # 提取非零数据
        brain_data = image_data[mask]

        return brain_data, mask

    def calculate_all_features(self, brain_data, mask):
        """
        计算所有特征（包含归一化版本）

        参数:
            brain_data: 脑组织数据（非零数据）
            mask: 非零掩码

        返回:
            features: 特征字典
        """
        if len(brain_data) < 100:  # 数据太少
            features = {feature: np.nan for feature in self.feature_names}
            features['case_id'] = ''  # 占位符
            return features

        # 基本统计特征
        mean_val = np.mean(brain_data)
        std_val = np.std(brain_data)
        median_val = np.median(brain_data)
        min_val = np.min(brain_data)
        max_val = np.max(brain_data)

        # 使用格式化函数控制精度
        features = {
            'mean_intensity': self.format_float(float(mean_val)),
            'std_intensity': self.format_float(float(std_val)),
            'median_intensity': self.format_float(float(median_val)),
            'range_intensity': self.format_float(float(max_val - min_val)),
        }

        # 偏度和峰度（原始值）
        try:
            skew_val = float(stats.skew(brain_data))
            kurt_val = float(stats.kurtosis(brain_data))
        except:
            skew_val = 0.0
            kurt_val = 0.0

        features['skewness'] = self.format_float(skew_val)
        features['kurtosis'] = self.format_float(kurt_val)

        # 归一化偏度和峰度
        skewness_norm_val = self.normalize_value(
            skew_val,
            self.global_stats['skewness_min'],
            self.global_stats['skewness_max'],
            'skewness'
        )

        kurtosis_norm_val = self.normalize_value(
            kurt_val,
            self.global_stats['kurtosis_min'],
            self.global_stats['kurtosis_max'],
            'kurtosis'
        )

        features['skewness_norm'] = self.format_float(skewness_norm_val)
        features['kurtosis_norm'] = self.format_float(kurtosis_norm_val)

        # 四分位距
        q75, q25 = np.percentile(brain_data, [75, 25])
        features['iqr'] = self.format_float(float(q75 - q25))

        # 平均绝对偏差
        features['mad'] = self.format_float(float(np.mean(np.abs(brain_data - mean_val))))

        # 变异系数
        if mean_val != 0:
            coeff_var = float(std_val / mean_val)
            features['coeff_variation'] = self.format_float(coeff_var)
        else:
            features['coeff_variation'] = ''

        # 能量（原始值）
        energy_val = float(np.sum(brain_data ** 2))
        features['energy'] = self.format_float(energy_val)

        # 归一化能量
        energy_norm_val = self.normalize_value(
            energy_val,
            self.global_stats['energy_min'],
            self.global_stats['energy_max'],
            'energy'
        )
        features['energy_norm'] = self.format_float(energy_norm_val)

        # 对比度（与range_intensity相同）
        features['contrast'] = features['range_intensity']

        # 均匀度
        hist, _ = np.histogram(brain_data, bins=50)
        hist = hist.astype(float)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            uniformity_val = float(np.sum((hist / hist_sum) ** 2))
            features['uniformity'] = self.format_float(uniformity_val)
        else:
            features['uniformity'] = ''

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

            # 提取非零数据（根据您的要求）
            brain_data, mask = self.extract_nonzero_data(data)

            if len(brain_data) < 100:
                print(f"  ⚠ 非零体素太少: {len(brain_data)}")
                features = {feature: np.nan for feature in self.feature_names}
                features['case_id'] = case_id
                return features

            # 计算所有特征
            features = self.calculate_all_features(brain_data, mask)
            features['case_id'] = case_id

            print(f"  ✓ 完成: {len(brain_data)}个非零体素")

            # 显示归一化值
            if 'skewness' in features and 'skewness_norm' in features:
                print(f"    偏度: {features['skewness']} → {features['skewness_norm']}")
            if 'kurtosis' in features and 'kurtosis_norm' in features:
                print(f"    峰度: {features['kurtosis']} → {features['kurtosis_norm']}")
            if 'energy' in features and 'energy_norm' in features:
                print(f"    能量: {features['energy']} → {features['energy_norm']}")

            return features

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            features = {feature: '' for feature in self.feature_names}
            features['case_id'] = case_id
            return features

    def extract_case_id(self, image_path):
        """
        从文件名提取病例ID

        参数:
            image_path: 图像文件路径

        返回:
            case_id: 病例ID
        """
        filename = image_path.stem.replace('.nii', '')

        import re

        # 优先匹配 s001, s002 等格式
        match = re.search(r'[sS](\d{1,3})', filename)
        if match:
            num = match.group(1).zfill(3)  # 填充为3位
            return f"s{num}"

        # 匹配数字部分
        match = re.search(r'(\d{1,3})', filename)
        if match:
            num = match.group(1).zfill(3)
            return f"s{num}"

        # 如果都没匹配到，使用文件名
        return filename

    def process_all_images(self, save_frequency=20):
        """
        处理所有图像

        参数:
            save_frequency: 保存频率（每处理N个病例保存一次）

        返回:
            all_features: 所有特征列表
        """
        print("=" * 60)
        print(f"开始提取MRI图像特征（精度: {self.precision}位小数）")
        print("=" * 60)

        # 显示归一化范围
        print("归一化范围:")
        print(
            f"  能量: [{self.format_float(self.global_stats['energy_min'])}, {self.format_float(self.global_stats['energy_max'])}]")
        print(
            f"  偏度: [{self.format_float(self.global_stats['skewness_min'])}, {self.format_float(self.global_stats['skewness_max'])}]")
        print(
            f"  峰度: [{self.format_float(self.global_stats['kurtosis_min'])}, {self.format_float(self.global_stats['kurtosis_max'])}]")
        print()

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
            final_df = self.save_features_final(all_features, existing_data)
            return final_df

        return pd.DataFrame()

    def load_existing_features(self):
        """加载已存在的特征文件"""
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
        """中间保存特征"""
        try:
            new_df = pd.DataFrame(new_features)

            if not existing_data.empty:
                existing_data = existing_data[~existing_data['case_id'].isin(new_df['case_id'])]
                combined_df = pd.concat([existing_data, new_df], ignore_index=True)
            else:
                combined_df = new_df

            combined_df.to_csv(self.output_file, index=False)

            print(f"\n[进度: {current}/{total}] 已保存中间结果")
            print(f"总病例数: {len(combined_df)}")

        except Exception as e:
            print(f"中间保存失败: {e}")

    def save_features_final(self, new_features, existing_data):
        """最终保存特征"""
        try:
            new_df = pd.DataFrame(new_features)

            # 设置列顺序
            base_cols = ['case_id', 'mean_intensity', 'std_intensity', 'skewness', 'skewness_norm',
                         'kurtosis', 'kurtosis_norm', 'median_intensity', 'range_intensity',
                         'iqr', 'mad', 'coeff_variation', 'energy', 'energy_norm',
                         'contrast', 'uniformity']

            # 确保所有列都存在
            missing_cols = [col for col in base_cols if col not in new_df.columns]
            for col in missing_cols:
                new_df[col] = ''

            new_df = new_df[base_cols]

            if not existing_data.empty:
                combined_df = pd.concat([existing_data, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['case_id'], keep='last')
            else:
                combined_df = new_df

            # 按case_id排序
            combined_df['case_num'] = combined_df['case_id'].str.extract(r'(\d+)').fillna(0).astype(int)
            combined_df = combined_df.sort_values('case_num').drop('case_num', axis=1)

            # 保存到CSV
            combined_df.to_csv(self.output_file, index=False)

            print("\n" + "=" * 60)
            print("特征提取完成!")
            print(f"输出文件: {self.output_file}")
            print(f"总病例数: {len(combined_df)}")
            print(f"特征数量: {len(combined_df.columns) - 1}")
            print(f"精度: {self.precision}位小数")

            # 显示归一化特征的统计
            print("\n归一化特征统计 (0-1范围):")
            norm_features = ['skewness_norm', 'kurtosis_norm', 'energy_norm']

            # 将字符串转换回数值进行统计
            for feat in norm_features:
                if feat in combined_df.columns:
                    # 转换为数值
                    numeric_series = pd.to_numeric(combined_df[feat], errors='coerce')
                    values = numeric_series.dropna()

                    if len(values) > 0:
                        print(f"  {feat}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

            return combined_df

        except Exception as e:
            print(f"最终保存失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_feature_summary_report(df, output_dir, precision=4):
    """创建特征摘要报告"""
    if df.empty:
        return

    report_file = Path(output_dir) / f"feature_summary_report_precision{precision}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("MRI图像特征提取报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"总病例数: {len(df)}\n")
        f.write(f"特征数量: {len(df.columns) - 1}\n")
        f.write(f"精度: {precision}位小数\n\n")

        f.write("特征描述:\n")
        feature_descriptions = {
            'mean_intensity': '平均强度（非零区域）',
            'std_intensity': '强度标准差（非零区域）',
            'skewness': '偏度（原始，非零区域）',
            'skewness_norm': '偏度（归一化0-1）',
            'kurtosis': '峰度（原始，非零区域）',
            'kurtosis_norm': '峰度（归一化0-1）',
            'median_intensity': '强度中位数（非零区域）',
            'range_intensity': '强度范围（max-min，非零区域）',
            'iqr': '四分位距（非零区域）',
            'mad': '平均绝对偏差（非零区域）',
            'coeff_variation': '变异系数(std/mean，非零区域）',
            'energy': '能量（原始，非零区域）',
            'energy_norm': '能量（归一化0-1）',
            'contrast': '对比度（同range_intensity）',
            'uniformity': '均匀度（非零区域）',
        }

        for col in df.columns:
            if col != 'case_id':
                desc = feature_descriptions.get(col, '未描述')
                f.write(f"{col}: {desc}\n")

        f.write("\n特征统计摘要:\n")

        # 将所有数值列转换为数值类型
        numeric_df = df.copy()
        for col in numeric_df.columns:
            if col != 'case_id':
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

        numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = numeric_df[col].dropna()
            if len(values) > 0:
                f.write(f"\n{col}:\n")
                f.write(f"  有效值: {len(values)}/{len(df)}\n")
                f.write(f"  最小值: {values.min():.{precision}f}\n")
                f.write(f"  最大值: {values.max():.{precision}f}\n")
                f.write(f"  平均值: {values.mean():.{precision}f}\n")
                f.write(f"  标准差: {values.std():.{precision}f}\n")

        # 检查归一化特征
        f.write("\n归一化特征检查:\n")
        norm_features = ['skewness_norm', 'kurtosis_norm', 'energy_norm']

        for feat in norm_features:
            if feat in numeric_df.columns:
                values = numeric_df[feat].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    f.write(f"{feat}: [{min_val:.{precision}f}, {max_val:.{precision}f}]")
                    if min_val >= 0 and max_val <= 1:
                        f.write(" ✓ 在[0,1]范围内\n")
                    else:
                        f.write(" ⚠ 超出[0,1]范围\n")

    print(f"报告已保存到: {report_file}")


def main():
    """主函数"""

    # 配置路径
    image_dir = r"D:\med_data\MR\TEST1"
    precision = 4  # 设置精度为4位小数

    # 自动生成输出文件名
    output_file = Path(image_dir) / f"test{precision}.csv"

    print("MRI图像特征提取程序（改进版）")
    print("=" * 60)
    print(f"图像目录: {image_dir}")
    print(f"输出文件: {output_file}")
    print(f"精度设置: {precision}位小数")
    print("数据处理: 仅使用非零区域（背景为0）")

    # 验证目录
    if not os.path.exists(image_dir):
        print(f"错误: 目录不存在: {image_dir}")
        return

    # 创建提取器
    extractor = MRIFeatureExtractorNormalized(image_dir, output_file, precision=precision)

    # 处理所有图像
    df = extractor.process_all_images(save_frequency=20)

    if df is not None and not df.empty:
        # 创建摘要报告
        create_feature_summary_report(df, image_dir, precision)

        # 显示数据预览
        print("\n" + "=" * 60)
        print(f"成功提取了 {len(df)} 个病例的特征")
        print(f"数据已保存到: {output_file}")

        print("\n数据前5行:")
        print(df.head())

        # 显示列信息
        print(f"\n特征列 ({len(df.columns) - 1}个):")
        for col in df.columns:
            if col != 'case_id':
                print(f"  - {col}")

    else:
        print("没有提取到特征")


if __name__ == "__main__":
    # 检查依赖
    try:
        import nibabel
        import scipy
        import pandas
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install nibabel scipy pandas numpy")
        exit(1)

    # 运行主函数
    main()