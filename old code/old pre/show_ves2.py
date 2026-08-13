import os
import numpy as np
import torch

import torch.nn.functional as F
from PIL import Image
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import matplotlib
# 设置matplotlib使用非交互式后端
matplotlib.use('Agg')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#warnings.filterwarnings('ignore')

# 导入血管注意力模块
try:
    from multi.ves_block1 import AsymmetricVascularAttention

    print("成功导入血管注意力模块")
except ImportError:
    print("导入血管注意力模块失败，使用虚拟模块")


    # 创建一个虚拟模块供测试
    class AsymmetricVascularAttention:
        def __init__(self, **kwargs):
            self.vessel_threshold = kwargs.get('vessel_threshold', 0.3)
            self.erosion_iterations = kwargs.get('erosion_iterations', 3)

        def extract_vessel_mask(self, image):
            """提取血管mask的简化版本"""
            # 使用简单的阈值方法
            binary_vessel = image < self.vessel_threshold
            return binary_vessel.astype(np.float32)


class DICOMVascularProcessor:
    """DICOM血管处理类"""

    def __init__(self,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 vessel_threshold=0.3,
                 erosion_iterations=3):

        self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # 初始化血管注意力模块
        self.attention_module = AsymmetricVascularAttention(
            vessel_threshold=vessel_threshold,
            erosion_iterations=erosion_iterations,
            max_attention_radius=40,
            dropout_rate=0.1
        ).to(self.device)

        # 设置为评估模式
        self.attention_module.eval()

    def load_dicom_image(self, dicom_path):
        """加载DICOM图像并预处理"""
        try:
            # 读取DICOM文件
            dicom_data = pydicom.dcmread(dicom_path, force=True)

            # 获取像素数组
            pixel_array = dicom_data.pixel_array.astype(np.float32)

            # DICOM图像可能有不同位深度，需要标准化到0-1范围
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                # 如果有窗宽窗位，使用它们进行标准化
                window_center = dicom_data.WindowCenter
                window_width = dicom_data.WindowWidth

                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]

                # 应用窗宽窗位
                min_val = window_center - window_width / 2
                max_val = window_center + window_width / 2

                # 裁剪到窗口范围
                pixel_array = np.clip(pixel_array, min_val, max_val)
                # 归一化到0-1
                pixel_array = (pixel_array - min_val) / (max_val - min_val)
            else:
                # 如果没有窗宽窗位，使用最大最小值归一化
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)

            # 转换为512x512（如果需要）
            if pixel_array.shape != (512, 512):
                # 使用PIL进行高质量resize
                from PIL import Image
                img_pil = Image.fromarray((pixel_array * 255).astype(np.uint8))
                img_pil = img_pil.resize((512, 512), Image.LANCZOS)
                pixel_array = np.array(img_pil).astype(np.float32) / 255.0

            print(f"成功加载图像: {os.path.basename(dicom_path)}, 尺寸: {pixel_array.shape}")
            return pixel_array

        except Exception as e:
            print(f"加载DICOM图像失败 {dicom_path}: {str(e)}")
            return None

    def extract_vessel_mask(self, image_array):
        """提取血管mask"""
        try:
            # 调用注意力模块提取血管mask
            vessel_mask = self.attention_module.extract_vessel_mask(image_array)

            # 确保mask是二值的（0或1）
            if vessel_mask.max() > 1:
                vessel_mask = vessel_mask / vessel_mask.max()

            return vessel_mask

        except Exception as e:
            print(f"提取血管mask失败: {str(e)}")
            # 使用简单的阈值方法作为后备
            binary_vessel = image_array < self.attention_module.vessel_threshold
            return binary_vessel.astype(np.float32)

    def generate_random_position_info(self):
        """生成随机的位置信息（模拟动脉瘤位置）"""
        # 随机生成0-1之间的值，表示在血管的不同位置
        position_value = np.random.uniform(0.0, 1.0)
        return torch.tensor([[position_value]], dtype=torch.float32).to(self.device)

    def generate_attention_map(self, image_array):
        """生成注意力权重图"""
        try:
            # 转换为tensor
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(self.device)

            # 生成随机位置信息
            position_info = self.generate_random_position_info()

            # 生成注意力图
            with torch.no_grad():
                attention_map = self.attention_module.generate_attention_map(
                    image_tensor, position_info
                )

            # 转回numpy并移除batch维度
            attention_map_np = attention_map.squeeze().cpu().numpy()

            return attention_map_np, position_info.item()

        except Exception as e:
            print(f"生成注意力图失败: {str(e)}")
            # 返回一个简单的圆形注意力图作为后备
            h, w = image_array.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 8
            dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
            attention_map = np.exp(-dist_sq / (2 * (radius ** 2)))
            return attention_map, 0.5

    def save_results(self, image_array, vessel_mask, attention_map,
                     output_dir, base_filename, position_value):
        """保存处理结果"""
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 1. 保存原始图像（归一化到0-255）
            image_uint8 = (image_array * 255).astype(np.uint8)
            image_path = os.path.join(output_dir, f"{base_filename}_original.png")
            Image.fromarray(image_uint8).save(image_path)

            # 2. 保存血管mask（归一化到0-255）
            mask_uint8 = (vessel_mask * 255).astype(np.uint8)
            mask_path = os.path.join(output_dir, f"{base_filename}_vessel_mask.png")
            Image.fromarray(mask_uint8).save(mask_path)

            # 3. 保存注意力权重图（归一化到0-255）
            attention_uint8 = (attention_map * 255).astype(np.uint8)
            attention_path = os.path.join(output_dir, f"{base_filename}_attention_map.png")
            Image.fromarray(attention_uint8).save(attention_path)

            # 4. 保存叠加可视化图像
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 原始图像
            axes[0, 0].imshow(image_array, cmap='gray')
            axes[0, 0].set_title(f'Original DSA Image\n{base_filename}')
            axes[0, 0].axis('off')

            # 血管mask
            axes[0, 1].imshow(vessel_mask, cmap='gray')
            axes[0, 1].set_title('Vessel Mask')
            axes[0, 1].axis('off')

            # 注意力图
            im = axes[1, 0].imshow(attention_map, cmap='hot')
            axes[1, 0].set_title(f'Attention Map (position={position_value:.3f})')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # 原始图像+注意力叠加
            axes[1, 1].imshow(image_array, cmap='gray', alpha=0.7)
            im2 = axes[1, 1].imshow(attention_map, cmap='hot', alpha=0.5)
            axes[1, 1].set_title('Overlay: Image + Attention')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # 保存可视化图像
            plt.tight_layout()
            viz_path = os.path.join(output_dir, f"{base_filename}_visualization.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 5. 保存元数据信息
            info_path = os.path.join(output_dir, f"{base_filename}_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Image: {base_filename}\n")
                f.write(f"Position value: {position_value:.4f}\n")
                f.write(f"Image shape: {image_array.shape}\n")
                f.write(f"Vessel mask non-zero ratio: {np.mean(vessel_mask > 0):.4f}\n")
                f.write(f"Attention map max: {attention_map.max():.4f}\n")
                f.write(f"Attention map min: {attention_map.min():.4f}\n")

            print(f"处理结果已保存到: {output_dir}")
            print(f"  - 原始图像: {os.path.basename(image_path)}")
            print(f"  - 血管mask: {os.path.basename(mask_path)}")
            print(f"  - 注意力图: {os.path.basename(attention_path)}")
            print(f"  - 可视化图: {os.path.basename(viz_path)}")
            print(f"  - 信息文件: {os.path.basename(info_path)}")

        except Exception as e:
            print(f"保存结果失败: {str(e)}")

    def process_single_image(self, dicom_path, output_dir):
        """处理单个DICOM图像"""
        print(f"\n{'=' * 60}")
        print(f"开始处理: {dicom_path}")
        print(f"{'=' * 60}")

        try:
            # 1. 加载DICOM图像
            image_array = self.load_dicom_image(dicom_path)
            if image_array is None:
                return False

            # 2. 提取血管mask
            print("提取血管mask...")
            vessel_mask = self.extract_vessel_mask(image_array)

            # 3. 生成注意力权重图
            print("生成注意力权重图...")
            attention_map, position_value = self.generate_attention_map(image_array)

            # 4. 获取基本文件名（无路径无后缀）
            base_filename = os.path.splitext(os.path.basename(dicom_path))[0]
            if base_filename == os.path.basename(dicom_path):  # 如果没有后缀
                base_filename = os.path.basename(dicom_path)

            # 5. 保存结果
            print("保存处理结果...")
            self.save_results(image_array, vessel_mask, attention_map,
                              output_dir, base_filename, position_value)

            print(f"✓ 成功处理: {os.path.basename(dicom_path)}")
            return True

        except Exception as e:
            print(f"✗ 处理失败 {dicom_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_directory(self, input_dir, output_dir, file_patterns=None):
        """处理目录中的所有图像"""
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"错误: 输入目录不存在 {input_dir}")
            return

        # 获取所有无后缀的文件（假设这些是DICOM文件）
        dicom_files = []
        for file_path in input_path.iterdir():
            if file_path.is_file():
                # 检查是否没有后缀
                if file_path.suffix == '':
                    dicom_files.append(str(file_path))
                # 或者检查是否是DICOM文件（根据命名模式）
                elif file_patterns and any(pattern in file_path.name for pattern in file_patterns):
                    dicom_files.append(str(file_path))

        print(f"在目录 {input_dir} 中找到 {len(dicom_files)} 个DICOM文件")

        if len(dicom_files) == 0:
            print("警告: 未找到DICOM文件")
            # 尝试查找任何文件
            all_files = [str(f) for f in input_path.iterdir() if f.is_file()]
            print(f"目录中的文件: {all_files}")
            return

        # 处理每个文件
        success_count = 0
        for i, dicom_file in enumerate(sorted(dicom_files)):
            print(f"\n处理文件 {i + 1}/{len(dicom_files)}: {os.path.basename(dicom_file)}")

            success = self.process_single_image(dicom_file, output_dir)
            if success:
                success_count += 1

        print(f"\n{'=' * 60}")
        print(f"处理完成!")
        print(f"成功处理: {success_count}/{len(dicom_files)} 个文件")
        print(f"输出目录: {output_dir}")
        print(f"{'=' * 60}")


def main():
    """主函数"""
    # 设置路径
    input_dir = r"D:\med_data\ai\ce"
    output_dir = r"D:\med_data\ai\ce1"

    print("DICOM血管处理程序")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 创建处理器实例
    processor = DICOMVascularProcessor(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        vessel_threshold=0.3,
        erosion_iterations=3
    )

    # 处理目录中的所有图像
    # 可以根据文件名模式过滤文件
    file_patterns = ['ANY_']  # 只处理包含ANY_的文件

    processor.process_directory(input_dir, output_dir, file_patterns)

    # 也可以指定特定文件处理
    print("\n可选: 指定文件处理示例")
    specific_files = [
        r"D:\med_data\ai\ce\ANY_001_0",
        r"D:\med_data\ANY\ce\ANY_001_1",
        r"D:\med_data\ANY\ce\ANY_002_0",
        r"D:\med_data\ANY\ce\ANY_002_1"
    ]

    # 检查特定文件是否存在并处理
    for file_path in specific_files:
        if os.path.exists(file_path):
            print(f"\n处理指定文件: {file_path}")
            processor.process_single_image(file_path, output_dir)
        else:
            print(f"文件不存在: {file_path}")


def test_with_sample_image():
    """使用样本图像测试"""
    print("\n使用样本图像测试...")

    # 创建一个样本DICOM图像（512x512）
    np.random.seed(42)
    sample_image = np.random.rand(512, 512) * 0.5

    # 在中心添加模拟血管（低像素值区域）
    center_y, center_x = 256, 256
    y, x = np.ogrid[:512, :512]

    # 创建主要血管
    dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
    vessel_region = dist_sq < 50 ** 2
    sample_image[vessel_region] = np.random.rand(*sample_image[vessel_region].shape) * 0.2

    # 创建分支血管
    branch1 = (y - 200) ** 2 + (x - 300) ** 2 < 30 ** 2
    sample_image[branch1] = np.random.rand(*sample_image[branch1].shape) * 0.25

    branch2 = (y - 300) ** 2 + (x - 200) ** 2 < 30 ** 2
    sample_image[branch2] = np.random.rand(*sample_image[branch2].shape) * 0.25

    # 初始化处理器
    processor = DICOMVascularProcessor(device='cpu')

    # 提取血管mask
    vessel_mask = processor.extract_vessel_mask(sample_image)

    # 生成注意力图
    image_tensor = torch.from_numpy(sample_image).unsqueeze(0).unsqueeze(0).to(processor.device)
    position_info = torch.tensor([[0.5]], dtype=torch.float32).to(processor.device)

    with torch.no_grad():
        attention_map = processor.attention_module.generate_attention_map(
            image_tensor, position_info
        )
    attention_map_np = attention_map.squeeze().cpu().numpy()

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sample_image, cmap='gray')
    axes[0].set_title('Sample DSA Image')
    axes[0].axis('off')

    axes[1].imshow(vessel_mask, cmap='gray')
    axes[1].set_title('Extracted Vessel Mask')
    axes[1].axis('off')

    im = axes[2].imshow(attention_map_np, cmap='hot')
    axes[2].set_title('Generated Attention Map (position=0.5)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(r"D:\med_data\ai\ce1\sample_test.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("样本测试完成，结果已保存")


if __name__ == "__main__":
    # 创建输出目录
    output_dir = r"D:\med_data\ai\ce1"
    os.makedirs(output_dir, exist_ok=True)

    # 运行主程序
    try:
        main()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback

        traceback.print_exc()

        # 尝试运行样本测试
        print("\n尝试运行样本测试...")
        test_with_sample_image()