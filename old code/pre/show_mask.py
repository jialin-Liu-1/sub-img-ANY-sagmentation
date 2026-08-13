import os
import torch
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


# 导入之前定义的AttentionMaskGenerator
# 导入之前定义的AttentionMaskGenerator
class AttentionMaskGenerator(torch.nn.Module):

    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()

        self.H, self.W = image_size  # 固定图像尺寸
        self.device = torch.device('cpu')  # 初始化设备

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, height_ratio: torch.Tensor,
                width_ratio: torch.Tensor) -> torch.Tensor:
        """
        生成长方形注意力mask

        参数:
            height_ratio: [B,] 高度比例（窗位），0-1之间
            width_ratio: [B,] 宽度比例（窗宽），0-1之间

        返回:
            attention_mask: [B, 1, H, W] 注意力mask（注意力区域为1，背景区域为0）
        """
        B = height_ratio.shape[0]
        H, W = self.H, self.W

        # 确保使用正确的设备
        if height_ratio.device != self.device:
            self.device = height_ratio.device

        batch_masks = []

        for b in range(B):
            h_ratio = height_ratio[b].item()
            w_ratio = width_ratio[b].item()

            # 计算中心高度位置（窗位）
            # 高度比例从顶部到底部，0表示顶部，1表示底部
            y_center = int(h_ratio * (H - 1))
            y_center = max(0, min(y_center, H - 1))

            # 计算注意力高度（窗宽）
            # 直接使用宽度比例乘以图像高度，得到实际高度范围
            # w_ratio是0-1之间的值，表示窗宽占图像高度的比例
            window_height = int(w_ratio * H)  # 窗宽（实际高度）

            # 确保窗宽至少为1像素
            if window_height < 1:
                window_height = 1
            elif window_height > H:
                window_height = H

            # 计算长方形边界
            half_height = window_height // 2
            y_min = max(0, y_center - half_height)
            y_max = min(H - 1, y_center + half_height)

            # 如果窗宽是奇数，确保总高度正确
            if window_height % 2 == 1:
                # 调整上边界或下边界
                if y_min > 0:
                    y_min -= 1
                elif y_max < H - 1:
                    y_max += 1

            # 创建长方形mask
            # 创建一个全0的背景mask
            attention_mask = torch.zeros(H, W, device=self.device)

            # 在高度范围内设置为1（注意力区域）
            # 注意：长方形从图像最左侧到最右侧（宽度为W）
            attention_mask[y_min:y_max + 1, :] = 1.0

            batch_masks.append(attention_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(batch_masks, dim=0)


class DSAImageProcessor:
    """DSA图像处理器"""

    def __init__(self,
                 image_dir: str = r"D:\med_data\ai\translate\train_all_trans(1)",
                 excel_path: str = r"D:\med_data\ai\translate\location_trans.xlsx",
                 output_dir: str = r"D:\med_data\ai\translate\mask1",
                 image_size: Tuple[int, int] = (512, 512),
                 max_radius: int = 50):
        """
        初始化处理器

        参数:
            image_dir: 图像目录路径
            excel_path: Excel表格路径
            output_dir: 输出目录路径
            image_size: 图像尺寸
            max_radius: 最大半径
        """
        self.image_dir = Path(image_dir)
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.max_radius = max_radius

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化注意力mask生成器
        self.attention_generator = AttentionMaskGenerator(
            image_size=image_size
        )

        # 加载Excel表格
        self.load_excel_data()

        # 获取所有图像文件
        self.image_files = self.get_image_files()

        print(f"找到 {len(self.image_files)} 个图像文件")
        print(f"表格中有 {len(self.excel_data)} 条记录")

    def load_excel_data(self):
        """加载Excel表格数据"""
        try:
            # 读取Excel文件
            self.excel_data = pd.read_excel(self.excel_path)

            # 检查必要的列
            required_columns = ['病历号', '高度比例', '宽度比例']
            for col in required_columns:
                if col not in self.excel_data.columns:
                    # 尝试其他可能的列名
                    if col == '病历号':
                        possible_names = ['filename', 'image_name', '病例号', '图像名']
                        for name in possible_names:
                            if name in self.excel_data.columns:
                                self.excel_data.rename(columns={name: '病历号'}, inplace=True)
                                break
                    elif col == '高度比例':
                        possible_names = ['height_ratio', 'height', 'y_ratio']
                        for name in possible_names:
                            if name in self.excel_data.columns:
                                self.excel_data.rename(columns={name: '高度比例'}, inplace=True)
                                break
                    elif col == '宽度比例':
                        possible_names = ['width_ratio', 'width', 'x_ratio', 'radius_ratio']
                        for name in possible_names:
                            if name in self.excel_data.columns:
                                self.excel_data.rename(columns={name: '宽度比例'}, inplace=True)
                                break

            # 再次检查必要的列
            if '病历号' not in self.excel_data.columns:
                # 使用第一列作为病历号
                self.excel_data.rename(columns={self.excel_data.columns[0]: '病历号'}, inplace=True)

            if '高度比例' not in self.excel_data.columns:
                # 使用第二列作为高度比例
                self.excel_data.rename(columns={self.excel_data.columns[1]: '高度比例'}, inplace=True)

            if '宽度比例' not in self.excel_data.columns:
                # 使用第三列作为宽度比例
                self.excel_data.rename(columns={self.excel_data.columns[2]: '宽度比例'}, inplace=True)

            print("Excel数据加载成功")
            print(f"列名: {list(self.excel_data.columns)}")

        except Exception as e:
            print(f"加载Excel数据失败: {e}")
            raise

    def get_image_files(self):
        """获取所有图像文件"""
        image_files = []

        # 遍历目录，查找无后缀的文件
        for file_path in self.image_dir.iterdir():
            if file_path.is_file() and not file_path.suffix:
                image_files.append(file_path)

        return image_files

    def load_dicom_image(self, file_path: Path) -> np.ndarray:
        """加载DICOM图像"""
        try:
            # 读取DICOM文件
            dicom_data = pydicom.dcmread(str(file_path))

            # 提取像素数据
            image_array = dicom_data.pixel_array

            # 转换为灰度图像（如果是彩色）
            if len(image_array.shape) == 3:
                # 如果是RGB，转换为灰度
                image_array = np.mean(image_array, axis=2).astype(np.uint8)

            # 调整大小到512x512
            if image_array.shape != self.image_size:
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image_array)
                pil_image = pil_image.resize(self.image_size, PILImage.Resampling.LANCZOS)
                image_array = np.array(pil_image)

            return image_array

        except Exception as e:
            print(f"加载DICOM图像失败 {file_path.name}: {e}")
            return None

    def find_image_info(self, image_name: str) -> Tuple[float, float]:
        """在表格中查找图像信息"""
        # 查找对应的行
        matched_rows = self.excel_data[self.excel_data['病历号'] == image_name]

        if len(matched_rows) == 0:
            # 尝试去除扩展名（如果有的话）
            base_name = image_name.split('.')[0]
            matched_rows = self.excel_data[self.excel_data['病历号'] == base_name]

        if len(matched_rows) == 0:
            print(f"警告: 未找到图像 {image_name} 在表格中的记录")
            return None, None

        # 获取高度和宽度比例
        row = matched_rows.iloc[0]
        height_ratio = float(row['高度比例'])
        width_ratio = float(row['宽度比例'])

        # 验证比例值在合理范围内
        if not (0 <= height_ratio <= 1):
            print(f"警告: 图像 {image_name} 的高度比例 {height_ratio} 不在0-1范围内")
            height_ratio = max(0, min(height_ratio, 1))

        if not (0 <= width_ratio <= 1):
            print(f"警告: 图像 {image_name} 的宽度比例 {width_ratio} 不在0-1范围内")
            width_ratio = max(0, min(width_ratio, 1))

        return height_ratio, width_ratio

    def normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """归一化图像到0-255范围"""
        # 转换为float
        image_float = image_array.astype(np.float32)

        # 归一化到0-1
        image_min = image_float.min()
        image_max = image_float.max()

        if image_max > image_min:
            image_normalized = (image_float - image_min) / (image_max - image_min)
        else:
            image_normalized = np.zeros_like(image_float)

        # 转换到0-255
        image_uint8 = (image_normalized * 255).astype(np.uint8)

        return image_uint8

    def apply_attention_mask(self, image_array: np.ndarray,
                             height_ratio: float, width_ratio: float) -> np.ndarray:
        """应用注意力mask到图像"""
        # 转换为torch tensor
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0) / 255.0

        # 创建高度和宽度比例tensor
        height_tensor = torch.tensor([height_ratio], dtype=torch.float32)
        width_tensor = torch.tensor([width_ratio], dtype=torch.float32)

        # 生成注意力mask
        attention_mask = self.attention_generator(height_tensor, width_tensor)

        # 应用mask到图像
        focused_image = image_tensor * attention_mask

        # 转换回numpy数组
        focused_image_np = focused_image.squeeze().numpy()

        return focused_image_np, attention_mask.squeeze().numpy()

    def save_image(self, image_array: np.ndarray, image_name: str, is_mask: bool = False):
        """保存图像"""
        # 确定保存路径
        if is_mask:
            save_dir = self.output_dir / "masks"
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"{image_name}_mask.png"
        else:
            save_path = self.output_dir / f"{image_name}.png"

        # 确保图像在0-255范围内
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        # 保存图像
        Image.fromarray(image_array).save(save_path)
        print(f"保存图像到: {save_path}")

    def process_single_image(self, image_path: Path):
        """处理单个图像"""
        image_name = image_path.name
        print(f"\n处理图像: {image_name}")

        # 1. 加载DICOM图像
        print("  1. 加载DICOM图像...")
        image_array = self.load_dicom_image(image_path)
        if image_array is None:
            print(f"  跳过图像 {image_name}")
            return False

        # 2. 在表格中查找图像信息
        print("  2. 查找图像信息...")
        height_ratio, width_ratio = self.find_image_info(image_name)
        if height_ratio is None or width_ratio is None:
            print(f"  跳过图像 {image_name} (未找到信息)")
            return False

        print(f"    高度比例: {height_ratio:.4f}, 宽度比例: {width_ratio:.4f}")

        # 3. 归一化原始图像
        print("  3. 归一化图像...")
        normalized_image = self.normalize_image(image_array)

        # 4. 保存原始图像（可选）
        # self.save_image(normalized_image, f"{image_name}_original")

        # 5. 应用注意力mask
        print("  4. 生成和应用注意力mask...")
        focused_image, attention_mask = self.apply_attention_mask(
            normalized_image, height_ratio, width_ratio
        )

        # 6. 保存注意力mask
        print("  5. 保存结果...")
        mask_uint8 = (attention_mask * 255).astype(np.uint8)
        self.save_image(mask_uint8, image_name, is_mask=True)

        # 7. 保存应用了注意力的图像
        focused_uint8 = (focused_image * 255).astype(np.uint8)
        self.save_image(focused_uint8, image_name)

        return True

    def process_all_images(self):
        """处理所有图像"""
        success_count = 0
        fail_count = 0

        print("开始处理所有图像...")
        print("=" * 50)

        for i, image_path in enumerate(self.image_files):
            print(f"\n处理进度: {i + 1}/{len(self.image_files)}")

            try:
                if self.process_single_image(image_path):
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                print(f"处理图像 {image_path.name} 时出错: {e}")
                fail_count += 1

        print("\n" + "=" * 50)
        print(f"处理完成!")
        print(f"成功: {success_count} 张图像")
        print(f"失败: {fail_count} 张图像")

        return success_count, fail_count

    def visualize_results(self, image_name: str):
        """可视化处理结果"""
        try:
            # 加载原始图像
            image_path = self.image_dir / image_name
            if not image_path.exists():
                print(f"图像 {image_name} 不存在")
                return

            image_array = self.load_dicom_image(image_path)
            normalized_image = self.normalize_image(image_array)

            # 获取图像信息
            height_ratio, width_ratio = self.find_image_info(image_name)
            if height_ratio is None or width_ratio is None:
                print(f"未找到图像 {image_name} 的信息")
                return

            # 生成处理结果
            focused_image, attention_mask = self.apply_attention_mask(
                normalized_image, height_ratio, width_ratio
            )

            # 可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原始图像
            axes[0].imshow(normalized_image, cmap='gray')
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            # 注意力mask
            axes[1].imshow(attention_mask, cmap='gray')
            axes[1].set_title('注意力mask')
            axes[1].axis('off')

            # 应用了注意力的图像
            axes[2].imshow(focused_image, cmap='gray')
            axes[2].set_title('应用注意力的图像')
            axes[2].axis('off')

            plt.suptitle(f"图像: {image_name}\n高度比例: {height_ratio:.3f}, 宽度比例: {width_ratio:.3f}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"可视化失败: {e}")


def main():
    """主函数"""
    # 初始化处理器
    processor = DSAImageProcessor(
        image_dir=r"D:\med_data\ai\translate\train_all_trans(1)",
        excel_path=r"D:\med_data\ai\translate\location_trans.xlsx",
        output_dir=r"D:\med_data\ai\translate\mask1",
        image_size=(512, 512),
        max_radius=50
    )

    # 处理所有图像
    success_count, fail_count = processor.process_all_images()

    # 可选：可视化某个图像的结果
    # processor.visualize_results("ANY_450_0")

    return success_count, fail_count


if __name__ == "__main__":
    # 运行主函数
    success, fail = main()

    # 打印总结
    print(f"\n{'=' * 50}")
    print(f"处理总结:")
    print(f"  成功处理: {success} 张图像")
    print(f"  失败: {fail} 张图像")
    print(f"  输出目录: D:\\med_data\\ai\\mask1")
    print(f"    原始图像: D:\\med_data\\ai\\mask1\\*.png")
    print(f"    注意力mask: D:\\med_data\\ai\\mask1\\masks\\*_mask.png")
    print(f"{'=' * 50}")