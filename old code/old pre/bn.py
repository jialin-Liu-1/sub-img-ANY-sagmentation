import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import pydicom
from PIL import Image

warnings.filterwarnings('ignore')


class DSANormalizer:
    """DSA图像归一化处理器（保持原始精度）"""

    def __init__(self):
        self.stats = {
            'dsa_images': {'processed': 0, 'errors': 0},
            'mask_images': {'processed': 0, 'errors': 0}
        }

    def load_dicom_image(self, dicom_path):
        """加载DICOM图像文件"""
        try:
            dicom_data = pydicom.dcmread(dicom_path, force=True)
            image = dicom_data.pixel_array.astype(np.float32)

            return image, dicom_data

        except Exception as e:
            print(f"加载DICOM失败 {dicom_path}: {e}")
            return None, None

    def normalize_dsa_image_simple(self, image_path):
        """
        简单的线性归一化
        原始值范围 → 0-1范围
        """
        try:
            file_ext = os.path.splitext(image_path)[1].lower()

            if file_ext == '.dcm' or file_ext == '':
                image, dicom_data = self.load_dicom_image(image_path)
                if image is None:
                    raise ValueError(f"无法读取DICOM图像: {image_path}")
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"无法读取图像: {image_path}")
                dicom_data = None

            image = image.astype(np.float32)

            # 显示原始统计
            orig_min = image.min()
            orig_max = image.max()
            print(f"  原始范围: [{orig_min:.1f}, {orig_max:.1f}]")

            # 简单的线性归一化到0-1
            if orig_max > orig_min:
                image_normalized = (image - orig_min) / (orig_max - orig_min)
            else:
                image_normalized = np.zeros_like(image)

            print(f"  归一化后: [{image_normalized.min():.3f}, {image_normalized.max():.3f}]")

            return image_normalized, dicom_data, (orig_min, orig_max)

        except Exception as e:
            print(f"DSA图像归一化失败: {e}")
            return None, None, None

    def save_dicom_for_display(self, normalized_image, original_dicom_data, output_path, orig_range):
        """
        保存为可视化的DICOM格式
        关键：存储为8位，设置合适的窗宽窗位
        """
        try:
            if original_dicom_data is not None:
                new_dataset = original_dicom_data.copy()
            else:
                new_dataset = pydicom.Dataset()
                new_dataset.PatientName = "Normalized_DSA"
                new_dataset.StudyDescription = "Normalized for Display"

            # 关键步骤1：将0-1的归一化图像转换回类似原始范围的8位图像
            # 这样看图软件才能正常显示

            # 方法A：直接映射到0-255（标准8位图像）
            display_image = (normalized_image * 255).astype(np.uint8)

            print(f"  显示图像范围: [{display_image.min()}, {display_image.max()}] (0-255)")

            # 更新像素数据
            new_dataset.PixelData = display_image.tobytes()
            new_dataset.Rows, new_dataset.Columns = display_image.shape

            # 关键步骤2：设置为8位DICOM
            new_dataset.BitsAllocated = 8
            new_dataset.BitsStored = 8
            new_dataset.HighBit = 7
            new_dataset.PixelRepresentation = 0  # 无符号

            # 关键步骤3：设置合理的窗宽窗位（针对0-255的8位图像）
            # 对于归一化到0-1再转0-255的图像
            new_dataset.WindowCenter = 128  # 中间值
            new_dataset.WindowWidth = 256  # 全范围

            # 或者根据实际图像设置
            pixel_min = display_image.min()
            pixel_max = display_image.max()
            if pixel_max > pixel_min:
                window_center = (pixel_min + pixel_max) // 2
                window_width = pixel_max - pixel_min
                if window_width < 50:  # 避免窗口太小
                    window_width = 256
            else:
                window_center = 128
                window_width = 256

            new_dataset.WindowCenter = window_center
            new_dataset.WindowWidth = window_width

            # 其他必要标签
            new_dataset.SamplesPerPixel = 1
            new_dataset.PhotometricInterpretation = "MONOCHROME2"

            # 保存（无后缀）
            pydicom.dcmwrite(output_path, new_dataset, write_like_original=False)

            print(f"  ✓ 保存为显示用DICOM")
            print(f"    像素范围: {display_image.min()}-{display_image.max()}")
            print(f"    窗宽窗位: Center={window_center}, Width={window_width}")

            # 同时保存一个PNG副本用于验证
            png_path = output_path + "_preview.png"
            cv2.imwrite(png_path, display_image)
            print(f"    预览PNG: {png_path}")

            return True

        except Exception as e:
            print(f"保存DICOM失败: {e}")
            # 尝试保存为TIFF
            try:
                tiff_path = output_path + ".tiff"
                display_image = (normalized_image * 255).astype(np.uint8)
                Image.fromarray(display_image).save(tiff_path)
                print(f"  → 转为TIFF: {tiff_path}")
                return True
            except:
                return False

    def normalize_mask_image(self, mask_path):
        """
        归一化mask图像
        等于0的部分设为0，大于0的部分设为1
        """
        try:
            file_ext = os.path.splitext(mask_path)[1].lower()

            if file_ext in ['.tif', '.tiff']:
                mask = Image.open(mask_path)
                mask = np.array(mask).astype(np.float32)
            elif file_ext == '.dcm' or file_ext == '':
                mask, _ = self.load_dicom_image(mask_path)
                if mask is None:
                    raise ValueError(f"无法读取DICOM mask: {mask_path}")
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"无法读取mask: {mask_path}")
                mask = mask.astype(np.float32)

            print(f"  Mask原始: min={mask.min():.1f}, max={mask.max():.1f}")

            # 二值化
            mask_binary = np.where(mask > 0, 1.0, 0.0)
            mask_binary = mask_binary.astype(np.float32)

            print(f"  二值化: 0像素={np.sum(mask_binary == 0):,}, 1像素={np.sum(mask_binary == 1):,}")

            return mask_binary

        except Exception as e:
            print(f"Mask归一化失败: {e}")
            return None

    def process_directory(self, input_dir, output_dir, is_mask=False, max_samples=None):
        """
        处理整个目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件
        all_files = []
        for f in os.listdir(input_dir):
            file_path = os.path.join(input_dir, f)
            if os.path.isfile(file_path):
                all_files.append(f)

        print(f"在 {input_dir} 中找到 {len(all_files)} 个文件")

        if max_samples:
            all_files = all_files[:max_samples]
            print(f"限制处理前 {max_samples} 个样本")

        processed_count = 0
        error_count = 0

        print(f"\n开始处理{'mask' if is_mask else 'DSA'}图像...")

        for img_file in tqdm(all_files, desc=f"处理{'mask' if is_mask else 'DSA'}"):
            try:
                input_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, img_file)

                if is_mask:
                    normalized = self.normalize_mask_image(input_path)

                    if normalized is not None:
                        # 保存为TIFF
                        file_ext = os.path.splitext(img_file)[1].lower()
                        if file_ext not in ['.tif', '.tiff']:
                            output_path = output_path + '.tiff'

                        mask_8bit = (normalized * 255).astype(np.uint8)
                        Image.fromarray(mask_8bit).save(output_path, format='TIFF')

                        processed_count += 1

                        if processed_count <= 2:
                            print(f"\n✓ 处理mask: {img_file}")
                            print(f"  保存到: {output_path}")
                    else:
                        error_count += 1

                else:
                    # DSA处理
                    normalized, dicom_data, orig_range = self.normalize_dsa_image_simple(input_path)

                    if normalized is not None:
                        success = self.save_dicom_for_display(normalized, dicom_data, output_path, orig_range)

                        if success:
                            processed_count += 1

                            if processed_count <= 2:
                                print(f"\n✓ 处理DSA: {img_file}")
                                print(f"  保存到: {output_path}")
                        else:
                            error_count += 1
                    else:
                        error_count += 1

            except Exception as e:
                error_count += 1
                print(f"\n✗ 处理 {img_file} 失败: {e}")

        if is_mask:
            self.stats['mask_images']['processed'] = processed_count
            self.stats['mask_images']['errors'] = error_count
        else:
            self.stats['dsa_images']['processed'] = processed_count
            self.stats['dsa_images']['errors'] = error_count

        return processed_count, error_count

    def test_viewing(self, output_dir):
        """测试图像能否正常查看"""
        print(f"\n测试图像查看...")
        files = os.listdir(output_dir)
        if not files:
            print("  无文件可测试")
            return

        test_file = files[0]
        test_path = os.path.join(output_dir, test_file)

        # 检查文件大小
        file_size = os.path.getsize(test_path)
        print(f"  测试文件: {test_file}")
        print(f"  文件大小: {file_size:,} bytes")

        # 尝试读取并显示信息
        try:
            if test_file.endswith('.tiff') or test_file.endswith('.tif'):
                img = Image.open(test_path)
                print(f"  TIFF信息:")
                print(f"    格式: {img.format}")
                print(f"    尺寸: {img.size}")
                print(f"    模式: {img.mode}")

                # 转换为numpy数组查看像素值
                img_array = np.array(img)
                print(f"    像素范围: [{img_array.min()}, {img_array.max()}]")

            else:
                # 可能是DICOM
                try:
                    dicom_data = pydicom.dcmread(test_path, force=True, stop_before_pixels=True)
                    print(f"  DICOM信息:")
                    print(f"    位深: {dicom_data.BitsAllocated}位")
                    if hasattr(dicom_data, 'WindowCenter'):
                        print(f"    窗中心: {dicom_data.WindowCenter}")
                    if hasattr(dicom_data, 'WindowWidth'):
                        print(f"    窗宽: {dicom_data.WindowWidth}")
                except:
                    # 不是DICOM，尝试作为图像读取
                    img_array = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is not None:
                        print(f"  图像信息:")
                        print(f"    尺寸: {img_array.shape}")
                        print(f"    像素范围: [{img_array.min()}, {img_array.max()}]")
                        print(f"    数据类型: {img_array.dtype}")

                        # 检查是否适合显示
                        if img_array.max() <= 255:
                            print(f"  ✓ 适合显示 (0-255范围)")
                        else:
                            print(f"  ⚠️ 可能显示异常 (值过高: {img_array.max()})")
        except Exception as e:
            print(f"  测试失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("DSA图像归一化程序（保持显示兼容性）")
    print("目标: 1) 线性归一化 2) 看图软件正常查看")
    print("=" * 70)

    config = {
        'original_dsa_dir': r"D:\med_data\ai\train1",
        'original_mask_dir': r"D:\med_data\ai\train2",
        'normalized_dsa_dir': r"D:\med_data\ai\train11",
        'normalized_mask_dir': r"D:\med_data\ai\train22",
        'max_samples': 5,
    }

    print("配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)

    normalizer = DSANormalizer()

    try:
        # 处理DSA
        print("\n1. 归一化DSA图像（保持显示兼容）")
        print("-" * 40)
        dsa_processed, dsa_errors = normalizer.process_directory(
            input_dir=config['original_dsa_dir'],
            output_dir=config['normalized_dsa_dir'],
            is_mask=False,
            max_samples=config['max_samples']
        )

        # 测试DSA图像查看
        if dsa_processed > 0:
            normalizer.test_viewing(config['normalized_dsa_dir'])

        # 处理Mask
        print("\n2. 归一化Mask图像")
        print("-" * 40)
        mask_processed, mask_errors = normalizer.process_directory(
            input_dir=config['original_mask_dir'],
            output_dir=config['normalized_mask_dir'],
            is_mask=True,
            max_samples=config['max_samples']
        )

        # 测试Mask图像查看
        if mask_processed > 0:
            normalizer.test_viewing(config['normalized_mask_dir'])

        print("\n" + "=" * 70)
        print("测试完成!")
        print(f"DSA图像: 成功 {dsa_processed} / 失败 {dsa_errors}")
        print(f"Mask图像: 成功 {mask_processed} / 失败 {mask_errors}")

        if dsa_processed > 0 and config['max_samples'] is not None:
            print("\n" + "=" * 70)
            response = input("测试成功！是否处理所有文件？(y/n): ")
            if response.lower() == 'y':
                print("开始处理所有文件...")

                dsa_processed, dsa_errors = normalizer.process_directory(
                    input_dir=config['original_dsa_dir'],
                    output_dir=config['normalized_dsa_dir'],
                    is_mask=False,
                    max_samples=None
                )

                mask_processed, mask_errors = normalizer.process_directory(
                    input_dir=config['original_mask_dir'],
                    output_dir=config['normalized_mask_dir'],
                    is_mask=True,
                    max_samples=None
                )

                print(f"\n全部完成!")
                print(f"DSA图像: 成功 {dsa_processed} / 失败 {dsa_errors}")
                print(f"Mask图像: 成功 {mask_processed} / 失败 {mask_errors}")

        print("=" * 70)

        # 重要提示
        print("\n💡 重要提示:")
        print("1. DSA图像保存为8位DICOM，像素值在0-255之间")
        print("2. 设置了合适的窗宽窗位（Center=128, Width=256）")
        print("3. 同时保存了PNG预览文件（文件名_preview.png）")
        print("4. 如果Fiji仍然显示异常，请检查窗宽窗位设置")
        print("5. 可以尝试用其他软件（如RadiAnt DICOM Viewer）查看")

    except Exception as e:
        print(f"处理失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
