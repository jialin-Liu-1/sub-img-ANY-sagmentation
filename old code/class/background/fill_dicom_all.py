import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy import ndimage
import shutil
from scipy import ndimage as ndi
from datetime import datetime


def extend_background_mask(background_mask, extension_pixels=10):
    """
    将背景区域向内延伸指定像素数

    参数:
    - background_mask: 原始背景掩码
    - extension_pixels: 向内延伸的像素数

    返回:
    - extended_mask: 延伸后的背景掩码
    """
    if extension_pixels <= 0:
        return background_mask

    # 使用形态学膨胀操作延伸背景区域
    kernel_size = 2 * extension_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    extended_mask = cv2.dilate(background_mask.astype(np.uint8), kernel, iterations=1)
    extended_mask = extended_mask.astype(bool)

    return extended_mask


def detect_edge_background_advanced(image, margin_ratio=0.15, intensity_threshold_ratio=0.1, extension_pixels=10):
    """
    高级边缘背景检测方法（添加延伸功能）

    参数:
    - image: 输入图像
    - margin_ratio: 边缘检测区域比例
    - intensity_threshold_ratio: 强度阈值比例
    - extension_pixels: 背景向内延伸的像素数

    返回:
    - left_crop: 左侧裁剪位置
    - right_crop: 右侧裁剪位置
    - background_mask: 背景掩码（包含延伸区域）
    - original_background_mask: 原始背景掩码（不含延伸）
    - has_background: 是否检测到背景区域
    """
    height, width = image.shape

    # 计算边缘区域
    margin_width = int(width * margin_ratio)

    # 方法1: 基于边缘强度分析
    left_edge = image[:, :margin_width]
    right_edge = image[:, -margin_width:]

    # 计算边缘区域的平均强度
    left_mean = np.mean(left_edge)
    right_mean = np.mean(right_edge)
    center_mean = np.mean(image[:, margin_width:-margin_width])

    # 计算全局阈值
    global_threshold = np.percentile(image, 5)

    # 方法2: 基于梯度分析
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_abs = np.abs(sobel_x)

    # 边缘区域的梯度强度
    left_gradient = np.mean(sobel_x_abs[:, :margin_width])
    right_gradient = np.mean(sobel_x_abs[:, -margin_width:])
    center_gradient = np.mean(sobel_x_abs[:, margin_width:-margin_width])

    # 方法3: 基于连通组件分析
    binary_image = image > global_threshold
    labeled_array, num_features = ndimage.label(binary_image)

    # 找到最大的连通组件（主要组织区域）
    component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
    if component_sizes:
        main_component = np.argmax(component_sizes) + 1
        main_component_mask = (labeled_array == main_component)

        # 找到主要组件的边界
        rows, cols = np.where(main_component_mask)
        left_bound = np.min(cols) if len(cols) > 0 else 0
        right_bound = np.max(cols) if len(cols) > 0 else width
    else:
        left_bound = 0
        right_bound = width

    # 综合判断
    left_crop = 0
    right_crop = width

    left_is_background = (left_mean < center_mean * 0.6 and
                          left_gradient < center_gradient * 0.4)
    right_is_background = (right_mean < center_mean * 0.6 and
                           right_gradient < center_gradient * 0.4)

    if left_is_background:
        left_crop = max(0, left_bound - 5)
    if right_is_background:
        right_crop = min(width, right_bound + 5)

    # 创建原始背景掩码
    original_background_mask = np.zeros_like(image, dtype=bool)
    has_background = False

    if left_is_background:
        original_background_mask[:, :left_crop] = True
        has_background = True
    if right_is_background:
        original_background_mask[:, right_crop:] = True
        has_background = True

    # 延伸背景区域
    if has_background and extension_pixels > 0:
        extended_background_mask = extend_background_mask(original_background_mask, extension_pixels)
    else:
        extended_background_mask = original_background_mask.copy()

    return left_crop, right_crop, extended_background_mask, original_background_mask, has_background


def fill_background_with_max_value(image, background_mask):
    """
    将背景区域设为图像的最大值
    """
    processed_image = image.copy()
    image_max = np.max(image)
    processed_image[background_mask] = image_max
    return processed_image


def save_consistent_dicom_with_dcmwrite(processed_image, original_dicom, output_path, processing_applied=False):
    """
    使用dcmwrite保存DICOM文件，支持无后缀文件名

    参数:
    - processed_image: 处理后的图像数据
    - original_dicom: 原始DICOM对象
    - output_path: 输出DICOM文件路径（可以无后缀）
    - processing_applied: 是否进行了背景处理
    """
    try:
        # 创建新的DICOM数据集
        output_ds = pydicom.Dataset()

        # 复制原始DICOM文件的元数据
        output_ds.file_meta = pydicom.Dataset()
        if hasattr(original_dicom, 'file_meta'):
            for elem in original_dicom.file_meta:
                output_ds.file_meta[elem.tag] = elem

        # 设置必要的文件元信息
        if not hasattr(output_ds.file_meta, 'MediaStorageSOPClassUID'):
            output_ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        if not hasattr(output_ds.file_meta, 'MediaStorageSOPInstanceUID'):
            output_ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        if not hasattr(output_ds.file_meta, 'TransferSyntaxUID'):
            output_ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # 复制患者和检查信息
        patient_study_tags = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'StudyDate', 'StudyTime', 'StudyID', 'StudyDescription',
            'AccessionNumber', 'ReferringPhysicianName'
        ]

        for tag in patient_study_tags:
            if hasattr(original_dicom, tag):
                setattr(output_ds, tag, getattr(original_dicom, tag))

        # 复制设备信息
        equipment_tags = [
            'Manufacturer', 'ManufacturerModelName', 'DeviceSerialNumber',
            'SoftwareVersions', 'StationName'
        ]

        for tag in equipment_tags:
            if hasattr(original_dicom, tag):
                setattr(output_ds, tag, getattr(original_dicom, tag))

        # 设置图像参数
        output_ds.SamplesPerPixel = 1
        output_ds.PhotometricInterpretation = "MONOCHROME2"
        output_ds.PixelRepresentation = 0
        output_ds.HighBit = 15 if processed_image.dtype == np.uint16 else 7
        output_ds.BitsStored = 16 if processed_image.dtype == np.uint16 else 8
        output_ds.BitsAllocated = 16 if processed_image.dtype == np.uint16 else 8

        # 设置图像尺寸
        output_ds.Rows = processed_image.shape[0]
        output_ds.Columns = processed_image.shape[1]

        # 确保图像数据格式正确
        if processed_image.dtype != original_dicom.pixel_array.dtype:
            processed_image = processed_image.astype(original_dicom.pixel_array.dtype)

        # 设置像素数据
        output_ds.PixelData = processed_image.tobytes()

        # 设置SOP信息
        output_ds.SOPClassUID = output_ds.file_meta.MediaStorageSOPClassUID
        output_ds.SOPInstanceUID = output_ds.file_meta.MediaStorageSOPInstanceUID

        # 生成新的Series和Study Instance UID
        output_ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        if not hasattr(output_ds, 'StudyInstanceUID'):
            output_ds.StudyInstanceUID = pydicom.uid.generate_uid()

        # 添加处理信息
        processing_status = "背景区域设为最大值" if processing_applied else "无背景处理"
        current_time = datetime.now().strftime("%Y%m%d %H:%M:%S")

        output_ds.SeriesDescription = f"Processed - {processing_status}"
        output_ds.SeriesNumber = getattr(original_dicom, 'SeriesNumber', 1) + 1000

        # 添加图像注释
        output_ds.ImageComments = f"Processed on {current_time} - {processing_status}"

        # 添加处理参数
        output_ds.ProcessingFunction = "Background Removal"
        output_ds.ProcessingSoftware = "DICOM Background Processor v1.0"

        # 确保输出路径有.dcm后缀
        pydicom.dcmwrite(output_path, output_ds, write_like_original=False)

        return True

    except Exception as e:
        print(f"使用dcmwrite保存DICOM文件失败: {e}")
        return False


def save_as_png(processed_image, output_path):
    """
    将处理后的图像保存为PNG格式
    """
    try:
        if processed_image.dtype != np.uint8:
            image_normalized = (processed_image - processed_image.min()) / (
                    processed_image.max() - processed_image.min() + 1e-8) * 255
            image_uint8 = image_normalized.astype(np.uint8)
        else:
            image_uint8 = processed_image

        pil_image = Image.fromarray(image_uint8)
        pil_image.save(output_path, 'PNG')
        return True

    except Exception as e:
        print(f"保存PNG文件失败: {e}")
        return False


def process_dicom_background_to_max(input_dir, output_dir, extension_pixels=10, show_comparison=True):
    """
    处理DICOM图像：将背景区域设为最大值，保存PNG和DICOM格式
    """
    # 创建输出目录
    png_dir = os.path.join(output_dir, "PNG")
    dicom_dir = os.path.join(output_dir, "DICOM")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(dicom_dir, exist_ok=True)

    # 获取所有DICOM文件（包括无后缀文件）
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # 筛选DICOM文件：包括.dcm文件和通过pydicom验证的文件
    dicom_files = []
    for filename in all_files:
        file_path = os.path.join(input_dir, filename)
        try:
            # 尝试读取文件，如果是DICOM格式则加入列表
            dicom_data = pydicom.dcmread(file_path)
            dicom_files.append(filename)
        except:
            # 如果不是DICOM格式，跳过
            continue

    if len(dicom_files) == 0:
        print("错误: 输入目录中没有有效的DICOM文件!")
        return

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print(f"PNG输出目录: {png_dir}")
    print(f"DICOM输出目录: {dicom_dir}")
    print(f"背景向内延伸: {extension_pixels} 像素")

    processed_count = 0
    background_processed_count = 0
    no_background_count = 0
    error_count = 0

    for filename in tqdm(dicom_files, desc="DICOM处理"):
        try:
            # 读取DICOM文件
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)
            original_image = dicom_data.pixel_array.astype(np.float32)

            # 归一化到0-1范围以便处理
            image_normalized = (original_image - original_image.min()) / (
                    original_image.max() - original_image.min() + 1e-8)

            # 检测边缘背景
            left_crop, right_crop, background_mask, original_background_mask, has_background = detect_edge_background_advanced(
                image_normalized, extension_pixels=extension_pixels
            )

            # 统计背景区域
            original_bg_pixels = np.sum(original_background_mask)
            extended_bg_pixels = np.sum(background_mask)
            total_pixels = original_image.size

            # 生成输出文件名（保持原文件名，无后缀的会添加.dcm）
            base_name = os.path.splitext(filename)[0]  # 移除任何现有后缀
            processing_applied = False

            # 处理逻辑
            if has_background and extended_bg_pixels > 0:
                processed_image = fill_background_with_max_value(original_image, background_mask)
                processing_applied = True
                background_processed_count += 1

                print(f"\n处理: {filename}")
                print(f"  原始背景区域: {original_bg_pixels} 像素 ({original_bg_pixels / total_pixels * 100:.1f}%)")
                print(f"  延伸后背景区域: {extended_bg_pixels} 像素 ({extended_bg_pixels / total_pixels * 100:.1f}%)")
                print(f"  延伸增加像素: {extended_bg_pixels - original_bg_pixels}")
                print(f"  ✓ 已进行背景处理")

            else:
                processed_image = original_image.copy()
                no_background_count += 1

                print(f"\n处理: {filename}")
                print(f"  未检测到背景区域")
                print(f"  ✓ 保持原始图像")

            # 保存PNG格式
            png_filename = f"{base_name}_processed.png"
            png_path = os.path.join(png_dir, png_filename)
            png_success = save_as_png(processed_image, png_path)

            # 保存DICOM格式 - 使用dcmwrite，自动处理后缀
            dicom_filename = base_name  # 保持原文件名，无后缀
            dicom_path_output = os.path.join(dicom_dir, dicom_filename)
            dicom_success = save_consistent_dicom_with_dcmwrite(
                processed_image, dicom_data, dicom_path_output, processing_applied
            )

            if png_success and dicom_success:
                # 获取实际保存的DICOM文件名（可能添加了.dcm后缀）
                actual_dicom_name = os.path.basename(dicom_path_output)

                print(f"  ✓ 已保存PNG: {png_filename}")
                print(f"  ✓ 已保存DICOM: {actual_dicom_name}")
                processed_count += 1

            # 显示对比
            if show_comparison and processing_applied and background_processed_count <= 3:
                display_processing_comparison(
                    original_image, processed_image,
                    original_background_mask, background_mask,
                    filename, extension_pixels
                )

        except Exception as e:
            print(f"错误处理文件 {filename}: {e}")
            error_count += 1

    # 打印处理总结
    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"{'=' * 60}")
    print(f"总DICOM文件数: {len(dicom_files)}")
    print(f"成功处理: {processed_count}")
    print(f"背景处理文件: {background_processed_count}")
    print(f"无背景文件: {no_background_count}")
    print(f"处理失败: {error_count}")
    print(f"背景延伸像素: {extension_pixels}")
    print(f"PNG文件位置: {png_dir}")
    print(f"DICOM文件位置: {dicom_dir}")


def display_processing_comparison(original, processed, original_mask, extended_mask, filename, extension_pixels):
    """
    显示处理前后的对比
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    im1 = ax1.imshow(original, cmap='gray')
    ax1.set_title(f'原始图像\n{filename}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(processed, cmap='gray')
    ax2.set_title('背景设为最大值')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3.imshow(original, cmap='gray', alpha=0.5)
    original_overlay = np.zeros((*original.shape, 4))
    original_overlay[original_mask] = [1, 0, 0, 0.6]
    ax3.imshow(original_overlay)
    extension_only = extended_mask & ~original_mask
    extension_overlay = np.zeros((*original.shape, 4))
    extension_overlay[extension_only] = [1, 1, 0, 0.8]
    ax3.imshow(extension_overlay)
    ax3.set_title(f'背景区域对比\n红色:原始背景, 黄色:延伸{extension_pixels}像素')
    ax3.axis('off')

    difference = processed - original
    im4 = ax4.imshow(difference, cmap='coolwarm',
                     vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
    ax4.set_title('差异图像')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    plt.tight_layout()
    plt.show()


def main():
    """
    主函数
    """
    input_dir = "D:/ai/train_0"
    output_dir = "D:/ai/background_to_max1"

    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    try:
        extension_pixels = int(input("请输入背景向内延伸的像素数 (默认10): ").strip() or "10")
    except:
        extension_pixels = 10

    process_dicom_background_to_max(input_dir, output_dir,
                                    extension_pixels=extension_pixels,
                                    show_comparison=True)


if __name__ == "__main__":
    main()