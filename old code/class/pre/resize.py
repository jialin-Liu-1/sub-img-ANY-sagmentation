import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm


def is_dicom_file(filepath):
    """
    检查文件是否为DICOM格式
    """
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            signature = f.read(4)
            return signature == b'DICM'
    except:
        return False


def read_dicom_file(filepath):
    """
    读取DICOM文件，返回DICOM数据和成功状态
    """
    try:
        dicom_data = pydicom.dcmread(filepath)
        return dicom_data, True
    except Exception as e:
        print(f"读取DICOM文件 {filepath} 失败: {e}")
        return None, False


def resize_dicom_to_512x512(input_path, output_path):
    """
    将DICOM文件调整为512×512大小并保持DICOM格式

    Args:
        input_path: 输入DICOM文件路径
        output_path: 输出DICOM文件路径
    """
    try:
        # 读取DICOM文件
        dicom_data, success = read_dicom_file(input_path)
        if not success:
            return False

        # 获取原始图像数据
        original_image = dicom_data.pixel_array.astype(np.float32)

        # 转换为PIL图像进行调整
        image_normalized = (original_image - original_image.min()) / (
                    original_image.max() - original_image.min() + 1e-8)
        image_uint8 = (image_normalized * 255).astype(np.uint8)

        pil_image = Image.fromarray(image_uint8)
        resized_pil = pil_image.resize((512, 512), Image.BILINEAR)
        resized_array = np.array(resized_pil)

        # 恢复原始像素值范围
        resized_float = resized_array.astype(np.float32) / 255.0
        resized_original_range = resized_float * (original_image.max() - original_image.min()) + original_image.min()

        # 更新DICOM数据
        # 确保数据类型匹配
        if dicom_data.BitsAllocated == 16:
            resized_original_range = resized_original_range.astype(np.uint16)
        else:
            resized_original_range = resized_original_range.astype(np.uint8)

        dicom_data.PixelData = resized_original_range.tobytes()
        dicom_data.Rows = 512
        dicom_data.Columns = 512

        # 更新相关的DICOM标签
        if hasattr(dicom_data, 'PixelSpacing'):
            # 调整像素间距（如果需要）
            original_height, original_width = original_image.shape
            scale_x = original_width / 512.0
            scale_y = original_height / 512.0
            if isinstance(dicom_data.PixelSpacing, list) and len(dicom_data.PixelSpacing) == 2:
                dicom_data.PixelSpacing = [
                    float(dicom_data.PixelSpacing[0]) * scale_x,
                    float(dicom_data.PixelSpacing[1]) * scale_y
                ]

        # 保存调整后的DICOM文件
        dicom_data.save_as(output_path)
        return True

    except Exception as e:
        print(f"处理DICOM文件 {input_path} 时出错: {e}")
        return False


def resize_mask_to_512x512(input_path, output_path):
    """
    将掩码文件调整为512×512大小

    Args:
        input_path: 输入掩码文件路径
        output_path: 输出掩码文件路径
    """
    try:
        # 读取TIF文件
        pil_image = Image.open(input_path)

        # 调整尺寸 - 掩码使用最近邻插值，避免模糊
        resized_image = pil_image.resize((512, 512), Image.NEAREST)

        # 保存调整后的掩码文件
        resized_image.save(output_path, 'TIFF')
        return True

    except Exception as e:
        print(f"处理掩码文件 {input_path} 时出错: {e}")
        return False


def find_matching_files(dicom_dir, mask_dir):
    """
    查找匹配的DICOM和掩码文件对

    Args:
        dicom_dir: DICOM文件目录
        mask_dir: 掩码文件目录

    Returns:
        list: 匹配的文件对列表 [(dicom_file, mask_file), ...]
    """
    samples = []

    # 检查目录是否存在
    if not os.path.exists(dicom_dir):
        print(f"错误: DICOM目录不存在: {dicom_dir}")
        return samples

    if not os.path.exists(mask_dir):
        print(f"错误: 掩码目录不存在: {mask_dir}")
        return samples

    # 获取所有文件（不限制扩展名，先找到所有文件）
    dicom_files = [f for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

    print(f"DICOM目录中有 {len(dicom_files)} 个文件")
    print(f"掩码目录中有 {len(mask_files)} 个文件")

    # 创建文件名到完整路径的映射（不含扩展名）
    dicom_base_names = {}
    for dicom_file in dicom_files:
        base_name = os.path.splitext(dicom_file)[0]  # 移除扩展名
        dicom_base_names[base_name] = dicom_file

    mask_base_names = {}
    for mask_file in mask_files:
        base_name = os.path.splitext(mask_file)[0]  # 移除扩展名
        mask_base_names[base_name] = mask_file

    # 查找匹配的文件对
    matched_base_names = set(dicom_base_names.keys()) & set(mask_base_names.keys())

    for base_name in matched_base_names:
        dicom_file = dicom_base_names[base_name]
        mask_file = mask_base_names[base_name]
        samples.append((dicom_file, mask_file))

    print(f"找到 {len(matched_base_names)} 个匹配的文件对")

    # 显示前几个匹配的文件对
    if samples:
        print("前10个匹配的文件对:")
        for i, (dicom, mask) in enumerate(samples[:10]):
            print(f"  {i + 1}. DICOM: {dicom} -> Mask: {mask}")

    return samples


def check_dicom_files(dicom_dir, file_pairs):
    """
    检查DICOM文件是否可读
    """
    valid_pairs = []
    invalid_files = []

    for dicom_file, mask_file in file_pairs:
        dicom_path = os.path.join(dicom_dir, dicom_file)

        # 检查是否为DICOM文件
        if not is_dicom_file(dicom_path):
            print(f"警告: 文件不是DICOM格式: {dicom_file}")
            invalid_files.append(dicom_file)
            continue

        # 尝试读取DICOM文件
        dicom_data, success = read_dicom_file(dicom_path)
        if success:
            valid_pairs.append((dicom_file, mask_file))
            print(f"验证成功: {dicom_file} -> 尺寸: {dicom_data.pixel_array.shape}")
        else:
            invalid_files.append(dicom_file)

    print(f"DICOM文件验证结果: {len(valid_pairs)} 个有效, {len(invalid_files)} 个无效")
    return valid_pairs


def process_image_pairs(dicom_dir, mask_dir, output_dicom_dir, output_mask_dir):
    """
    处理DICOM和掩码图像对，统一调整为512×512

    Args:
        dicom_dir: 原始DICOM文件目录
        mask_dir: 原始掩码文件目录
        output_dicom_dir: 输出DICOM文件目录
        output_mask_dir: 输出掩码文件目录
    """
    # 创建输出目录
    os.makedirs(output_dicom_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 查找匹配的文件对
    file_pairs = find_matching_files(dicom_dir, mask_dir)

    if not file_pairs:
        print("错误: 没有找到匹配的文件对!")
        return

    # 验证DICOM文件
    valid_pairs = check_dicom_files(dicom_dir, file_pairs)

    if not valid_pairs:
        print("错误: 没有有效的DICOM文件!")
        return

    # 处理每个匹配的文件对
    success_count = 0
    for dicom_file, mask_file in tqdm(valid_pairs, desc="处理图像对"):
        # 输入文件路径
        dicom_input_path = os.path.join(dicom_dir, dicom_file)
        mask_input_path = os.path.join(mask_dir, mask_file)

        # 输出文件路径
        # DICOM文件保持原格式，掩码文件保持TIF格式
        dicom_output_path = os.path.join(output_dicom_dir, dicom_file)
        mask_output_path = os.path.join(output_mask_dir, mask_file)

        # 处理DICOM文件（保持DICOM格式）
        dicom_success = resize_dicom_to_512x512(dicom_input_path, dicom_output_path)

        # 处理掩码文件
        mask_success = resize_mask_to_512x512(mask_input_path, mask_output_path)

        if dicom_success and mask_success:
            success_count += 1
        else:
            print(f"处理失败: {dicom_file} -> {mask_file}")

    print(f"成功处理 {success_count}/{len(valid_pairs)} 个文件对")

    # 统计输出文件信息
    output_dicom_files = [f for f in os.listdir(output_dicom_dir) if is_dicom_file(os.path.join(output_dicom_dir, f))]
    output_mask_files = [f for f in os.listdir(output_mask_dir) if f.lower().endswith(('.tif', '.tiff'))]

    print(f"输出目录统计:")
    print(f"  - DICOM文件: {len(output_dicom_files)} 个DICOM文件")
    print(f"  - 掩码文件: {len(output_mask_files)} 个TIF文件")

    # 检查输出文件尺寸
    if output_dicom_files:
        sample_file = os.path.join(output_dicom_dir, output_dicom_files[0])
        dicom_data, success = read_dicom_file(sample_file)
        if success:
            print(f"  - 输出DICOM尺寸: {dicom_data.pixel_array.shape}")

    if output_mask_files:
        sample_file = os.path.join(output_mask_dir, output_mask_files[0])
        with Image.open(sample_file) as img:
            print(f"  - 输出掩码尺寸: {img.size}")


def check_original_sizes(dicom_dir, mask_dir, sample_count=5):
    """
    检查原始文件的尺寸
    """
    print("\n检查原始文件尺寸:")

    # 检查DICOM文件尺寸
    dicom_files = [f for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    print(f"DICOM文件尺寸样本:")
    for i, filename in enumerate(dicom_files[:sample_count]):
        filepath = os.path.join(dicom_dir, filename)
        if is_dicom_file(filepath):
            dicom_data, success = read_dicom_file(filepath)
            if success:
                print(f"  {filename}: {dicom_data.pixel_array.shape}")

    # 检查掩码文件尺寸
    mask_files = [f for f in os.listdir(mask_dir) if
                  os.path.isfile(os.path.join(mask_dir, f)) and
                  f.lower().endswith(('.tif', '.tiff'))]
    print(f"掩码文件尺寸样本:")
    for i, filename in enumerate(mask_files[:sample_count]):
        filepath = os.path.join(mask_dir, filename)
        try:
            with Image.open(filepath) as img:
                print(f"  {filename}: {img.size}")
        except Exception as e:
            print(f"  {filename}: 读取失败 - {e}")


def main():
    # 设置路径
    dicom_dir = "D:/med_data/ai/data"  # 原始DICOM文件目录
    mask_dir = "D:/med_data/ai/mask"  # 原始掩码文件目录
    output_dicom_dir = "D:/med_data/ai/data1"  # 输出DICOM文件目录
    output_mask_dir = "D:/med_data/ai/mask1"  # 输出掩码文件目录

    print("开始处理图像尺寸统一...")
    print(f"输入目录: {dicom_dir}, {mask_dir}")
    print(f"输出目录: {output_dicom_dir}, {output_mask_dir}")

    # 检查原始文件尺寸
    check_original_sizes(dicom_dir, mask_dir)

    print("\n开始处理图像对...")
    process_image_pairs(dicom_dir, mask_dir, output_dicom_dir, output_mask_dir)

    print("\n处理完成!")


if __name__ == "__main__":
    main()