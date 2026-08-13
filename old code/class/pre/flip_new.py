import os
import pydicom
import numpy as np
import cv2
import re


def read_dicom_file(filepath):
    """读取无后缀的DICOM文件"""
    try:
        dicom_data = pydicom.dcmread(filepath)
        image_array = dicom_data.pixel_array
        return image_array
    except Exception as e:
        print(f"读取DICOM文件失败 {filepath}: {e}")
        return None


def normalize_image(image):
    """将图像归一化到0-255范围"""
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min + 1e-8) * 255
        image = image.astype(np.uint8)
    return image


def parse_filename(filename):
    """
    解析文件名，提取病例编号和其他部分

    参数:
        filename: 文件名，如 "ANY_001_0" 或 "ANY_001_0.tif"

    返回:
        tuple: (前缀, 病例编号, 后缀, 扩展名)
        如: ("ANY_", "001", "_0", "") 或 ("ANY_", "001", "_0", ".tif")
    """
    # 去掉扩展名
    base_name = filename
    ext = ""
    if '.' in filename:
        base_name, ext = os.path.splitext(filename)

    # 使用正则表达式匹配模式
    pattern = r'^([A-Za-z_]+)_(\d+)_(\d+)$'
    match = re.match(pattern, base_name)

    if match:
        prefix = match.group(1) + "_"  # 如 "ANY_"
        case_num = match.group(2)  # 如 "001"
        suffix = "_" + match.group(3)  # 如 "_0"
        return prefix, case_num, suffix, ext
    else:
        # 尝试其他可能的模式
        parts = base_name.split('_')
        if len(parts) >= 3:
            # 假设最后一部分是数字，倒数第二部分是病例编号
            try:
                prefix = '_'.join(parts[:-2]) + "_" if len(parts) > 2 else ""
                case_num = parts[-2]
                suffix = "_" + parts[-1]
                # 验证病例编号是否为数字
                if case_num.isdigit():
                    return prefix, case_num, suffix, ext
            except:
                pass

    # 如果无法解析，返回原始文件名
    return "", "", base_name, ext


def increment_case_number(case_num, increment=500):
    """
    将病例编号增加指定数值

    参数:
        case_num: 原始病例编号字符串，如 "001"
        increment: 增加的数值，默认为500

    返回:
        新的病例编号字符串，如 "501"
    """
    try:
        # 将字符串转换为整数，增加数值，再转换回字符串
        num = int(case_num)
        new_num = num + increment

        # 保持相同的位数（如果原始有前导零）
        if len(case_num) > 1 and case_num[0] == '0':
            # 如果原始有前导零，保持相同位数
            return str(new_num).zfill(len(case_num))
        else:
            return str(new_num)
    except ValueError:
        # 如果不是数字，返回原始值
        return case_num


def generate_new_filename(original_filename, increment=500):
    """
    根据原始文件名生成新的文件名（病例编号加500）

    参数:
        original_filename: 原始文件名
        increment: 增加的数值，默认为500

    返回:
        新的文件名
    """
    # 解析文件名
    prefix, case_num, suffix, ext = parse_filename(original_filename)

    if case_num:  # 如果成功提取到病例编号
        # 增加病例编号
        new_case_num = increment_case_number(case_num, increment)

        # 构建新文件名
        new_base_name = f"{prefix}{new_case_num}{suffix}"
        new_filename = new_base_name + ext
        return new_filename
    else:
        # 如果无法解析，在原始文件名后添加 "_flipped"
        base_name, ext = os.path.splitext(original_filename)
        return f"{base_name}_flipped{ext}"


def process_images():
    # 输入路径
    dicom_dir = r"D:/med_data/ai/preprocess/aug/5"  # DICOM医学图像目录
    mask_dir = r"D:/med_data/ai/preprocess/aug/5(1)"  # TIF mask图像目录

    # 输出路径
    output_dicom_dir = r"D:/med_data/ai/preprocess/aug/filp5_1"  # 翻转后的DICOM图像
    output_png_dir =  r"D:/med_data/ai/preprocess/aug/filp5_png" # 翻转后的PNG医学图像
    output_mask_dir = r"D:/med_data/ai/preprocess/aug/filp5_2"  # 翻转后的TIF mask图像

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = []
    for f in os.listdir(dicom_dir):
        file_path = os.path.join(dicom_dir, f)
        if os.path.isfile(file_path) and '.' not in f:  # 无后缀文件
            dicom_files.append(f)

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print("=" * 50)

    processed_count = 0
    skipped_count = 0

    for dicom_file in dicom_files:
        try:
            print(f"处理文件: {dicom_file}")

            # 构建文件路径
            dicom_path = os.path.join(dicom_dir, dicom_file)

            # 生成mask文件名（假设mask文件有.tif扩展名）
            mask_filename = dicom_file + ".tif"
            mask_path = os.path.join(mask_dir, mask_filename)

            # 检查mask文件是否存在
            if not os.path.exists(mask_path):
                # 尝试其他可能的mask文件名
                mask_filename2 = dicom_file + ".tiff"
                mask_path2 = os.path.join(mask_dir, mask_filename2)

                if os.path.exists(mask_path2):
                    mask_path = mask_path2
                    mask_filename = mask_filename2
                else:
                    print(f"  警告: 找不到对应的mask文件 {mask_filename} 或 {mask_filename2}")
                    skipped_count += 1
                    continue

            # 读取DICOM文件
            dicom_image = read_dicom_file(dicom_path)
            if dicom_image is None:
                print(f"  错误: 无法读取DICOM文件")
                skipped_count += 1
                continue

            # 读取mask文件
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                print(f"  错误: 无法读取mask文件 {mask_path}")
                skipped_count += 1
                continue

            # 检查图像尺寸是否匹配
            if dicom_image.shape != mask_image.shape:
                print(f"  警告: DICOM和mask图像尺寸不匹配")
                print(f"    DICOM: {dicom_image.shape}, mask: {mask_image.shape}")

                # 尝试调整mask尺寸以匹配DICOM
                if mask_image.shape[0] != dicom_image.shape[0] or mask_image.shape[1] != dicom_image.shape[1]:
                    mask_image = cv2.resize(mask_image, (dicom_image.shape[1], dicom_image.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    print(f"    已调整mask尺寸: {mask_image.shape}")

            # 执行左右翻转
            flipped_dicom = cv2.flip(dicom_image, 1)  # 1表示水平翻转
            flipped_mask = cv2.flip(mask_image, 1)

            # 归一化DICOM图像用于PNG保存
            normalized_dicom = normalize_image(flipped_dicom)

            # 生成新的文件名（病例编号加500）
            new_dicom_name = generate_new_filename(dicom_file, increment=500)
            new_mask_name = generate_new_filename(mask_filename, increment=500)

            # PNG文件名（基于新的DICOM文件名）
            new_png_name = os.path.splitext(new_dicom_name)[0] + ".png"

            print(f"  原始: {dicom_file}")
            print(f"  新DICOM: {new_dicom_name}")
            print(f"  新mask: {new_mask_name}")
            print(f"  新PNG: {new_png_name}")

            # 保存翻转后的DICOM文件（保持原始格式，无后缀）
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)

            try:
                # 读取原始DICOM文件的元数据
                original_dicom = pydicom.dcmread(dicom_path)

                # 更新像素数据
                # 确保数据类型匹配
                if flipped_dicom.dtype != original_dicom.pixel_array.dtype:
                    # 如果需要，转换数据类型
                    if original_dicom.pixel_array.dtype == np.uint16:
                        flipped_dicom = flipped_dicom.astype(np.uint16)
                    elif original_dicom.pixel_array.dtype == np.uint8:
                        flipped_dicom = flipped_dicom.astype(np.uint8)

                original_dicom.PixelData = flipped_dicom.tobytes()
                original_dicom.Rows, original_dicom.Columns = flipped_dicom.shape

                # 更新DICOM文件信息
                if hasattr(original_dicom, 'ImageComments'):
                    original_comment = original_dicom.ImageComments
                    original_dicom.ImageComments = f"Flipped - {original_comment}"

                original_dicom.save_as(output_dicom_path)
                print(f"  ✓ DICOM已保存: {new_dicom_name}")

            except Exception as e:
                print(f"  ✗ 保存DICOM文件失败 {output_dicom_path}: {e}")
                # 如果DICOM保存失败，保存为numpy格式
                np.save(output_dicom_path + ".npy", flipped_dicom)
                print(f"    已保存为numpy格式: {output_dicom_path}.npy")

            # 保存PNG格式的医学图像
            output_png_path = os.path.join(output_png_dir, new_png_name)
            cv2.imwrite(output_png_path, normalized_dicom)
            print(f"  ✓ PNG已保存: {new_png_name}")

            # 保存TIF格式的mask图像
            output_mask_path = os.path.join(output_mask_dir, new_mask_name)
            cv2.imwrite(output_mask_path, flipped_mask)
            print(f"  ✓ mask已保存: {new_mask_name}")

            processed_count += 1
            print(f"  --- 处理完成 ---")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    print("=" * 50)
    print(f"处理完成!")
    print(f"成功处理: {processed_count} 组图像")
    print(f"跳过/失败: {skipped_count} 组图像")
    print(f"总计: {len(dicom_files)} 个DICOM文件")
    print("=" * 50)
    print(f"DICOM文件保存在: {output_dicom_dir}")
    print(f"PNG文件保存在: {output_png_dir}")
    print(f"Mask文件保存在: {output_mask_dir}")

    # 验证文件数量
    dicom_output_files = [f for f in os.listdir(output_dicom_dir) if os.path.isfile(os.path.join(output_dicom_dir, f))]
    png_output_files = [f for f in os.listdir(output_png_dir) if f.endswith('.png')]
    mask_output_files = [f for f in os.listdir(output_mask_dir) if f.endswith(('.tif', '.tiff'))]

    print("=" * 50)
    print("输出文件统计:")
    print(f"  DICOM文件: {len(dicom_output_files)} 个")
    print(f"  PNG文件: {len(png_output_files)} 个")
    print(f"  Mask文件: {len(mask_output_files)} 个")

    # 显示几个示例文件
    if dicom_output_files:
        print("\n示例文件 (前5个):")
        for i, f in enumerate(dicom_output_files[:5]):
            print(f"  {i + 1}. {f}")
            if i + 1 < len(dicom_output_files[:5]):
                # 尝试找到对应的mask和PNG文件
                base_name = os.path.splitext(f)[0]
                png_file = base_name + ".png"
                tif_file = base_name + ".tif"

                if png_file in png_output_files:
                    print(f"      对应PNG: {png_file}")
                if tif_file in mask_output_files:
                    print(f"      对应mask: {tif_file}")


if __name__ == "__main__":
    process_images()