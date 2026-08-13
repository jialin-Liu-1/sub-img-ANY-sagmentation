import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import math
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm


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
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)
    return image


def ensure_numeric_fill_value(image, fill_value=None):
    """
    确保填充值是适合图像数据类型的数值
    """
    if fill_value is None:
        fill_value = image.max()

    # 根据图像数据类型转换填充值
    if image.dtype == np.uint8:
        fill_value = np.uint8(fill_value)
    elif image.dtype == np.uint16:
        fill_value = np.uint16(fill_value)
    elif image.dtype == np.int16:
        fill_value = np.int16(fill_value)
    elif image.dtype == np.float32:
        fill_value = np.float32(fill_value)
    elif image.dtype == np.float64:
        fill_value = np.float64(fill_value)

    return fill_value


def translate_image_vertical(image, shift_pixels, fill_value=None):
    """
    上下平移图像，保持原始图像大小
    shift_pixels: 平移像素数，正数向上平移，负数向下平移
    fill_value: 填充值
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 确保填充值是合适的数据类型
    fill_value = ensure_numeric_fill_value(image, fill_value)

    # 创建平移矩阵 [1, 0, tx; 0, 1, ty]
    # 对于垂直平移，tx=0，ty=shift_pixels
    translation_matrix = np.float32([[1, 0, 0], [0, 1, -shift_pixels]])  # 负号是因为OpenCV的坐标系：正y向下

    # 执行平移，保持原始图像大小，用指定值填充空白区域
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=float(fill_value))

    return translated_image


def get_nonzero_area(image):
    """
    计算图像中非零区域的面积（像素数）
    """
    return np.count_nonzero(image)


def generate_new_filename(original_filename, nubmer):
    """
    根据原始文件名生成新的文件名
    格式：ANY_204_1 -> ANY_100204_1 (中间数字加100000)
          ANY_216_0 -> ANY_100216_0 (中间数字加100000)
    """
    # 使用正则表达式匹配文件名模式
    # 匹配 ANY_数字_数字 的格式
    pattern = r'^(.*?)(\d+)(_\d+)$'
    match = re.match(pattern, original_filename)

    if match:
        prefix = match.group(1)  # "ANY_"
        middle_num = int(match.group(2))  # 204 或 216
        suffix = match.group(3)  # "_1" 或 "_0"

        # 中间数字加100000（确保不与其他序号冲突）
        new_middle_num = middle_num + nubmer

        # 生成新文件名
        new_filename = f"{prefix}{new_middle_num}{suffix}"
        print(f"  文件名转换: {original_filename} -> {new_filename}")
        return new_filename
    else:
        # 如果文件名格式不匹配，在原文件名后添加"_new"作为后备方案
        print(f"  警告: 文件名格式不匹配 {original_filename}，使用后备命名方案")
        return original_filename + "_new"


def extract_case_number(filename):
    """
    从文件名中提取病例序号
    例如: "ANY_001_0" -> "ANY_001"
          "ANY_204_1" -> "ANY_204"
    """
    pattern = r'^(.*?)(\d+)(_\d+)$'
    match = re.match(pattern, filename)
    if match:
        prefix = match.group(1)  # "ANY_"
        middle_num = match.group(2)  # "001"
        return f"{prefix}{middle_num}"
    else:
        # 尝试直接匹配数字部分
        num_match = re.search(r'(\d+)', filename)
        if num_match:
            return f"ANY_{num_match.group(1)}"
        else:
            return filename


def load_classification_table(excel_path):
    """
    加载分类表格
    """
    try:
        df = pd.read_excel(excel_path)
        print(f"成功加载分类表格: {excel_path}")
        print(f"表格形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"加载分类表格失败: {e}")
        # 如果读取失败，创建一个空的DataFrame
        return pd.DataFrame(columns=['filename', 'classification'])


def process_images():
    # 输入路径
    dicom_dir = r"D:\med_data\ai\train11"  # DICOM医学图像目录
    tif_dir = r"D:\med_data\ai\train22"  # TIF图像目录
    excel_path = r"D:\med_data\ai\classify1.xlsx"  # 分类表格路径

    # 输出路径
    output_dicom_dir = r"D:\med_data\ai\translate\train1(-15)"  # 平移后的DICOM文件
    output_png_dir = r"D:\med_data\ai\translate\train1PNG(-15)"  # 平移后的PNG文件
    output_tif_dir = r"D:\med_data\ai\translate\train2(-15)"  # 平移后的TIF文件
    output_excel_path = r"D:\med_data\ai\translate\classify(-15).xlsx"  # 新的分类表格

    # 平移参数
    # 所有图像尺寸为[512, 512]，平移15%为77像素
    shift_percentage = 15  # 平移比例（%）
    shift_pixels = int(512 * shift_percentage / 100)  # 计算平移像素数

    # 设置平移方向：正数向上平移，负数向下平移
    # 默认向上平移77像素
    shift_direction = -1  # 1表示向上，-1表示向下
    number = 20000

    # 计算最终的平移像素数
    final_shift_pixels = shift_direction * shift_pixels

    # 创建输出目录
    for dir_path in [output_dicom_dir, output_png_dir, output_tif_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 加载分类表格
    classification_df = load_classification_table(excel_path)

    # 创建病例序号到分类的映射
    # 假设表格第一列是"filename"，包含"ANY_001"这样的病例序号
    if 'filename' in classification_df.columns and len(classification_df.columns) > 1:
        # 第二列是分类信息
        class_col = classification_df.columns[1]
        case_to_class = dict(zip(classification_df['filename'], classification_df[class_col]))
        print(f"成功创建病例分类映射，共 {len(case_to_class)} 个病例")
    else:
        print("警告: 表格格式不符合预期，将创建空映射")
        case_to_class = {}
        # 如果表格格式不对，尝试自动识别
        if len(classification_df.columns) >= 2:
            first_col = classification_df.columns[0]
            second_col = classification_df.columns[1]
            case_to_class = dict(zip(classification_df[first_col], classification_df[second_col]))
            print(f"使用列 '{first_col}' 作为病例号，'{second_col}' 作为分类信息")

    # 获取DICOM文件列表（无后缀文件）
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f)) and not f.endswith('.tif')]

    print(f"找到 {len(dicom_files)} 个DICOM文件")
    print(f"图像尺寸: 512×512")
    print(f"平移比例: {shift_percentage}% ({shift_pixels}像素)")
    print(f"平移方向: {'向上' if final_shift_pixels > 0 else '向下'} ({abs(final_shift_pixels)}像素)")
    print(f"DICOM空白填充: 图像最大值")
    print(f"TIF空白填充: 0")
    print(f"文件命名方式: 原文件名中间数字 + 100000")
    print(f"将检查TIF非零区域面积是否变化\n")

    processed_count = 0
    skipped_count = 0

    # 用于记录新的分类信息
    new_classification_records = []

    for dicom_file in tqdm(dicom_files, desc="处理图像"):
        try:
            # 构建文件路径
            dicom_path = os.path.join(dicom_dir, dicom_file)
            tif_path = os.path.join(tif_dir, dicom_file + ".tif")

            # 检查TIF文件是否存在
            if not os.path.exists(tif_path):
                print(f"警告: 找不到对应的TIF文件 {tif_path}")
                continue

            # 读取DICOM文件
            dicom_image = read_dicom_file(dicom_path)
            if dicom_image is None:
                continue

            # 读取TIF文件
            tif_image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
            if tif_image is None:
                print(f"读取TIF文件失败: {tif_path}")
                continue

            # 计算原始TIF的非零区域面积
            original_nonzero_area = get_nonzero_area(tif_image)

            # 打印图像数据类型信息（仅在调试时取消注释）
            # print(f"处理 {dicom_file}:")
            # print(f"  DICOM - 尺寸: {dicom_image.shape}, 数据类型: {dicom_image.dtype}, 最大值: {dicom_image.max()}")
            # print(f"  TIF - 尺寸: {tif_image.shape}, 数据类型: {tif_image.dtype}, 非零区域面积: {original_nonzero_area}")

            # 计算DICOM图像的最大值（用于填充空白区域）
            dicom_max_value = float(dicom_image.max())
            # TIF图像使用0填充
            tif_fill_value = 0.0

            # 执行平移操作
            translated_dicom = translate_image_vertical(dicom_image, final_shift_pixels, dicom_max_value)
            translated_tif = translate_image_vertical(tif_image, final_shift_pixels, tif_fill_value)

            # 计算平移后TIF的非零区域面积
            translated_nonzero_area = get_nonzero_area(translated_tif)

            # 检查非零区域面积是否改变
            if original_nonzero_area != translated_nonzero_area:
                print(f"警告: TIF非零区域面积改变!")
                print(f"  文件: {dicom_file}")
                print(f"  原始面积: {original_nonzero_area}")
                print(f"  平移后面积: {translated_nonzero_area}")
                print(f"  跳过此组图像\n")
                skipped_count += 1
                continue  # 跳过这组图像

            # 检查平移后的图像尺寸是否与原始相同
            if translated_dicom.shape != dicom_image.shape:
                print(f"警告: DICOM图像尺寸改变 {dicom_image.shape} -> {translated_dicom.shape}")
            if translated_tif.shape != tif_image.shape:
                print(f"警告: TIF图像尺寸改变 {tif_image.shape} -> {translated_tif.shape}")

            # 归一化DICOM图像用于PNG保存
            normalized_dicom = normalize_image(translated_dicom.copy())

            # 生成新的文件名（中间数字加100000）
            base_name = dicom_file  # 原文件名无后缀
            new_base_name = generate_new_filename(base_name, number)

            # 为不同格式生成文件名
            new_dicom_name = new_base_name  # 无后缀DICOM
            new_png_name = new_base_name + ".png"  # PNG格式
            new_tif_name = new_base_name + ".tif"  # TIF格式

            # 保存平移后的DICOM文件（保持原始格式）
            output_dicom_path = os.path.join(output_dicom_dir, new_dicom_name)

            # 对于DICOM文件，使用pydicom保存
            try:
                original_dicom = pydicom.dcmread(dicom_path)
                # 更新图像数据
                original_dicom.PixelData = translated_dicom.tobytes()
                original_dicom.Rows, original_dicom.Columns = translated_dicom.shape
                # 保存为无后缀文件
                original_dicom.save_as(output_dicom_path)
                # print(f"  已保存DICOM: {output_dicom_path}")
            except Exception as e:
                print(f"  保存DICOM文件失败 {output_dicom_path}: {e}")
                # 如果DICOM保存失败，保存为RAW格式
                translated_dicom.tofile(output_dicom_path + ".raw")
                print(f"  已保存为RAW格式: {output_dicom_path}.raw")

            # 保存PNG格式的医学图像
            output_png_path = os.path.join(output_png_dir, new_png_name)
            Image.fromarray(normalized_dicom).save(output_png_path, 'PNG')
            # print(f"  已保存PNG: {output_png_path}")

            # 保存TIF格式的图像
            output_tif_path = os.path.join(output_tif_dir, new_tif_name)
            cv2.imwrite(output_tif_path, translated_tif)
            # print(f"  已保存TIF: {output_tif_path}")

            # ========== 处理分类表格信息 ==========
            # 从原始文件名提取病例序号（如 "ANY_001"）
            case_number = extract_case_number(dicom_file)

            # 从映射中获取分类信息
            classification = case_to_class.get(case_number, None)

            if classification is not None:
                # 从新文件名提取新的病例序号
                new_case_number = extract_case_number(new_base_name)

                # 记录新的分类信息
                new_classification_records.append({
                    'filename': new_case_number,  # 新病例序号
                    'classification': classification  # 原分类信息
                })
                # print(f"  分类信息: {case_number} -> {new_case_number}, 分类: {classification}")
            else:
                print(f"  警告: 未找到病例 {case_number} 的分类信息")

            processed_count += 1
            # print(f"  完成处理: {dicom_file} -> {new_base_name} (平移{final_shift_pixels}像素)\n")

        except Exception as e:
            print(f"处理文件 {dicom_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========== 保存新的分类表格 ==========
    if new_classification_records:
        # 创建新的DataFrame
        new_df = pd.DataFrame(new_classification_records)

        # 去除重复记录（如果有的话）
        new_df = new_df.drop_duplicates(subset=['filename'])

        # 按文件名排序
        new_df = new_df.sort_values(by='filename')

        # 保存为Excel文件
        try:
            new_df.to_excel(output_excel_path, index=False)
            print(f"\n成功保存新的分类表格: {output_excel_path}")
            print(f"表格包含 {len(new_df)} 个病例记录")
            print(f"前5条记录:")
            print(new_df.head())
        except Exception as e:
            print(f"保存分类表格失败: {e}")
            # 如果保存Excel失败，保存为CSV
            csv_path = output_excel_path.replace('.xlsx', '.csv')
            new_df.to_csv(csv_path, index=False)
            print(f"已保存为CSV格式: {csv_path}")
    else:
        print("\n警告: 没有生成任何分类记录！")
        # 创建一个空的表格
        empty_df = pd.DataFrame(columns=['filename', 'classification'])
        empty_df.to_excel(output_excel_path, index=False)
        print(f"已创建空表格: {output_excel_path}")

    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 组图像")
    print(f"跳过 (非零区域面积改变): {skipped_count} 组图像")
    print(f"平移后的DICOM文件保存在: {output_dicom_dir}")
    print(f"平移后的PNG文件保存在: {output_png_dir}")
    print(f"平移后的TIF文件保存在: {output_tif_dir}")
    print(f"新的分类表格保存在: {output_excel_path}")


if __name__ == "__main__":
    process_images()