import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def adjust_contrast(dicom_data, contrast_percent):
    """
    调整DICOM图像的对比度

    Parameters:
    dicom_data: DICOM图像的像素数据
    contrast_percent: 对比度调整百分比（正数增加对比度，负数减少对比度）

    Returns:
    调整后的像素数据
    """
    # 将百分比转换为系数
    factor = 1 + contrast_percent / 100.0

    # 计算均值和标准差
    mean = np.mean(dicom_data)
    std = np.std(dicom_data)

    # 调整对比度
    adjusted_data = mean + (dicom_data - mean) * factor

    # 确保数据类型正确
    adjusted_data = np.clip(adjusted_data, np.min(dicom_data), np.max(dicom_data))

    # 转换为原始数据类型
    adjusted_data = adjusted_data.astype(dicom_data.dtype)

    return adjusted_data


def parse_case_number(filename):
    """
    从文件名解析病例号（不包含图像序号）
    例如: "ANY_002_0" -> "ANY_002"
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:-1])
    return filename


def get_image_number(filename):
    """
    从文件名获取图像序号
    例如: "ANY_002_0" -> "0"
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[-1]
    return ''


def get_full_image_id(filename):
    """
    获取完整的图像ID（包含图像序号）
    例如: "ANY_002_0" -> "ANY_002_0"
    """
    return filename


def process_dicom_images(
        input_dir=r"D:\med_data\ai\translate\train_all_trans(1)",
        excel_path=r"D:\med_data\ai\translate\classify_all_trans.xlsx",
        location_excel_path=r"D:\med_data\ai\translate\location_trans.xlsx",
        output_dir=r"D:\med_data\ai\translate\contrast",
        png_output_dir=r"D:\med_data\ai\translate\contrastPNG",
        contrast_percent=15,
        case_number_increment=40000,
        batch_size=100
):
    """
    处理DICOM图像的主函数

    Parameters:
    input_dir: 输入DICOM图像的目录
    excel_path: 病例信息Excel文件路径
    location_excel_path: 位置信息Excel文件路径（按图像编号）
    output_dir: 输出DICOM图像的目录
    png_output_dir: 输出PNG图像的目录
    contrast_percent: 对比度调整百分比（可正可负）
    case_number_increment: 病例号增加的数字
    batch_size: 批处理大小
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(png_output_dir, exist_ok=True)

    # 读取病例分类表格（按病例号）
    print("正在读取病例分类Excel表格...")
    df_classify = pd.read_excel(excel_path, header=None, names=['case_number', 'disease_code'])
    print(f"读取到 {len(df_classify)} 个病例分类记录")

    # 创建病例号到病名编号的映射
    case_to_disease = dict(zip(df_classify['case_number'], df_classify['disease_code']))

    # 读取位置信息表格（按图像编号）
    print("正在读取位置信息Excel表格（按图像编号）...")
    try:
        df_location = pd.read_excel(location_excel_path, header=None,
                                    names=['image_id', 'height_ratio', 'radius_ratio'])
        print(f"读取到 {len(df_location)} 个位置信息记录（每个对应一个图像）")

        # 创建图像ID到位置信息的映射
        image_to_location = {}
        for _, row in df_location.iterrows():
            image_id = row['image_id']
            image_to_location[image_id] = {
                'height_ratio': row['height_ratio'],
                'radius_ratio': row['radius_ratio']
            }
    except FileNotFoundError:
        print(f"警告: 位置信息文件 {location_excel_path} 不存在，将只生成病例分类表格")
        image_to_location = {}
    except Exception as e:
        print(f"警告: 读取位置信息文件时出错: {e}")
        image_to_location = {}

    # 获取所有DICOM文件
    dicom_files = glob.glob(os.path.join(input_dir, "*"))
    print(f"找到 {len(dicom_files)} 个DICOM文件")

    # 创建新的病例号映射
    new_case_numbers = {}

    # 批处理
    new_classify_records = []  # 用于分类表格的记录（病例级别）
    new_location_records = []  # 用于位置表格的记录（图像级别）
    processed_images = set()  # 记录已处理的图像
    processed_cases = set()  # 记录已处理的病例

    for i in tqdm(range(0, len(dicom_files), batch_size), desc="处理批次"):
        batch_files = dicom_files[i:i + batch_size]

        for file_path in batch_files:
            try:
                # 获取文件名（不含路径）
                filename = os.path.basename(file_path)
                full_image_id = filename  # 完整的图像ID，如 "ANY_002_0"

                # 解析病例号和图像序号
                old_case_number = parse_case_number(filename)
                image_number = get_image_number(filename)

                # 检查病例号是否在分类表格中
                if old_case_number not in case_to_disease:
                    print(f"警告: 病例号 {old_case_number} 不在分类Excel表格中，跳过文件 {filename}")
                    continue

                # 生成新的病例号
                old_case_num = old_case_number.split('_')[-1]
                new_case_num = str(int(old_case_num) + case_number_increment)
                new_case_number = f"ANY_{new_case_num}"

                # 生成新的完整图像ID
                new_full_image_id = f"{new_case_number}_{image_number}"

                # 记录新的病例号映射
                new_case_numbers[old_case_number] = new_case_number

                # 生成新文件名
                new_file_path = os.path.join(output_dir, new_full_image_id)

                # 读取DICOM文件
                dicom_data = pydicom.dcmread(file_path, force=True)

                # 获取像素数据
                pixel_array = dicom_data.pixel_array

                # 调整对比度
                adjusted_pixel_array = adjust_contrast(pixel_array, contrast_percent)

                # 更新DICOM数据
                dicom_data.PixelData = adjusted_pixel_array.tobytes()
                dicom_data.Rows, dicom_data.Columns = adjusted_pixel_array.shape

                # 保存调整后的DICOM文件
                dicom_data.save_as(new_file_path)

                # 生成并保存PNG文件
                png_filename = f"{new_full_image_id}.png"
                png_path = os.path.join(png_output_dir, png_filename)

                # 将DICOM数据归一化到0-255范围并保存为PNG
                if adjusted_pixel_array.max() > adjusted_pixel_array.min():
                    normalized_array = ((adjusted_pixel_array - adjusted_pixel_array.min()) /
                                        (adjusted_pixel_array.max() - adjusted_pixel_array.min()) * 255)
                else:
                    normalized_array = np.zeros_like(adjusted_pixel_array)

                normalized_array = normalized_array.astype(np.uint8)

                # 处理不同维度的图像
                if len(normalized_array.shape) == 3:
                    if normalized_array.shape[2] == 3:
                        img = Image.fromarray(normalized_array, 'RGB')
                    elif normalized_array.shape[2] == 1:
                        img = Image.fromarray(normalized_array[:, :, 0], 'L')
                    else:
                        img = Image.fromarray(normalized_array[:, :, 0], 'L')
                else:
                    img = Image.fromarray(normalized_array, 'L')

                img.save(png_path)

                # 记录新分类表格数据（病例级别，每个病例只记录一次）
                if old_case_number not in processed_cases:
                    disease_code = case_to_disease[old_case_number]
                    new_classify_records.append([new_case_number, disease_code])
                    processed_cases.add(old_case_number)

                # 记录新位置表格数据（图像级别，每个图像都有记录）
                if full_image_id in image_to_location:
                    location_info = image_to_location[full_image_id]
                    new_location_records.append([
                        new_full_image_id,  # 使用新的完整图像ID
                        location_info['height_ratio'],
                        location_info['radius_ratio']
                    ])
                    processed_images.add(full_image_id)
                else:
                    # 如果位置表格中没有这个图像的信息，可以选择记录或跳过
                    # 这里选择记录空值，以便保留所有图像的位置信息
                    new_location_records.append([
                        new_full_image_id,
                        None,  # 高度比例设为空
                        None  # 半径比例设为空
                    ])

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue

    # 创建新分类表格（病例级别）
    if new_classify_records:
        new_classify_df = pd.DataFrame(new_classify_records, columns=['case_number', 'disease_code'])

        # 保存新分类表格
        new_classify_excel_path = os.path.join(output_dir, "classify_all_trans_updated.xlsx")
        new_classify_df.to_excel(new_classify_excel_path, index=False, header=False)

        # 也保存一份CSV格式便于查看
        new_classify_csv_path = os.path.join(output_dir, "classify_all_trans_updated.csv")
        new_classify_df.to_csv(new_classify_csv_path, index=False, header=False)

        print(f"\n分类表格处理完成! 保存了 {len(new_classify_df)} 个唯一病例记录")

    # 创建新位置表格（图像级别）
    if new_location_records:
        new_location_df = pd.DataFrame(new_location_records, columns=['image_id', 'height_ratio', 'radius_ratio'])

        # 保存新位置表格
        new_location_excel_path = os.path.join(output_dir, "location_trans_updated.xlsx")
        new_location_df.to_excel(new_location_excel_path, index=False, header=False)

        # 也保存一份CSV格式便于查看
        new_location_csv_path = os.path.join(output_dir, "location_trans_updated.csv")
        new_location_df.to_csv(new_location_csv_path, index=False, header=False)

        print(f"位置表格处理完成! 保存了 {len(new_location_df)} 个图像记录")
    else:
        print("警告: 没有生成任何位置表格记录")
        new_location_excel_path = None
        new_location_csv_path = None

    print(f"\n处理完成!")
    print(f"处理了 {len(dicom_files)} 个DICOM文件")
    print(f"生成了 {len(new_classify_records)} 个病例分类记录")
    print(f"生成了 {len(new_location_records)} 个图像位置记录")
    print(f"新DICOM文件保存在: {output_dir}")
    print(f"新PNG文件保存在: {png_output_dir}")

    # 返回结果
    return {
        'total_images': len(dicom_files),
        'processed_images': len(new_location_records),
        'unique_cases': len(new_classify_records),
        'location_records': len(new_location_records),
        'output_dir': output_dir,
        'png_output_dir': png_output_dir,
        'classify_excel': new_classify_excel_path if new_classify_records else None,
        'location_excel': new_location_excel_path
    }


def main():
    """
    主函数，可以在这里调整各种参数
    """

    # 可调整的参数
    params = {
        # 输入目录
        'input_dir': r"D:\med_data\ai\translate\train_all_trans(1)",

        # 病例分类Excel文件路径（病例级别）
        'excel_path': r"D:\med_data\ai\translate\classify_all_trans.xlsx",

        # 位置信息Excel文件路径（图像级别）- 现在每个图像对应一行
        'location_excel_path': r"D:\med_data\ai\translate\location_trans.xlsx",

        # 输出目录
        'output_dir': r"D:\med_data\ai\translate\contrast20",

        # PNG输出目录
        'png_output_dir': r"D:\med_data\ai\translate\contrastPNG20",

        # 对比度调整百分比（正数增加对比度，负数减少对比度）
        'contrast_percent': 20,  # 例如：15%增加对比度，-20%减少对比度

        # 病例号增加的数字
        'case_number_increment': 60000,

        # 批处理大小
        'batch_size': 50
    }

    print("=" * 60)
    print("DICOM图像对比度调整和表格处理程序")
    print("=" * 60)
    print("参数设置:")
    print(f"  输入目录: {params['input_dir']}")
    print(f"  分类表格（病例级）: {params['excel_path']}")
    print(f"  位置表格（图像级）: {params['location_excel_path']}")
    print(f"  输出目录: {params['output_dir']}")
    print(f"  PNG输出目录: {params['png_output_dir']}")
    print(f"  对比度调整: {params['contrast_percent']}%")
    print(f"  病例号增量: {params['case_number_increment']}")
    print(f"  批处理大小: {params['batch_size']}")
    print("-" * 60)

    # 执行处理
    results = process_dicom_images(**params)

    # 显示结果摘要
    print("\n" + "=" * 60)
    print("处理摘要:")
    print(f"  总图像数: {results['total_images']}")
    print(f"  处理的图像数: {results['processed_images']}")
    print(f"  处理的唯一病例数: {results['unique_cases']}")
    print(f"  位置信息记录数: {results['location_records']}")
    print(f"  分类表格输出: {results['classify_excel']}")
    print(f"  位置表格输出: {results['location_excel']}")
    print("=" * 60)


if __name__ == "__main__":
    # 安装必要的库
    # pip install pydicom pandas numpy pillow openpyxl tqdm

    main()