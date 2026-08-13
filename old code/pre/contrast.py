import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import glob
import shutil
from tqdm import tqdm


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
    从文件名解析病例号
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


def process_dicom_images(
        input_dir=r"D:\med_data\ai\translate\train_all_trans(1)",
        excel_path=r"D:\med_data\ai\translate\classify_all_trans.xlsx",
        output_dir=r"D:\med_data\ai\translate\contrast1",
        png_output_dir=r"D:\med_data\ai\translate\contrastPNG",
        contrast_percent=15,
        case_number_increment=60000,
        batch_size=100
):
    """
    处理DICOM图像的主函数

    Parameters:
    input_dir: 输入DICOM图像的目录
    excel_path: 病例信息Excel文件路径
    output_dir: 输出DICOM图像的目录
    png_output_dir: 输出PNG图像的目录
    contrast_percent: 对比度调整百分比（可正可负）
    case_number_increment: 病例号增加的数字
    batch_size: 批处理大小
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(png_output_dir, exist_ok=True)

    # 读取Excel表格
    print("正在读取Excel表格...")
    df = pd.read_excel(excel_path, header=None, names=['case_number', 'disease_code'])
    print(f"读取到 {len(df)} 个病例记录")

    # 创建病例号到病名编号的映射
    case_to_disease = dict(zip(df['case_number'], df['disease_code']))

    # 获取所有DICOM文件
    dicom_files = glob.glob(os.path.join(input_dir, "*"))
    print(f"找到 {len(dicom_files)} 个DICOM文件")

    # 创建新的病例号映射
    new_case_numbers = {}

    # 批处理
    new_records = []
    processed_cases = set()

    for i in tqdm(range(0, len(dicom_files), batch_size), desc="处理批次"):
        batch_files = dicom_files[i:i + batch_size]

        for file_path in batch_files:
            try:
                # 获取文件名
                filename = os.path.basename(file_path)

                # 解析病例号和图像序号
                old_case_number = parse_case_number(filename)
                image_number = get_image_number(filename)

                # 检查病例号是否在表格中
                if old_case_number not in case_to_disease:
                    print(f"警告: 病例号 {old_case_number} 不在Excel表格中，跳过文件 {filename}")
                    continue

                # 生成新的病例号
                old_case_num = old_case_number.split('_')[-1]
                new_case_num = str(int(old_case_num) + case_number_increment)
                new_case_number = f"ANY_{new_case_num}"

                # 记录新的病例号映射
                new_case_numbers[old_case_number] = new_case_number

                # 生成新文件名
                new_filename = f"{new_case_number}_{image_number}"
                new_file_path = os.path.join(output_dir, new_filename)

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
                png_filename = f"{new_filename}.png"
                png_path = os.path.join(png_output_dir, png_filename)

                # 将DICOM数据归一化到0-255范围并保存为PNG
                normalized_array = ((adjusted_pixel_array - adjusted_pixel_array.min()) /
                                    (adjusted_pixel_array.max() - adjusted_pixel_array.min()) * 255)
                normalized_array = normalized_array.astype(np.uint8)

                # 如果是3通道图像，需要处理
                if len(normalized_array.shape) == 3 and normalized_array.shape[2] == 3:
                    img = Image.fromarray(normalized_array, 'RGB')
                elif len(normalized_array.shape) == 2:
                    img = Image.fromarray(normalized_array, 'L')
                else:
                    # 处理其他情况
                    img = Image.fromarray(
                        normalized_array[:, :, 0] if normalized_array.shape[2] > 1 else normalized_array)

                img.save(png_path)

                # 记录新表格数据
                disease_code = case_to_disease[old_case_number]
                new_records.append([new_case_number, disease_code])

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue

    # 创建新表格
    new_df = pd.DataFrame(new_records, columns=['case_number', 'disease_code'])

    # 保存新表格
    new_excel_path = os.path.join(output_dir, "classify_all_trans_updated.xlsx")
    new_df.to_excel(new_excel_path, index=False, header=False)

    # 也保存一份CSV格式便于查看
    new_csv_path = os.path.join(output_dir, "classify_all_trans_updated.csv")
    new_df.to_csv(new_csv_path, index=False, header=False)

    print(f"\n处理完成!")
    print(f"处理了 {len(new_records)} 个图像记录")
    print(f"新DICOM文件保存在: {output_dir}")
    print(f"新PNG文件保存在: {png_output_dir}")
    print(f"新Excel文件保存在: {new_excel_path}")
    print(f"新CSV文件保存在: {new_csv_path}")

    # 返回一些统计信息
    return {
        'total_processed': len(new_records),
        'unique_cases': len(set([r[0] for r in new_records])),
        'output_dir': output_dir,
        'png_output_dir': png_output_dir,
        'excel_output': new_excel_path
    }


def main():
    """
    主函数，可以在这里调整各种参数
    """

    # 可调整的参数
    params = {
        # 输入目录
        'input_dir': r"D:\med_data\ai\translate\train_all_trans(1)",

        # Excel文件路径
        'excel_path': r"D:\med_data\ai\translate\classify_all_trans.xlsx",

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

    print("开始处理DICOM图像...")
    print(f"参数设置:")
    print(f"  输入目录: {params['input_dir']}")
    print(f"  Excel文件: {params['excel_path']}")
    print(f"  输出目录: {params['output_dir']}")
    print(f"  PNG输出目录: {params['png_output_dir']}")
    print(f"  对比度调整: {params['contrast_percent']}%")
    print(f"  病例号增量: {params['case_number_increment']}")
    print(f"  批处理大小: {params['batch_size']}")
    print("-" * 50)

    # 执行处理
    results = process_dicom_images(**params)

    # 显示结果摘要
    print("\n" + "=" * 50)
    print("处理摘要:")
    print(f"  处理的图像总数: {results['total_processed']}")
    print(f"  处理的唯一病例数: {results['unique_cases']}")
    print("=" * 50)


if __name__ == "__main__":
    # 安装必要的库
    # pip install pydicom pandas numpy opencv-python pillow openpyxl tqdm

    main()
