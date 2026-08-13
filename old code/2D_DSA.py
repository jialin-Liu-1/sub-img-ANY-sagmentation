import os
import pydicom
import numpy as np
from tqdm import tqdm
import logging
import cv2

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_pixel_data(pixel_array):
    """
    将像素数据归一化到0-255范围
    """
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)

    if pixel_max > pixel_min:
        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)

    return normalized


def save_as_png(pixel_array, output_path):
    """
    将像素数组保存为PNG文件
    """
    try:
        normalized_array = normalize_pixel_data(pixel_array)
        cv2.imwrite(output_path, normalized_array)
        return True
    except Exception as e:
        logger.error(f"保存PNG文件时出错: {str(e)}")
        return False


def calculate_first_half_projection(pixel_array, method='mean'):
    """
    计算时间序列前半段的投影

    参数:
        pixel_array: 3D时间序列数组 [时间帧, 高度, 宽度]
        method: 投影方法，'mean'为平均值投影，'min'为最小值投影，'mixed'为混合投影

    返回:
        二维投影图像和是否需要归一化标志
    """
    num_frames = pixel_array.shape[0]

    if num_frames == 0:
        return np.zeros((pixel_array.shape[1], pixel_array.shape[2])), False

    if num_frames == 1:
        return pixel_array[0], False

    # 计算前半段的帧数
    half_frames = num_frames // 2
    if half_frames == 0:
        half_frames = 1

    logger.info(f"时间序列总帧数: {num_frames}, 使用前 {half_frames} 帧")

    # 选取前半段时间序列
    first_half_series = pixel_array[:half_frames]

    # 根据选择的方法计算投影
    if method == 'mean':
        # 平均值投影
        projection = np.mean(first_half_series, axis=0)
        logger.info(f"使用平均值投影")
        return projection, False

    elif method == 'min':
        # 最小值投影
        projection = np.min(first_half_series, axis=0)
        logger.info(f"使用最小值投影")
        return projection, False

    elif method == 'mixed':
        # 混合投影：平均值和最小值的平均
        mean_projection = np.mean(first_half_series, axis=0)
        min_projection = np.min(first_half_series, axis=0)

        # 将两个投影归一化到相同范围后再平均
        mean_norm = (mean_projection - np.min(mean_projection)) / (
                    np.max(mean_projection) - np.min(mean_projection) + 1e-8)
        min_norm = (min_projection - np.min(min_projection)) / (np.max(min_projection) - np.min(min_projection) + 1e-8)

        projection = (mean_norm + min_norm) / 2
        logger.info(f"使用混合投影（平均值和最小值的平均）")
        return projection, True  # 返回归一化标志

    else:
        logger.warning(f"未知方法 '{method}'，默认使用平均值投影")
        projection = np.mean(first_half_series, axis=0)
        return projection, False


def process_individual_dsa_files(input_folder, output_dicom_folder, output_png_folder, projection_method='mean'):
    """
    逐个处理DICOM文件，使用指定的投影方法

    参数:
        input_folder: 输入文件夹路径
        output_dicom_folder: 输出DICOM文件夹路径
        output_png_folder: 输出PNG文件夹路径
        projection_method: 投影方法，'mean'、'min'或'mixed'
    """

    # 创建输出文件夹
    os.makedirs(output_dicom_folder, exist_ok=True)
    os.makedirs(output_png_folder, exist_ok=True)

    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"输出DICOM文件夹: {output_dicom_folder}")
    logger.info(f"输出PNG文件夹: {output_png_folder}")
    logger.info(f"投影方法: {projection_method}")

    # 获取所有DICOM文件
    dicom_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.dcm') or ('.' not in file):
            file_path = os.path.join(input_folder, file)
            if os.path.getsize(file_path) > 1024:
                dicom_files.append(file)

    logger.info(f"找到 {len(dicom_files)} 个DICOM文件")

    if len(dicom_files) == 0:
        logger.error("未找到有效的DICOM文件")
        return

    # 按文件名排序
    dicom_files.sort()

    # 处理统计
    processed_count = 0
    skipped_count = 0

    logger.info("开始处理DICOM文件...")

    for filename in tqdm(dicom_files, desc=f"处理文件"):
        input_path = os.path.join(input_folder, filename)
        output_dicom_path = os.path.join(output_dicom_folder, filename)

        # PNG文件命名
        if '.' in filename:
            png_filename = os.path.splitext(filename)[0] + '.png'
        else:
            png_filename = filename + '.png'
        output_png_path = os.path.join(output_png_folder, png_filename)

        # 如果输出文件已存在，跳过
        if os.path.exists(output_dicom_path) and os.path.exists(output_png_path):
            skipped_count += 1
            continue

        try:
            # 读取DICOM文件
            ds = pydicom.dcmread(input_path, force=True)
            pixel_array = ds.pixel_array
            original_dtype = pixel_array.dtype

            # 处理3D DSA时间序列数据
            if len(pixel_array.shape) == 3:
                num_frames = pixel_array.shape[0]

                # 使用指定的投影方法
                projection, is_normalized = calculate_first_half_projection(pixel_array, projection_method)

                # 保存为PNG格式
                if not os.path.exists(output_png_path):
                    save_as_png(projection, output_png_path)

                # 保存为DICOM格式
                if not os.path.exists(output_dicom_path):
                    output_ds = ds.copy()

                    # ====== 修复的关键部分 ======
                    # 准备DICOM像素数据
                    if is_normalized:
                        # 混合投影返回的是0-1范围的浮点数
                        # 需要根据原始数据类型进行转换
                        if original_dtype == np.uint16:
                            # 将0-1范围映射到0-65535
                            projection_dicom = (projection * 65535).astype(np.uint16)
                        elif original_dtype == np.uint8:
                            # 将0-1范围映射到0-255
                            projection_dicom = (projection * 255).astype(np.uint8)
                        else:
                            # 对于其他类型，尝试保持原始范围
                            original_min = np.min(pixel_array)
                            original_max = np.max(pixel_array)
                            projection_dicom = (projection * (original_max - original_min) + original_min).astype(
                                original_dtype)
                    else:
                        # 对于mean和min投影，直接使用计算出的值
                        # 但需要确保数据类型匹配
                        if projection.dtype != original_dtype:
                            # 转换到原始数据类型
                            if original_dtype == np.uint16:
                                # 确保数值在合适范围内
                                projection_min = np.min(projection)
                                projection_max = np.max(projection)

                                if projection_max > projection_min:
                                    original_min = np.min(pixel_array)
                                    original_max = np.max(pixel_array)

                                    if original_max > original_min:
                                        # 线性映射到原始范围
                                        projection_dicom = ((projection - projection_min) /
                                                            (projection_max - projection_min) *
                                                            (original_max - original_min) + original_min).astype(
                                            np.uint16)
                                    else:
                                        # 使用默认范围
                                        projection_dicom = ((projection - projection_min) /
                                                            (projection_max - projection_min) * 65535).astype(np.uint16)
                                else:
                                    projection_dicom = np.zeros_like(projection, dtype=np.uint16)
                            else:
                                projection_dicom = projection.astype(original_dtype)
                        else:
                            projection_dicom = projection
                    # ============================

                    # 更新DICOM数据
                    output_ds.PixelData = projection_dicom.tobytes()
                    output_ds.Rows, output_ds.Columns = projection.shape

                    if hasattr(output_ds, 'NumberOfFrames'):
                        output_ds.NumberOfFrames = 1

                    # 更新相关标签
                    half_frames = num_frames // 2 if num_frames > 1 else 1
                    method_name = {
                        'mean': '平均值投影',
                        'min': '最小值投影',
                        'mixed': '混合投影'
                    }.get(projection_method, '投影')

                    if hasattr(output_ds, 'SeriesDescription'):
                        original_desc = output_ds.SeriesDescription
                        output_ds.SeriesDescription = f"{original_desc}_{method_name}_前{half_frames}帧"

                    # 保存DICOM文件
                    pydicom.dcmwrite(output_dicom_path, output_ds)
                    logger.info(f"✓ 成功保存DICOM: {filename}")

                processed_count += 1

            elif len(pixel_array.shape) == 2:
                # 2D图像直接复制
                if not os.path.exists(output_dicom_path):
                    ds.save_as(output_dicom_path)

                if not os.path.exists(output_png_path):
                    save_as_png(pixel_array, output_png_path)

                processed_count += 1

            else:
                logger.error(f"文件 {filename} 的维度不支持: {pixel_array.shape}")
                skipped_count += 1

        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            skipped_count += 1

    # 输出处理总结
    logger.info("=" * 50)
    logger.info("处理完成!")
    logger.info(f"成功处理: {processed_count} 个文件")
    logger.info(f"跳过/失败: {skipped_count} 个文件")
    logger.info(f"总计: {len(dicom_files)} 个DICOM文件")
    logger.info("=" * 50)


if __name__ == "__main__":
    # ============================================
    # 在这里选择投影方法：
    # 'mean'  - 平均值投影
    # 'min'   - 最小值投影
    # 'mixed' - 混合投影（平均值和最小值的平均）
    # ============================================
    PROJECTION_METHOD = 'mixed'  # 修改这里选择投影方法

    # 设置输入和输出文件夹路径
    input_folder = r"D:\med_data\ANY\0"
    output_dicom_folder = r"D:\med_data\ai\0"
    output_png_folder = r"D:\med_data\ai\0PNG"

    # 执行处理
    process_individual_dsa_files(
        input_folder=input_folder,
        output_dicom_folder=output_dicom_folder,
        output_png_folder=output_png_folder,
        projection_method=PROJECTION_METHOD
    )