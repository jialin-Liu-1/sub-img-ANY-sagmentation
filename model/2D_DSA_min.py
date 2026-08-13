import os
import pydicom
import numpy as np
from tqdm import tqdm
import logging
import cv2
from PIL import Image

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_pixel_data(pixel_array):
    """
    将DICOM像素数据归一化到0-255范围
    """
    # 将像素值缩放到0-255范围
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)

    if pixel_max > pixel_min:
        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)

    return normalized


def save_as_png(pixel_array, output_path):
    """
    将像素数组保存为PNG文件（可选功能）
    """
    try:
        # 归一化像素数据
        normalized_array = normalize_pixel_data(pixel_array)

        # 使用OpenCV保存
        cv2.imwrite(output_path, normalized_array)
        return True
    except Exception as e:
        logger.error(f"保存PNG文件时出错: {str(e)}")
        return False


def save_as_dicom(pixel_array, original_ds, output_path, projection_type="最小值投影"):
    """
    将像素数组保存为DICOM文件

    参数:
        pixel_array: 处理后的像素数组
        original_ds: 原始DICOM数据集
        output_path: 输出路径
        projection_type: 投影类型描述
    """
    try:
        # 创建输出DICOM文件
        output_ds = original_ds.copy()

        # 更新像素数据
        output_ds.PixelData = pixel_array.tobytes()

        # 更新图像尺寸信息
        output_ds.Rows, output_ds.Columns = pixel_array.shape

        # 更新帧数信息（从多帧变为单帧）
        if hasattr(output_ds, 'NumberOfFrames'):
            output_ds.NumberOfFrames = 1

        # 更新相关的DICOM标签
        if hasattr(output_ds, 'ImagesInAcquisition'):
            output_ds.ImagesInAcquisition = 1

        # 添加处理说明到图像注释
        original_comment = ""
        if 'ImageComments' in output_ds:
            original_comment = output_ds.ImageComments

        frame_count = original_ds.pixel_array.shape[0] if len(original_ds.pixel_array.shape) == 3 else 1
        output_ds.ImageComments = f"DSA {projection_type} - 原始时间帧数: {frame_count} - {original_comment}"

        # 更新序列描述
        if hasattr(output_ds, 'SeriesDescription'):
            original_series_desc = output_ds.SeriesDescription
            output_ds.SeriesDescription = f"{original_series_desc} {projection_type}"

        # 更新图像类型（如果存在）
        if hasattr(output_ds, 'ImageType'):
            image_type = list(output_ds.ImageType)
            if 'DERIVED' not in image_type:
                image_type.append('DERIVED')
            if 'SECONDARY' not in image_type:
                image_type.append('SECONDARY')
            output_ds.ImageType = image_type

        # 保存处理后的DICOM图像
        pydicom.dcmwrite(output_path, output_ds)
        return True

    except Exception as e:
        logger.error(f"保存DICOM文件时出错: {str(e)}")
        return False


def process_individual_dsa_files_min_intensity(input_folder, output_dicom_folder,
                                               output_png_folder=None, time_fraction=0.5):
    """
    使用最小值投影处理每个患者的DSA图像文件
    取前50%时间点中每个像素位置的最小值

    参数:
        input_folder: 输入文件夹路径
        output_dicom_folder: 输出DICOM文件夹路径
        output_png_folder: 输出PNG文件夹路径（可选）
        time_fraction: 截取时间序列的前几分之一，默认0.5表示前50%
    """

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dicom_folder):
        os.makedirs(output_dicom_folder)
        logger.info(f"创建DICOM输出文件夹: {output_dicom_folder}")

    # 如果指定了PNG输出文件夹，则创建
    if output_png_folder and not os.path.exists(output_png_folder):
        os.makedirs(output_png_folder)
        logger.info(f"创建PNG输出文件夹: {output_png_folder}")

    # 获取所有DICOM文件
    dicom_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.dcm') or ('.' not in file):
            dicom_files.append(file)

    logger.info(f"找到 {len(dicom_files)} 个DICOM文件（每个文件对应一个患者）")

    if len(dicom_files) == 0:
        logger.error("在输入文件夹中未找到DICOM文件")
        return

    # 按文件名排序
    dicom_files.sort()

    # 处理统计
    processed_count = 0
    skipped_count = 0

    logger.info(f"开始最小值投影处理患者DSA图像（取前 {time_fraction * 100:.0f}% 时间点）...")

    for filename in tqdm(dicom_files, desc="最小值投影处理"):
        input_path = os.path.join(input_folder, filename)
        output_dicom_path = os.path.join(output_dicom_folder, filename)

        # 如果指定了PNG输出，生成PNG文件名
        if output_png_folder:
            png_filename = os.path.splitext(filename)[0] + '.png'
            output_png_path = os.path.join(output_png_folder, png_filename)
        else:
            output_png_path = None

        # 如果输出DICOM文件已存在，跳过
        if os.path.exists(output_dicom_path):
            logger.info(f"DICOM文件 {filename} 已存在，跳过")
            skipped_count += 1
            continue

        try:
            # 读取单个患者的DICOM文件
            logger.info(f"正在进行最小值投影处理: {filename}")
            ds = pydicom.dcmread(input_path)

            # 获取像素数据
            pixel_array = ds.pixel_array
            logger.info(f"原始图像形状: {pixel_array.shape}")

            # 处理3D DSA时间序列数据
            if len(pixel_array.shape) == 3:
                # 假设形状为 [时间帧, 高度, 宽度]
                total_frames = pixel_array.shape[0]

                # 计算截取的时间点数量（前50%）
                frames_to_use = max(1, int(total_frames * time_fraction))

                # 截取前50%时间点的图像
                selected_frames = pixel_array[:frames_to_use, :, :]

                # 沿时间维度取最小值 - 这是核心改变
                min_projection = np.min(selected_frames, axis=0).astype(pixel_array.dtype)

                # 可选：同时计算统计信息用于日志
                mean_value = np.mean(min_projection)
                std_value = np.std(min_projection)
                min_value = np.min(min_projection)
                max_value = np.max(min_projection)

                logger.info(f"时间序列总帧数: {total_frames}")
                logger.info(f"截取时间点: 前 {frames_to_use} 帧（前 {time_fraction * 100:.0f}%）")
                logger.info(f"最小值投影后形状: {min_projection.shape}")
                logger.info(f"投影统计 - 均值: {mean_value:.2f}, 标准差: {std_value:.2f}")
                logger.info(f"投影统计 - 最小值: {min_value:.2f}, 最大值: {max_value:.2f}")

                # 设置投影类型描述
                projection_type = f"前{time_fraction * 100:.0f}%时间最小值投影"

                # 保存为DICOM格式
                if save_as_dicom(min_projection, ds, output_dicom_path, projection_type):
                    logger.info(f"✓ 成功保存DICOM: {filename}")
                    processed_count += 1
                else:
                    logger.error(f"✗ 保存DICOM失败: {filename}")
                    skipped_count += 1
                    continue

                # 可选：保存为PNG格式
                if output_png_folder and not os.path.exists(output_png_path):
                    if save_as_png(min_projection, output_png_path):
                        logger.info(f"✓ 成功保存PNG: {png_filename}")
                    else:
                        logger.error(f"✗ 保存PNG失败: {png_filename}")

            elif len(pixel_array.shape) == 2:
                # 如果是2D图像，说明已经是单帧，直接复制（最小值投影无意义）
                logger.warning(f"文件 {filename} 是2D图像，直接复制DICOM")

                # 直接保存原始DICOM
                ds.save_as(output_dicom_path)
                logger.info(f"✓ 成功复制DICOM: {filename}")
                processed_count += 1

                # 可选：保存为PNG格式
                if output_png_folder and not os.path.exists(output_png_path):
                    if save_as_png(pixel_array, output_png_path):
                        logger.info(f"✓ 成功保存PNG: {png_filename}")
                    else:
                        logger.error(f"✗ 保存PNG失败: {png_filename}")

            else:
                logger.error(f"文件 {filename} 的维度不支持: {pixel_array.shape}")
                skipped_count += 1

        except Exception as e:
            logger.error(f"处理患者 {filename} 时出错: {str(e)}")
            skipped_count += 1
            continue

    # 输出处理总结
    logger.info("=" * 60)
    logger.info("最小值投影处理完成!")
    logger.info(f"成功处理: {processed_count} 个患者")
    logger.info(f"跳过/失败: {skipped_count} 个患者")
    logger.info(f"总计: {len(dicom_files)} 个患者文件")
    logger.info(f"时间截取比例: 前 {time_fraction * 100:.0f}% 时间点")
    logger.info(f"投影方法: 最小值投影 (Minimum Intensity Projection)")
    logger.info(f"DICOM文件保存到: {output_dicom_folder}")
    if output_png_folder:
        logger.info(f"PNG文件保存到: {output_png_folder}")
    logger.info("=" * 60)


def compare_projections(input_folder, output_folder, time_fraction=0.5):
    """
    比较不同投影方法的结果（用于分析）
    同时生成平均值投影和最小值投影，便于对比
    """
    logger.info("开始对比投影分析...")

    # 创建子文件夹
    mean_folder = os.path.join(output_folder, "mean_projection")
    min_folder = os.path.join(output_folder, "min_projection")

    for folder in [mean_folder, min_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 获取文件列表
    dicom_files = [f for f in os.listdir(input_folder)
                   if f.endswith('.dcm') or ('.' not in f)]
    dicom_files.sort()

    comparison_stats = []

    for filename in tqdm(dicom_files[:3], desc="对比分析"):  # 只分析前3个文件
        try:
            input_path = os.path.join(input_folder, filename)
            ds = pydicom.dcmread(input_path)
            pixel_array = ds.pixel_array

            if len(pixel_array.shape) == 3:
                total_frames = pixel_array.shape[0]
                frames_to_use = max(1, int(total_frames * time_fraction))
                selected_frames = pixel_array[:frames_to_use, :, :]

                # 计算平均值投影
                mean_proj = np.mean(selected_frames, axis=0)

                # 计算最小值投影
                min_proj = np.min(selected_frames, axis=0)

                # 比较统计信息
                stats = {
                    'filename': filename,
                    'total_frames': total_frames,
                    'used_frames': frames_to_use,
                    'mean_proj_stats': {
                        'mean': np.mean(mean_proj),
                        'std': np.std(mean_proj),
                        'min': np.min(mean_proj),
                        'max': np.max(mean_proj)
                    },
                    'min_proj_stats': {
                        'mean': np.mean(min_proj),
                        'std': np.std(min_proj),
                        'min': np.min(min_proj),
                        'max': np.max(min_proj)
                    },
                    'correlation': np.corrcoef(mean_proj.flatten(), min_proj.flatten())[0, 1]
                }

                comparison_stats.append(stats)

                logger.info(f"文件 {filename} 对比分析:")
                logger.info(f"  平均值投影 - 均值: {stats['mean_proj_stats']['mean']:.2f}, "
                            f"标准差: {stats['mean_proj_stats']['std']:.2f}")
                logger.info(f"  最小值投影 - 均值: {stats['min_proj_stats']['mean']:.2f}, "
                            f"标准差: {stats['min_proj_stats']['std']:.2f}")
                logger.info(f"  相关系数: {stats['correlation']:.4f}")
                logger.info(f"  像素值差异 - 平均: {np.mean(np.abs(mean_proj - min_proj)):.2f}")

        except Exception as e:
            logger.error(f"对比分析文件 {filename} 时出错: {str(e)}")

    return comparison_stats


def check_dicom_structure(input_folder):
    """
    检查DICOM文件结构，了解数据组织方式
    """
    logger.info("正在分析DICOM文件结构...")

    dicom_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.dcm') or ('.' not in file):
            dicom_files.append(file)

    if len(dicom_files) == 0:
        logger.error("未找到DICOM文件")
        return

    # 随机检查几个文件了解结构
    sample_files = dicom_files[:3] if len(dicom_files) >= 3 else dicom_files

    for filename in sample_files:
        file_path = os.path.join(input_folder, filename)
        try:
            ds = pydicom.dcmread(file_path)
            logger.info(f"文件 {filename}:")
            logger.info(f"  - 图像形状: {ds.pixel_array.shape}")
            logger.info(f"  - 图像维度: {len(ds.pixel_array.shape)}D")
            if hasattr(ds, 'NumberOfFrames'):
                logger.info(f"  - 帧数: {ds.NumberOfFrames}")
            if hasattr(ds, 'SeriesDescription'):
                logger.info(f"  - 序列描述: {ds.SeriesDescription}")
            if hasattr(ds, 'ImageType'):
                logger.info(f"  - 图像类型: {ds.ImageType}")
            logger.info("  " + "-" * 30)
        except Exception as e:
            logger.error(f"分析文件 {filename} 时出错: {str(e)}")


def preview_dicom_files(dicom_folder):
    """
    预览生成的DICOM文件
    """
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm') or ('.' not in f)]
    if dicom_files:
        logger.info(f"生成的DICOM文件数量: {len(dicom_files)}")
        logger.info("前5个DICOM文件:")
        for i, file in enumerate(dicom_files[:5]):
            file_path = os.path.join(dicom_folder, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            logger.info(f"  {i + 1}. {file} ({file_size:.1f} KB)")
    else:
        logger.warning("未找到DICOM文件")


if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"D:\med_data\ANY\process\500+"
    output_dicom_folder = r"D:\med_data\ANY\process\NEW_DSA_min"
    output_png_folder = None  # 设置为 None 如果不需输出PNG，或指定路径如 r"D:\med_data\ANY\3"

    # 设置时间截取比例（默认0.5表示前50%）
    time_fraction = 0.6

    # 是否启用对比分析
    enable_comparison = False  # 设置为True可同时生成并比较两种投影

    logger.info("开始最小值投影处理患者DSA图像序列...")
    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"DICOM输出文件夹: {output_dicom_folder}")
    logger.info(f"时间截取比例: 前 {time_fraction * 100:.0f}% 时间点")
    logger.info(f"投影方法: 最小值投影 (每像素取前50%时间点中的最小值)")
    if output_png_folder:
        logger.info(f"PNG输出文件夹: {output_png_folder}")
    else:
        logger.info("PNG输出: 禁用")

    # 首先检查文件结构
    check_dicom_structure(input_folder)

    try:
        # 执行最小值投影处理
        process_individual_dsa_files_min_intensity(
            input_folder, output_dicom_folder,
            output_png_folder, time_fraction
        )

        # 如果启用对比分析
        if enable_comparison:
            comparison_folder = os.path.join(os.path.dirname(output_dicom_folder), "comparison")
            compare_projections(input_folder, comparison_folder, time_fraction)

        # 预览生成的DICOM文件
        preview_dicom_files(output_dicom_folder)

        logger.info("最小值投影处理完成！")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")