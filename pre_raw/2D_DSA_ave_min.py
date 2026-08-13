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


def save_as_dicom(pixel_array, original_ds, output_path, projection_type="混合投影"):
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


def process_individual_dsa_files_hybrid(input_folder, output_dicom_folder,
                                        output_png_folder=None, time_fraction=0.5,
                                        save_intermediate_projections=False):
    """
    混合投影处理：先计算平均值投影和最小值投影，然后取两者的平均值
    取前50%时间点进行计算

    参数:
        input_folder: 输入文件夹路径
        output_dicom_folder: 输出DICOM文件夹路径
        output_png_folder: 输出PNG文件夹路径（可选）
        time_fraction: 截取时间序列的前几分之一，默认0.5表示前50%
        save_intermediate_projections: 是否保存中间投影结果（平均值和最小值投影）
    """

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dicom_folder):
        os.makedirs(output_dicom_folder)
        logger.info(f"创建混合投影DICOM输出文件夹: {output_dicom_folder}")

    # 如果指定了PNG输出文件夹，则创建
    if output_png_folder and not os.path.exists(output_png_folder):
        os.makedirs(output_png_folder)
        logger.info(f"创建PNG输出文件夹: {output_png_folder}")

    # 如果需要保存中间投影，创建子文件夹
    if save_intermediate_projections:
        mean_projection_dicom_folder = os.path.join(output_dicom_folder, "mean_projection")
        min_projection_dicom_folder = os.path.join(output_dicom_folder, "min_projection")
        for folder in [mean_projection_dicom_folder, min_projection_dicom_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        if output_png_folder:
            mean_projection_png_folder = os.path.join(output_png_folder, "mean_projection")
            min_projection_png_folder = os.path.join(output_png_folder, "min_projection")
            for folder in [mean_projection_png_folder, min_projection_png_folder]:
                if not os.path.exists(folder):
                    os.makedirs(folder)

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

    logger.info(f"开始混合投影处理患者DSA图像（取前 {time_fraction * 100:.0f}% 时间点）...")
    logger.info("处理流程: 1)计算平均值投影 2)计算最小值投影 3)两者求平均")

    for filename in tqdm(dicom_files, desc="混合投影处理"):
        input_path = os.path.join(input_folder, filename)
        output_dicom_path = os.path.join(output_dicom_folder, filename)

        # 如果指定了PNG输出，生成PNG文件名
        if output_png_folder:
            png_filename = os.path.splitext(filename)[0] + '_hybrid.png'
            output_png_path = os.path.join(output_png_folder, png_filename)
        else:
            output_png_path = None

        # 如果输出DICOM文件已存在，跳过
        if os.path.exists(output_dicom_path):
            logger.info(f"混合投影DICOM文件 {filename} 已存在，跳过")
            skipped_count += 1
            continue

        try:
            # 读取单个患者的DICOM文件
            logger.info(f"正在进行混合投影处理: {filename}")
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

                # 第一步：计算平均值投影
                mean_projection = np.mean(selected_frames, axis=0).astype(np.float32)

                # 第二步：计算最小值投影
                min_projection = np.min(selected_frames, axis=0).astype(np.float32)

                # 第三步：计算两个投影的平均值（混合投影）
                hybrid_projection = ((mean_projection + min_projection) / 2.0).astype(pixel_array.dtype)

                # 计算统计信息
                mean_stats = {
                    'mean': np.mean(mean_projection),
                    'std': np.std(mean_projection),
                    'min': np.min(mean_projection),
                    'max': np.max(mean_projection)
                }

                min_stats = {
                    'mean': np.mean(min_projection),
                    'std': np.std(min_projection),
                    'min': np.min(min_projection),
                    'max': np.max(min_projection)
                }

                hybrid_stats = {
                    'mean': np.mean(hybrid_projection),
                    'std': np.std(hybrid_projection),
                    'min': np.min(hybrid_projection),
                    'max': np.max(hybrid_projection)
                }

                # 计算投影间的相关性
                correlation_mean_min = np.corrcoef(mean_projection.flatten(), min_projection.flatten())[0, 1]
                correlation_mean_hybrid = np.corrcoef(mean_projection.flatten(), hybrid_projection.flatten())[0, 1]
                correlation_min_hybrid = np.corrcoef(min_projection.flatten(), hybrid_projection.flatten())[0, 1]

                logger.info(f"时间序列总帧数: {total_frames}")
                logger.info(f"截取时间点: 前 {frames_to_use} 帧（前 {time_fraction * 100:.0f}%）")
                logger.info(f"混合投影后形状: {hybrid_projection.shape}")
                logger.info("投影统计对比:")
                logger.info(f"  平均值投影 - 均值: {mean_stats['mean']:.2f}, 标准差: {mean_stats['std']:.2f}, "
                            f"范围: [{mean_stats['min']:.2f}, {mean_stats['max']:.2f}]")
                logger.info(f"  最小值投影 - 均值: {min_stats['mean']:.2f}, 标准差: {min_stats['std']:.2f}, "
                            f"范围: [{min_stats['min']:.2f}, {min_stats['max']:.2f}]")
                logger.info(f"  混合投影 - 均值: {hybrid_stats['mean']:.2f}, 标准差: {hybrid_stats['std']:.2f}, "
                            f"范围: [{hybrid_stats['min']:.2f}, {hybrid_stats['max']:.2f}]")
                logger.info(f"投影相关性: 平均值vs最小值={correlation_mean_min:.4f}, "
                            f"平均值vs混合={correlation_mean_hybrid:.4f}, 最小值vs混合={correlation_min_hybrid:.4f}")

                # 设置投影类型描述
                projection_type = f"前{time_fraction * 100:.0f}%时间混合投影(均值+最小值)/2"

                # 保存混合投影为DICOM格式
                if save_as_dicom(hybrid_projection, ds, output_dicom_path, projection_type):
                    logger.info(f"✓ 成功保存混合投影DICOM: {filename}")
                    processed_count += 1
                else:
                    logger.error(f"✗ 保存混合投影DICOM失败: {filename}")
                    skipped_count += 1
                    continue

                # 可选：保存混合投影为PNG格式
                if output_png_folder and not os.path.exists(output_png_path):
                    if save_as_png(hybrid_projection, output_png_path):
                        logger.info(f"✓ 成功保存混合投影PNG: {png_filename}")
                    else:
                        logger.error(f"✗ 保存混合投影PNG失败: {png_filename}")

                # 如果需要保存中间投影结果
                if save_intermediate_projections:
                    # 保存平均值投影
                    mean_dicom_path = os.path.join(mean_projection_dicom_folder, filename)
                    mean_png_path = None
                    if output_png_folder:
                        mean_png_filename = os.path.splitext(filename)[0] + '_mean.png'
                        mean_png_path = os.path.join(mean_projection_png_folder, mean_png_filename)

                    save_as_dicom(mean_projection.astype(pixel_array.dtype), ds, mean_dicom_path,
                                  f"前{time_fraction * 100:.0f}%时间平均值投影")
                    if mean_png_path and not os.path.exists(mean_png_path):
                        save_as_png(mean_projection, mean_png_path)

                    # 保存最小值投影
                    min_dicom_path = os.path.join(min_projection_dicom_folder, filename)
                    min_png_path = None
                    if output_png_folder:
                        min_png_filename = os.path.splitext(filename)[0] + '_min.png'
                        min_png_path = os.path.join(min_projection_png_folder, min_png_filename)

                    save_as_dicom(min_projection.astype(pixel_array.dtype), ds, min_dicom_path,
                                  f"前{time_fraction * 100:.0f}%时间最小值投影")
                    if min_png_path and not os.path.exists(min_png_path):
                        save_as_png(min_projection, min_png_path)

                    logger.info(f"✓ 已保存中间投影结果（平均值投影和最小值投影）")

            elif len(pixel_array.shape) == 2:
                # 如果是2D图像，直接复制（混合投影无意义）
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
    logger.info("混合投影处理完成!")
    logger.info(f"成功处理: {processed_count} 个患者")
    logger.info(f"跳过/失败: {skipped_count} 个患者")
    logger.info(f"总计: {len(dicom_files)} 个患者文件")
    logger.info(f"时间截取比例: 前 {time_fraction * 100:.0f}% 时间点")
    logger.info(f"投影方法: 混合投影 (平均值投影 + 最小值投影) / 2")
    logger.info(f"混合投影DICOM保存到: {output_dicom_folder}")
    if save_intermediate_projections:
        logger.info(f"中间投影结果保存到: {output_dicom_folder}/mean_projection 和 min_projection")
    if output_png_folder:
        logger.info(f"PNG文件保存到: {output_png_folder}")
    logger.info("=" * 60)


def analyze_hybrid_benefits(input_folder, output_analysis_file, time_fraction=0.5, num_samples=5):
    """
    分析混合投影的优势，生成对比报告
    """
    logger.info("开始混合投影优势分析...")

    # 获取文件列表
    dicom_files = [f for f in os.listdir(input_folder)
                   if f.endswith('.dcm') or ('.' not in f)]
    dicom_files.sort()

    # 选择样本文件进行分析
    sample_files = dicom_files[:min(num_samples, len(dicom_files))]

    analysis_results = []

    for filename in tqdm(sample_files, desc="优势分析"):
        try:
            input_path = os.path.join(input_folder, filename)
            ds = pydicom.dcmread(input_path)
            pixel_array = ds.pixel_array

            if len(pixel_array.shape) == 3:
                total_frames = pixel_array.shape[0]
                frames_to_use = max(1, int(total_frames * time_fraction))
                selected_frames = pixel_array[:frames_to_use, :, :]

                # 计算三种投影
                mean_proj = np.mean(selected_frames, axis=0).astype(np.float32)
                min_proj = np.min(selected_frames, axis=0).astype(np.float32)
                hybrid_proj = (mean_proj + min_proj) / 2.0

                # 计算对比度（标准差/均值）
                mean_contrast = np.std(mean_proj) / (np.mean(mean_proj) + 1e-6)
                min_contrast = np.std(min_proj) / (np.mean(min_proj) + 1e-6)
                hybrid_contrast = np.std(hybrid_proj) / (np.mean(hybrid_proj) + 1e-6)

                # 计算信噪比估计
                mean_snr = np.mean(mean_proj) / (np.std(mean_proj) + 1e-6)
                min_snr = np.mean(min_proj) / (np.std(min_proj) + 1e-6)
                hybrid_snr = np.mean(hybrid_proj) / (np.std(hybrid_proj) + 1e-6)

                analysis = {
                    'filename': filename,
                    'total_frames': total_frames,
                    'used_frames': frames_to_use,
                    'contrast': {
                        'mean_projection': mean_contrast,
                        'min_projection': min_contrast,
                        'hybrid_projection': hybrid_contrast
                    },
                    'snr': {
                        'mean_projection': mean_snr,
                        'min_projection': min_snr,
                        'hybrid_projection': hybrid_snr
                    },
                    'correlation_with_hybrid': {
                        'mean_projection': np.corrcoef(mean_proj.flatten(), hybrid_proj.flatten())[0, 1],
                        'min_projection': np.corrcoef(min_proj.flatten(), hybrid_proj.flatten())[0, 1]
                    }
                }

                analysis_results.append(analysis)

                logger.info(f"文件 {filename} 对比度分析:")
                logger.info(f"  平均值投影对比度: {mean_contrast:.4f}, SNR: {mean_snr:.2f}")
                logger.info(f"  最小值投影对比度: {min_contrast:.4f}, SNR: {min_snr:.2f}")
                logger.info(f"  混合投影对比度: {hybrid_contrast:.4f}, SNR: {hybrid_snr:.2f}")

        except Exception as e:
            logger.error(f"分析文件 {filename} 时出错: {str(e)}")

    # 生成总结报告
    if analysis_results:
        avg_mean_contrast = np.mean([r['contrast']['mean_projection'] for r in analysis_results])
        avg_min_contrast = np.mean([r['contrast']['min_projection'] for r in analysis_results])
        avg_hybrid_contrast = np.mean([r['contrast']['hybrid_projection'] for r in analysis_results])

        logger.info("=" * 50)
        logger.info("混合投影优势分析总结:")
        logger.info(f"  平均对比度 - 平均值投影: {avg_mean_contrast:.4f}")
        logger.info(f"  平均对比度 - 最小值投影: {avg_min_contrast:.4f}")
        logger.info(f"  平均对比度 - 混合投影: {avg_hybrid_contrast:.4f}")
        logger.info(
            f"  混合投影相对于平均值投影的对比度提升: {((avg_hybrid_contrast / avg_mean_contrast) - 1) * 100:.2f}%")
        logger.info(
            f"  混合投影相对于最小值投影的对比度提升: {((avg_hybrid_contrast / avg_min_contrast) - 1) * 100:.2f}%")

    return analysis_results


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
    output_dicom_folder = r"D:\med_data\ANY\process\NEW_DSA_mix"
    output_png_folder = None  # 设置为 None 如果不需输出PNG，或指定路径

    # 设置时间截取比例（默认0.5表示前50%）
    time_fraction = 0.6

    # 是否保存中间投影结果（平均值投影和最小值投影）
    save_intermediate_projections = False

    # 是否进行优势分析
    enable_analysis = True

    logger.info("开始混合投影处理患者DSA图像序列...")
    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"混合投影DICOM输出文件夹: {output_dicom_folder}")
    logger.info(f"时间截取比例: 前 {time_fraction * 100:.0f}% 时间点")
    logger.info(f"投影方法: 混合投影 = (平均值投影 + 最小值投影) / 2")
    if output_png_folder:
        logger.info(f"PNG输出文件夹: {output_png_folder}")
    else:
        logger.info("PNG输出: 禁用")

    # 首先检查文件结构
    check_dicom_structure(input_folder)

    try:
        # 执行混合投影处理
        process_individual_dsa_files_hybrid(
            input_folder, output_dicom_folder,
            output_png_folder, time_fraction,
            save_intermediate_projections
        )

        # 如果启用优势分析
        if enable_analysis:
            analyze_hybrid_benefits(input_folder, None, time_fraction)

        # 预览生成的DICOM文件
        preview_dicom_files(output_dicom_folder)

        logger.info("混合投影处理完成！")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")