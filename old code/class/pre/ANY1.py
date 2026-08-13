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


def save_as_png(pixel_array, output_path, method='opencv'):
    """
    将像素数组保存为PNG文件

    参数:
        pixel_array: 像素数组
        output_path: 输出路径
        method: 保存方法 ('opencv' 或 'pil')
    """
    try:
        # 归一化像素数据
        normalized_array = normalize_pixel_data(pixel_array)

        if method == 'opencv':
            # 使用OpenCV保存
            cv2.imwrite(output_path, normalized_array)
        else:
            # 使用PIL保存
            image = Image.fromarray(normalized_array)
            image.save(output_path)

        return True
    except Exception as e:
        logger.error(f"保存PNG文件时出错: {str(e)}")
        return False


def process_individual_dsa_files(input_folder, output_dicom_folder, output_png_folder):
    """
    逐个处理每个患者的DSA图像文件，并保存为DICOM和PNG格式

    参数:
        input_folder: 输入文件夹路径
        output_dicom_folder: 输出DICOM文件夹路径
        output_png_folder: 输出PNG文件夹路径
    """

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dicom_folder):
        os.makedirs(output_dicom_folder)
        logger.info(f"创建DICOM输出文件夹: {output_dicom_folder}")

    if not os.path.exists(output_png_folder):
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

    logger.info("开始逐个处理患者DSA图像...")

    for filename in tqdm(dicom_files, desc="处理患者文件"):
        input_path = os.path.join(input_folder, filename)
        output_dicom_path = os.path.join(output_dicom_folder, filename)

        # 生成PNG文件名（保持原文件名，只修改扩展名）
        png_filename = os.path.splitext(filename)[0] + '.png'
        output_png_path = os.path.join(output_png_folder, png_filename)

        # 如果输出文件已存在，跳过
        dicom_exists = os.path.exists(output_dicom_path)
        png_exists = os.path.exists(output_png_path)

        if dicom_exists and png_exists:
            logger.info(f"文件 {filename} 已存在，跳过")
            skipped_count += 1
            continue

        try:
            # 读取单个患者的DICOM文件
            logger.info(f"正在处理患者: {filename}")
            ds = pydicom.dcmread(input_path)

            # 获取像素数据
            pixel_array = ds.pixel_array
            logger.info(f"原始图像形状: {pixel_array.shape}")

            # 处理3D DSA时间序列数据
            if len(pixel_array.shape) == 3:
                # 假设形状为 [时间帧, 高度, 宽度]
                # 沿时间维度取最小值，显示注入对比剂后的血管形状
                min_projection = np.min(pixel_array, axis=0)

                logger.info(f"时间序列帧数: {pixel_array.shape[0]}")
                logger.info(f"最小值投影后形状: {min_projection.shape}")

                # 保存为DICOM格式
                if not dicom_exists:
                    # 创建输出DICOM文件
                    output_ds = ds.copy()

                    # 更新像素数据
                    output_ds.PixelData = min_projection.astype(ds.pixel_array.dtype).tobytes()

                    # 更新图像尺寸信息
                    output_ds.Rows, output_ds.Columns = min_projection.shape

                    # 更新帧数信息（从多帧变为单帧）
                    if hasattr(output_ds, 'NumberOfFrames'):
                        output_ds.NumberOfFrames = 1
                        logger.info("已更新帧数: 多帧 -> 单帧")

                    # 添加处理说明
                    original_comment = ""
                    if 'ImageComments' in output_ds:
                        original_comment = output_ds.ImageComments

                    output_ds.ImageComments = f"DSA最小值投影 - 时间帧数: {pixel_array.shape[0]} - {original_comment}"

                    # 保存处理后的DICOM图像
                    pydicom.dcmwrite(output_dicom_path, output_ds)
                    logger.info(f"✓ 成功保存DICOM: {filename}")

                # 保存为PNG格式
                if not png_exists:
                    if save_as_png(min_projection, output_png_path):
                        logger.info(f"✓ 成功保存PNG: {png_filename}")
                    else:
                        logger.error(f"✗ 保存PNG失败: {png_filename}")

                processed_count += 1

            elif len(pixel_array.shape) == 2:
                # 如果是2D图像，说明已经是单帧，直接复制
                logger.warning(f"文件 {filename} 是2D图像，直接复制")

                # 保存DICOM
                if not dicom_exists:
                    ds.save_as(output_dicom_path)

                # 保存PNG
                if not png_exists:
                    if save_as_png(pixel_array, output_png_path):
                        logger.info(f"✓ 成功保存PNG: {png_filename}")
                    else:
                        logger.error(f"✗ 保存PNG失败: {png_filename}")

                processed_count += 1

            else:
                logger.error(f"文件 {filename} 的维度不支持: {pixel_array.shape}")
                skipped_count += 1

        except Exception as e:
            logger.error(f"处理患者 {filename} 时出错: {str(e)}")
            skipped_count += 1
            continue

    # 输出处理总结
    logger.info("=" * 50)
    logger.info("处理完成!")
    logger.info(f"成功处理: {processed_count} 个患者")
    logger.info(f"跳过/失败: {skipped_count} 个患者")
    logger.info(f"总计: {len(dicom_files)} 个患者文件")
    logger.info(f"DICOM文件保存到: {output_dicom_folder}")
    logger.info(f"PNG文件保存到: {output_png_folder}")
    logger.info("=" * 50)


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
            logger.info("  " + "-" * 30)
        except Exception as e:
            logger.error(f"分析文件 {filename} 时出错: {str(e)}")


def preview_png_files(png_folder):
    """
    预览生成的PNG文件
    """
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]
    if png_files:
        logger.info(f"生成的PNG文件数量: {len(png_files)}")
        logger.info("前5个PNG文件:")
        for i, file in enumerate(png_files[:5]):
            file_path = os.path.join(png_folder, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            logger.info(f"  {i + 1}. {file} ({file_size:.1f} KB)")
    else:
        logger.warning("未找到PNG文件")


if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"D:\med_data\ANY\0"
    output_dicom_folder = r"D:\med_data\ai\0"
    output_png_folder = r"D:\med_data\ai\0PNG"

    logger.info("开始处理患者DSA图像序列...")
    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"DICOM输出文件夹: {output_dicom_folder}")
    logger.info(f"PNG输出文件夹: {output_png_folder}")

    # 首先检查文件结构
    check_dicom_structure(input_folder)

    try:
        # 逐个处理每个患者的DSA文件
        process_individual_dsa_files(input_folder, output_dicom_folder, output_png_folder)

        # 预览生成的PNG文件
        preview_png_files(output_png_folder)

        logger.info("所有患者处理完成！")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
