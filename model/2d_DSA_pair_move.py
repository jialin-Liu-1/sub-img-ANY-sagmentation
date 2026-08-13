import os
import shutil
import pydicom
import numpy as np
from tqdm import tqdm
import logging
import cv2
from PIL import Image
from enum import Enum

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 定义投影类型枚举
class ProjectionType(Enum):
    MIN = "最小值投影"
    MEAN = "平均值投影"


def normalize_to_display_range(pixel_array):
    """
    将像素数据归一化到0-255范围用于显示
    保持数据的相对关系，不改变数据分布
    """
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)

    if pixel_max > pixel_min:
        # 线性映射到0-255范围
        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)

    return normalized


def save_as_png_grayscale(pixel_array, output_path):
    """
    将像素数组保存为灰度PNG文件，保持原始数据特征

    参数:
        pixel_array: 原始像素数组（保持原始数据类型和数值范围）
        output_path: 输出路径
    """
    try:
        # 将原始数据归一化到0-255范围用于显示
        display_array = normalize_to_display_range(pixel_array)

        # 保存为灰度PNG，不进行任何颜色映射或修改
        cv2.imwrite(output_path, display_array)
        return True
    except Exception as e:
        logger.error(f"保存PNG文件时出错: {str(e)}")
        return False


def save_as_dicom(pixel_array, original_ds, output_path, projection_type="最小值投影"):
    """
    将像素数组保存为DICOM文件（无后缀格式），保持原始数据

    参数:
        pixel_array: 原始像素数组（保持原始数据类型和数值范围）
        original_ds: 原始DICOM数据集
        output_path: 输出路径
        projection_type: 投影类型描述
    """
    try:
        # 创建输出DICOM文件
        output_ds = original_ds.copy()

        # 保持原始像素数据类型，直接更新像素数据
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

        # 保存为无后缀的DICOM文件，保持原始数据格式
        pydicom.dcmwrite(output_path, output_ds)
        return True

    except Exception as e:
        logger.error(f"保存DICOM文件时出错: {str(e)}")
        return False


def apply_time_projection(dsa_sequence, projection_type=ProjectionType.MIN, time_fraction=0.5):
    """
    对DSA时间序列进行时间投影，保持原始数据类型和数值范围

    参数:
        dsa_sequence: DSA时间序列数据 [时间帧, 高度, 宽度]
        projection_type: 投影类型（最小值或平均值）
        time_fraction: 时间截取比例

    返回:
        projection: 投影结果，保持原始数据类型
        total_frames: 总帧数
        frames_to_use: 使用的帧数
    """
    total_frames = dsa_sequence.shape[0]
    frames_to_use = max(1, int(total_frames * time_fraction))
    selected_frames = dsa_sequence[:frames_to_use, :, :]

    if projection_type == ProjectionType.MIN:
        # 沿时间轴取最小值，保持原始数据类型
        projection = np.min(selected_frames, axis=0)
    else:  # MEAN
        # 沿时间轴取平均值，保持原始数据类型
        projection = np.mean(selected_frames, axis=0).astype(dsa_sequence.dtype)

    return projection, total_frames, frames_to_use


def get_matched_file_pairs(dsa_folder, mask_folder):
    """
    获取DSA文件和mask文件的匹配对
    只返回文件名（不含扩展名）完全匹配的文件对

    参数:
        dsa_folder: DSA序列文件夹路径
        mask_folder: Mask文件夹路径

    返回:
        matched_pairs: 匹配的文件对列表，每个元素为(dsa_filename, mask_info)
        dsa_only: 仅有DSA没有mask的文件列表
        mask_only: 仅有mask没有DSA的文件列表
    """
    # 获取DSA文件名（不含扩展名）和完整文件名
    dsa_files_dict = {}
    for file in os.listdir(dsa_folder):
        if file.endswith('.dcm') or ('.' not in file):
            name_without_ext = get_filename_without_ext(file)
            dsa_files_dict[name_without_ext] = file

    # 获取mask文件信息
    mask_files_dict = {}
    for file in os.listdir(mask_folder):
        file_path = os.path.join(mask_folder, file)
        if os.path.isfile(file_path):
            name_without_ext = os.path.splitext(file)[0] if '.' in file else file
            extension = os.path.splitext(file)[1] if '.' in file else ''
            mask_files_dict[name_without_ext] = {
                'original_path': file_path,
                'original_filename': file,
                'extension': extension
            }

    # 找出匹配的文件对
    matched_names = set(dsa_files_dict.keys()) & set(mask_files_dict.keys())
    dsa_only_names = set(dsa_files_dict.keys()) - set(mask_files_dict.keys())
    mask_only_names = set(mask_files_dict.keys()) - set(dsa_files_dict.keys())

    # 构建匹配对列表
    matched_pairs = []
    for name in sorted(matched_names):
        matched_pairs.append((dsa_files_dict[name], mask_files_dict[name]))

    # 构建仅有DSA的列表
    dsa_only = [dsa_files_dict[name] for name in sorted(dsa_only_names)]

    # 构建仅有mask的列表
    mask_only = [mask_files_dict[name]['original_filename'] for name in sorted(mask_only_names)]

    return matched_pairs, dsa_only, mask_only


def get_filename_without_ext(file_path):
    """
    获取文件名（不含扩展名）
    """
    filename = os.path.basename(file_path)
    # 移除已知扩展名
    if filename.lower().endswith('.dcm'):
        return filename[:-4]
    elif '.' in filename:
        return filename.rsplit('.', 1)[0]
    else:
        return filename


def add_suffix_to_filename(filename_without_ext, suffix="_0"):
    """
    给文件名添加后缀

    参数:
        filename_without_ext: 原始文件名（不含扩展名）
        suffix: 要添加的后缀，默认"_0"
    """
    return f"{filename_without_ext}{suffix}"


def move_and_rename_mask(mask_info, new_filename_without_ext, target_folder):
    """
    移动并重命名mask文件到目标文件夹，保持原始文件不变

    参数:
        mask_info: mask文件信息字典
        new_filename_without_ext: 新的文件名（不含扩展名）
        target_folder: 目标文件夹路径
    """
    try:
        # 创建目标文件夹
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            logger.info(f"创建mask输出文件夹: {target_folder}")

        original_path = mask_info['original_path']
        extension = mask_info['extension']

        # 构建新的文件名
        if extension:
            new_filename = f"{new_filename_without_ext}{extension}"
        else:
            new_filename = new_filename_without_ext

        new_path = os.path.join(target_folder, new_filename)

        # 检查目标文件是否已存在
        if os.path.exists(new_path):
            logger.info(f"Mask文件已存在，跳过: {new_filename}")
            return True

        # 复制文件到新位置并重命名，保持原始文件不变
        shutil.copy2(original_path, new_path)
        logger.info(f"移动并重命名mask: {mask_info['original_filename']} -> {new_filename}")
        return True

    except Exception as e:
        logger.error(f"移动mask文件时出错: {str(e)}")
        return False


def process_matched_dsa_sequences(dsa_folder, mask_folder, output_base_folder, mask_output_folder,
                                  projection_type=ProjectionType.MIN,
                                  time_fraction=0.5, filename_suffix="_0"):
    """
    仅处理文件名匹配的DSA序列和mask文件对

    参数:
        dsa_folder: DSA序列文件夹路径
        mask_folder: Mask文件夹路径
        output_base_folder: DSA投影输出基础文件夹路径
        mask_output_folder: Mask输出文件夹路径
        projection_type: 投影类型 (ProjectionType.MIN 或 ProjectionType.MEAN)
        time_fraction: 时间截取比例
        filename_suffix: 文件名后缀，默认"_0"
    """

    # 创建DICOM和PNG输出子文件夹
    dicom_output_folder = os.path.join(output_base_folder, "DICOM")
    png_output_folder = os.path.join(output_base_folder, "PNG")

    for folder in [dicom_output_folder, png_output_folder, mask_output_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"创建输出文件夹: {folder}")

    # 获取匹配的文件对
    logger.info("正在分析文件匹配情况...")
    matched_pairs, dsa_only, mask_only = get_matched_file_pairs(dsa_folder, mask_folder)

    logger.info(f"文件匹配分析结果:")
    logger.info(f"  - 匹配的文件对: {len(matched_pairs)} 对")
    logger.info(f"  - 仅有DSA无mask: {len(dsa_only)} 个")
    logger.info(f"  - 仅有mask无DSA: {len(mask_only)} 个")

    if not matched_pairs:
        logger.error("没有找到任何匹配的DSA-mask文件对，无法继续处理")
        return

    # 显示不匹配的文件
    if dsa_only:
        logger.info("未处理的DSA文件（无对应mask）:")
        for file in dsa_only[:10]:
            logger.info(f"  - {file}")
        if len(dsa_only) > 10:
            logger.info(f"  ... 还有 {len(dsa_only) - 10} 个文件")

    if mask_only:
        logger.info("未处理的mask文件（无对应DSA）:")
        for file in mask_only[:10]:
            logger.info(f"  - {file}")
        if len(mask_only) > 10:
            logger.info(f"  ... 还有 {len(mask_only) - 10} 个文件")

    # 处理统计
    processed_count = 0
    skipped_count = 0
    dicom_saved_count = 0
    png_saved_count = 0
    mask_moved_count = 0
    error_count = 0

    logger.info(f"开始处理 {len(matched_pairs)} 对匹配的文件...")
    logger.info(f"投影类型: {projection_type.value}")
    logger.info(f"文件名后缀: {filename_suffix}")
    logger.info(f"数据处理原则:")
    logger.info(f"  - 仅处理文件名匹配的DSA-mask对")
    logger.info(f"  - DICOM: 保持原始数据类型和数值范围")
    logger.info(f"  - PNG: 真实灰度图像，仅做线性归一化")
    logger.info(f"  - Mask: 仅重命名和移动")

    for dsa_filename, mask_info in tqdm(matched_pairs, desc=f"处理匹配文件对"):
        dsa_path = os.path.join(dsa_folder, dsa_filename)
        dsa_name = get_filename_without_ext(dsa_filename)

        # 生成新的文件名（添加后缀）
        new_filename_without_ext = add_suffix_to_filename(dsa_name, filename_suffix)

        # 输出文件路径
        output_dicom_path = os.path.join(dicom_output_folder, new_filename_without_ext)
        output_png_path = os.path.join(png_output_folder, f"{new_filename_without_ext}.png")

        # 检查输出文件是否都已存在
        dicom_exists = os.path.exists(output_dicom_path)
        png_exists = os.path.exists(output_png_path)

        # 检查对应的mask文件是否已处理
        mask_extension = mask_info['extension']
        if mask_extension:
            mask_check_name = f"{new_filename_without_ext}{mask_extension}"
        else:
            mask_check_name = new_filename_without_ext
        mask_check_path = os.path.join(mask_output_folder, mask_check_name)
        mask_exists = os.path.exists(mask_check_path)

        if dicom_exists and png_exists and mask_exists:
            logger.info(f"所有输出文件已存在，跳过: {new_filename_without_ext}")
            skipped_count += 1
            continue

        try:
            # 读取DSA序列
            logger.info(f"处理匹配对: {dsa_filename} <-> {mask_info['original_filename']}")
            logger.info(f"新文件名: {new_filename_without_ext}")

            ds = pydicom.dcmread(dsa_path)
            pixel_array = ds.pixel_array
            logger.info(f"DSA原始形状: {pixel_array.shape}, 数据类型: {pixel_array.dtype}")
            logger.info(f"DSA数据范围: [{np.min(pixel_array)}, {np.max(pixel_array)}]")

            if len(pixel_array.shape) == 3:
                # 3D时间序列数据 - 进行时间投影
                projection_result, total_frames, frames_to_use = apply_time_projection(
                    pixel_array, projection_type, time_fraction
                )

                logger.info(f"时间序列: {total_frames}帧, 使用前{frames_to_use}帧")
                logger.info(f"投影后形状: {projection_result.shape}, 数据类型: {projection_result.dtype}")
                logger.info(f"投影数据范围: [{np.min(projection_result)}, {np.max(projection_result)}]")
                logger.info(f"投影统计 - 均值: {np.mean(projection_result):.2f}, "
                            f"标准差: {np.std(projection_result):.2f}")

                # 保存DICOM格式 - 保持原始数据
                if not dicom_exists:
                    if save_as_dicom(projection_result, ds, output_dicom_path,
                                     projection_type.value):
                        logger.info(f"✓ 保存DICOM: {new_filename_without_ext}")
                        dicom_saved_count += 1
                    else:
                        logger.error(f"✗ DICOM保存失败: {new_filename_without_ext}")

                # 保存PNG格式 - 灰度图像
                if not png_exists:
                    if save_as_png_grayscale(projection_result, output_png_path):
                        logger.info(f"✓ 保存PNG: {new_filename_without_ext}.png")
                        png_saved_count += 1
                    else:
                        logger.error(f"✗ PNG保存失败: {new_filename_without_ext}.png")

                processed_count += 1

            elif len(pixel_array.shape) == 2:
                # 2D图像，直接使用
                logger.info(f"2D图像，直接保存")

                # 保存DICOM格式
                if not dicom_exists:
                    if save_as_dicom(pixel_array, ds, output_dicom_path, "2D原始图像"):
                        logger.info(f"✓ 保存DICOM: {new_filename_without_ext}")
                        dicom_saved_count += 1
                    else:
                        logger.error(f"✗ DICOM保存失败: {new_filename_without_ext}")

                # 保存PNG格式
                if not png_exists:
                    if save_as_png_grayscale(pixel_array, output_png_path):
                        logger.info(f"✓ 保存PNG: {new_filename_without_ext}.png")
                        png_saved_count += 1
                    else:
                        logger.error(f"✗ PNG保存失败: {new_filename_without_ext}.png")

                processed_count += 1

            else:
                logger.error(f"不支持的图像维度: {pixel_array.shape}")
                error_count += 1
                continue

            # 处理对应的mask文件（移动和重命名）
            if move_and_rename_mask(mask_info, new_filename_without_ext, mask_output_folder):
                logger.info(f"✓ 处理Mask: {new_filename_without_ext}")
                mask_moved_count += 1

        except Exception as e:
            logger.error(f"处理文件对 {dsa_filename} 时出错: {str(e)}")
            error_count += 1
            continue

    # 输出处理总结
    logger.info("=" * 60)
    logger.info(f"{projection_type.value}处理完成!")
    logger.info(f"文件匹配统计:")
    logger.info(f"  - 总匹配对: {len(matched_pairs)} 对")
    logger.info(f"  - 成功处理: {processed_count} 对")
    logger.info(f"  - 跳过(已存在): {skipped_count} 对")
    logger.info(f"  - 处理失败: {error_count} 对")
    logger.info(f"输出文件统计:")
    logger.info(f"  - DICOM保存: {dicom_saved_count} 个")
    logger.info(f"  - PNG保存: {png_saved_count} 个")
    logger.info(f"  - Mask移动: {mask_moved_count} 个")
    logger.info(f"未处理文件:")
    logger.info(f"  - DSA无mask: {len(dsa_only)} 个")
    logger.info(f"  - Mask无DSA: {len(mask_only)} 个")
    logger.info(f"处理参数:")
    logger.info(f"  - 投影类型: {projection_type.value}")
    logger.info(f"  - 文件名后缀: {filename_suffix}")
    logger.info(f"  - 时间截取: 前 {time_fraction * 100:.0f}%")
    logger.info(f"输出路径:")
    logger.info(f"  - DICOM: {dicom_output_folder}")
    logger.info(f"  - PNG: {png_output_folder}")
    logger.info(f"  - Mask: {mask_output_folder}")
    logger.info("=" * 60)


def check_file_matching_status(dsa_folder, mask_folder):
    """
    详细检查DSA文件和mask文件的匹配状态
    """
    logger.info("=" * 60)
    logger.info("文件匹配状态详细检查")
    logger.info("=" * 60)

    matched_pairs, dsa_only, mask_only = get_matched_file_pairs(dsa_folder, mask_folder)

    logger.info(f"DSA文件夹: {dsa_folder}")
    logger.info(f"Mask文件夹: {mask_folder}")
    logger.info(f"匹配的文件对数量: {len(matched_pairs)}")
    logger.info(f"仅有DSA的文件数量: {len(dsa_only)}")
    logger.info(f"仅有Mask的文件数量: {len(mask_only)}")

    if matched_pairs:
        logger.info(f"前5对匹配的文件:")
        for i, (dsa_file, mask_info) in enumerate(matched_pairs[:5]):
            logger.info(f"  {i + 1}. {dsa_file} <-> {mask_info['original_filename']}")

    if dsa_only:
        logger.info(f"未匹配的DSA文件（前10个）:")
        for file in dsa_only[:10]:
            logger.info(f"  - {file}")

    if mask_only:
        logger.info(f"未匹配的Mask文件（前10个）:")
        for file in mask_only[:10]:
            logger.info(f"  - {file}")

    return matched_pairs, dsa_only, mask_only


def preview_output_files(output_base_folder, mask_output_folder, num_files=10):
    """
    预览输出文件
    """
    logger.info("=" * 60)
    logger.info("预览输出文件:")

    # 检查DICOM输出文件
    dicom_folder = os.path.join(output_base_folder, "DICOM")
    if os.path.exists(dicom_folder):
        dicom_files = [f for f in os.listdir(dicom_folder) if '.' not in f]
        logger.info(f"DICOM输出: {dicom_folder}")
        logger.info(f"文件数量: {len(dicom_files)}")
        if dicom_files:
            logger.info(f"前{min(num_files, len(dicom_files))}个文件:")
            for file in dicom_files[:num_files]:
                file_path = os.path.join(dicom_folder, file)
                file_size = os.path.getsize(file_path) / 1024
                logger.info(f"  - {file} ({file_size:.1f} KB)")

    # 检查PNG输出文件
    png_folder = os.path.join(output_base_folder, "PNG")
    if os.path.exists(png_folder):
        png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]
        logger.info(f"PNG输出: {png_folder}")
        logger.info(f"文件数量: {len(png_files)}")
        if png_files:
            logger.info(f"前{min(num_files, len(png_files))}个文件:")
            for file in png_files[:num_files]:
                file_path = os.path.join(png_folder, file)
                file_size = os.path.getsize(file_path) / 1024
                logger.info(f"  - {file} ({file_size:.1f} KB)")

    # 检查Mask输出文件
    if os.path.exists(mask_output_folder):
        mask_files = os.listdir(mask_output_folder)
        logger.info(f"Mask输出: {mask_output_folder}")
        logger.info(f"文件数量: {len(mask_files)}")
        if mask_files:
            logger.info(f"前{min(num_files, len(mask_files))}个文件:")
            for file in mask_files[:num_files]:
                file_path = os.path.join(mask_output_folder, file)
                file_size = os.path.getsize(file_path) / 1024
                logger.info(f"  - {file} ({file_size:.1f} KB)")


if __name__ == "__main__":
    # ========== 配置参数 ==========

    # 输入文件夹路径
    dsa_folder = r"D:\med_data\ANY\0"  # DSA序列文件夹
    mask_folder = r"D:\med_data\ANY\mask2"  # 原始Mask文件夹

    # 输出文件夹路径
    output_base_folder = r"D:\med_data\multi\preprocess\MEAN"  # DSA投影输出基础文件夹
    mask_output_folder = r"D:\med_data\multi\preprocess\MEAN_mask"  # Mask输出文件夹

    # 投影类型选择 - 在这里修改！！！
    # ProjectionType.MIN  -> 最小值投影
    # ProjectionType.MEAN -> 平均值投影
    selected_projection = ProjectionType.MEAN  # 修改这里选择投影类型

    # 文件名后缀 - 在这里修改！！！
    filename_suffix = "_1"  # 默认添加"_0"后缀

    # 时间截取比例（默认0.5表示前50%时间点）
    time_fraction = 0.6

    # 其他选项
    detailed_check = True  # 是否进行详细文件匹配检查
    preview_output = True  # 是否预览输出结果

    # ========== 开始处理 ==========

    logger.info("=" * 60)
    logger.info("DSA序列时间投影处理程序 (仅处理匹配文件对)")
    logger.info("=" * 60)
    logger.info(f"输入:")
    logger.info(f"  - DSA文件夹: {dsa_folder}")
    logger.info(f"  - Mask文件夹: {mask_folder}")
    logger.info(f"输出:")
    logger.info(f"  - DICOM: {output_base_folder}\\DICOM")
    logger.info(f"  - PNG: {output_base_folder}\\PNG")
    logger.info(f"  - Mask: {mask_output_folder}")
    logger.info(f"处理规则:")
    logger.info(f"  - 仅处理文件名完全匹配的DSA-mask对")
    logger.info(f"  - 无匹配mask的DSA序列将被跳过")
    logger.info(f"  - 无匹配DSA的mask文件将被忽略")
    logger.info(f"参数:")
    logger.info(f"  - 投影类型: {selected_projection.value}")
    logger.info(f"  - 文件名后缀: {filename_suffix}")
    logger.info(f"  - 时间截取: 前 {time_fraction * 100:.0f}%")

    # 文件命名示例
    example_dsa = "ANY_001_0"
    example_new = add_suffix_to_filename(example_dsa, filename_suffix)
    logger.info(f"命名示例:")
    logger.info(f"  输入: {example_dsa} (DSA) + {example_dsa}.tif (Mask)")
    logger.info(f"  输出: {example_new} (DICOM) + {example_new}.png (PNG) + {example_new}.tif (Mask)")

    # 详细文件匹配检查
    if detailed_check:
        matched_pairs, dsa_only, mask_only = check_file_matching_status(dsa_folder, mask_folder)

        if not matched_pairs:
            logger.error("没有找到任何匹配的文件对，程序终止")
            exit()

    try:
        # 执行处理
        process_matched_dsa_sequences(
            dsa_folder=dsa_folder,
            mask_folder=mask_folder,
            output_base_folder=output_base_folder,
            mask_output_folder=mask_output_folder,
            projection_type=selected_projection,
            time_fraction=time_fraction,
            filename_suffix=filename_suffix
        )

        # 预览输出结果
        if preview_output:
            preview_output_files(output_base_folder, mask_output_folder)

        logger.info("所有处理完成！")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()