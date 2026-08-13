import os
import numpy as np
import nibabel as nib
import re


def safe_flip(data, axis='lr'):
    """
    安全翻转：沿指定轴翻转3D图像数据

    参数:
        data: 3D numpy数组 [D, H, W]
        axis: 翻转轴
            'lr' 或 'x' - 左右翻转 (默认)
            'ud' 或 'y' - 上下翻转
            'ap' 或 'z' - 前后翻转
    """
    if axis in ['lr', 'x']:
        # 左右翻转 - 沿宽度维度翻转
        flipped_data = np.flip(data, axis=2)
    elif axis in ['ud', 'y']:
        # 上下翻转 - 沿高度维度翻转
        flipped_data = np.flip(data, axis=1)
    elif axis in ['ap', 'z']:
        # 前后翻转 - 沿深度维度翻转
        flipped_data = np.flip(data, axis=0)
    else:
        raise ValueError(f"不支持的翻转轴: {axis}。支持的值: 'lr', 'x', 'ud', 'y', 'ap', 'z'")

    return flipped_data


def extract_subject_id(filename):
    """
    从文件名中提取病历号

    参数:
        filename: 如 "ANY_204_1_T1.nii.gz" 或 "s201_T1.nii.gz"

    返回:
        提取的数字ID，如果无法提取则返回None
    """
    # 尝试匹配 ANY_数字_数字 格式
    pattern1 = r'ANY_(\d+)_\d+_T1\.nii\.gz'
    match1 = re.match(pattern1, filename)
    if match1:
        return int(match1.group(1))

    # 尝试匹配 s数字 格式
    pattern2 = r's(\d+)_T1\.nii\.gz'
    match2 = re.match(pattern2, filename)
    if match2:
        return int(match2.group(1))

    # 尝试匹配纯数字格式
    pattern3 = r'^(\d+)_T1\.nii\.gz'
    match3 = re.match(pattern3, filename)
    if match3:
        return int(match3.group(1))

    return None


def generate_new_subject_id(original_id, add_value=5000):
    """
    生成新的病历号：原病历号 + add_value

    参数:
        original_id: 原始病历号（整数）
        add_value: 要加上的数值，默认2500

    返回:
        新的病历号（整数）
    """
    return original_id + add_value


def format_subject_id(subject_id):
    """
    格式化病历号，例如：204 -> "204"，也可以根据需要添加前缀

    参数:
        subject_id: 病历号（整数）

    返回:
        格式化的字符串
    """
    return f"{subject_id}"


def process_flip_augmentation(t1w_dir, t2w_dir, output_t1_dir, output_t2_dir, axis='lr'):
    """处理翻转数据增强，病历号加2500后保存"""

    file_pairs = get_matching_file_pairs(t1w_dir, t2w_dir)
    os.makedirs(output_t1_dir, exist_ok=True)
    os.makedirs(output_t2_dir, exist_ok=True)

    print(f"找到 {len(file_pairs)} 对匹配的文件")
    print(f"翻转轴: {axis}")
    print(f"病历号增加: 2500")
    print()

    processed_count = 0
    skipped_count = 0

    for t1w_file, t2w_file in file_pairs:
        try:
            # 从文件名提取原始病历号
            original_id = extract_subject_id(t1w_file)

            if original_id is None:
                print(f"警告: 无法从文件名提取病历号 {t1w_file}，跳过处理")
                skipped_count += 1
                continue

            # 生成新的病历号（原病历号 + 2500）
            new_id = generate_new_subject_id(original_id, 5000)

            # 加载图像
            t1w_img = nib.load(os.path.join(t1w_dir, t1w_file))
            t1w_data = t1w_img.get_fdata().astype(np.float32)

            t2w_img = nib.load(os.path.join(t2w_dir, t2w_file))
            t2w_data = t2w_img.get_fdata().astype(np.float32)

            original_shape = t1w_data.shape
            print(f"处理: {t1w_file}")
            print(f"  原始病历号: {original_id}, 新病历号: {new_id}")


            # 应用翻转
            flipped_t1 = safe_flip(t1w_data, axis=axis)
            flipped_t2 = safe_flip(t2w_data, axis=axis)

            # 验证形状
            assert flipped_t1.shape == original_shape, "形状改变！"

            # 格式化病历号
            new_id_str = format_subject_id(new_id)

            # 保存翻转后的图像，使用新的病历号
            t1_output_path = os.path.join(output_t1_dir, f"{new_id_str}_T1.nii.gz")
            t2_output_path = os.path.join(output_t2_dir, f"{new_id_str}_T2.nii.gz")

            nib.save(nib.Nifti1Image(flipped_t1, t1w_img.affine, t1w_img.header),
                     t1_output_path)
            nib.save(nib.Nifti1Image(flipped_t2, t2w_img.affine, t2w_img.header),
                     t2_output_path)

            print(f"  生成: {new_id_str}_T1.nii.gz, {new_id_str}_T2.nii.gz (沿{axis}轴翻转)")
            print()
            processed_count += 1

        except Exception as e:
            print(f"错误: 处理 {t1w_file} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n处理完成!")
    print(f"  成功处理: {processed_count} 对图像")
    print(f"    T1: {output_t1_dir}")
    print(f"    T2: {output_t2_dir}")


def get_matching_file_pairs(t1w_dir, t2w_dir):
    """获取匹配的文件对"""
    t1w_files = [f for f in os.listdir(t1w_dir) if f.endswith('_T1.nii.gz')]
    file_pairs = []

    for t1w_file in t1w_files:
        # 从T1文件名提取病历号
        subject_id = extract_subject_id(t1w_file)

        if subject_id is None:
            print(f"警告: 无法从文件名提取病历号 {t1w_file}，将尝试其他匹配方式")
            # 后备方案：使用原逻辑
            subject_id = t1w_file.split('_')[0]
            t2w_file = f"{subject_id}_T2.nii.gz"
        else:
            # 使用提取的数字ID构建T2文件名
            t2w_file = f"{subject_id}_T2.nii.gz"

        if os.path.exists(os.path.join(t2w_dir, t2w_file)):
            file_pairs.append((t1w_file, t2w_file))
        else:
            print(f"警告: 找不到对应的T2文件 {t2w_file}")

    return file_pairs


if __name__ == "__main__":
    process_flip_augmentation(
        t1w_dir='D:\\med_data\\MR\\T12W2\\1',
        t2w_dir='D:\\med_data\\MR\\T12W2\\2',
        output_t1_dir='D:\\med_data\\MR\\train_1',
        output_t2_dir='D:\\med_data\\MR\\train_2',
        axis='ap'  # 默认左右翻转，可改为 'ud' 或 'ap'
    )