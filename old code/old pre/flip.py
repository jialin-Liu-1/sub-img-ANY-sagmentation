import os
import numpy as np
import nibabel as nib


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


def process_flip_augmentation(t1w_dir, t2w_dir, output_t1_dir, output_t2_dir, axis='lr', start_index=201):
    """处理翻转数据增强"""

    file_pairs = get_matching_file_pairs(t1w_dir, t2w_dir)
    os.makedirs(output_t1_dir, exist_ok=True)
    os.makedirs(output_t2_dir, exist_ok=True)

    processed_count = 0

    for t1w_file, t2w_file in file_pairs:
        try:
            # 加载图像
            t1w_img = nib.load(os.path.join(t1w_dir, t1w_file))
            t1w_data = t1w_img.get_fdata().astype(np.float32)

            t2w_img = nib.load(os.path.join(t2w_dir, t2w_file))
            t2w_data = t2w_img.get_fdata().astype(np.float32)

            original_shape = t1w_data.shape
            print(f"处理: {t1w_file}, 形状: {original_shape}")

            # 应用翻转
            flipped_t1 = safe_flip(t1w_data, axis=axis)
            flipped_t2 = safe_flip(t2w_data, axis=axis)

            # 验证形状
            assert flipped_t1.shape == original_shape, "形状改变！"

            # 保存
            new_id = f"s{start_index + processed_count:03d}"
            nib.save(nib.Nifti1Image(flipped_t1, t1w_img.affine),
                     os.path.join(output_t1_dir, f"{new_id}_T1.nii.gz"))
            nib.save(nib.Nifti1Image(flipped_t2, t2w_img.affine),
                     os.path.join(output_t2_dir, f"{new_id}_T2.nii.gz"))

            print(f"生成: {new_id}_T1.nii.gz (沿{axis}轴翻转)")
            processed_count += 1

        except Exception as e:
            print(f"错误: {str(e)}")
            continue

    print(f"完成! 生成 {processed_count} 对图像")


# 辅助函数
def get_matching_file_pairs(t1w_dir, t2w_dir):
    """获取匹配的文件对"""
    t1w_files = [f for f in os.listdir(t1w_dir) if f.endswith('_T1.nii.gz')]
    file_pairs = []

    for t1w_file in t1w_files:
        subject_id = t1w_file.split('_')[0]
        t2w_file = f"{subject_id}_T2.nii.gz"
        if os.path.exists(os.path.join(t2w_dir, t2w_file)):
            file_pairs.append((t1w_file, t2w_file))

    return file_pairs


if __name__ == "__main__":
    process_flip_augmentation(
        t1w_dir='D:\\med_data\\MR\\T12W2\\1',
        t2w_dir='D:\\med_data\\MR\\T12W2\\2',
        output_t1_dir='D:\\med_data\\MR\\train_1',
        output_t2_dir='D:\\med_data\\MR\\train_2',
        axis='ap',  # 默认左右翻转，可改为 'ud' 或 'ap'
        start_index=201
    )