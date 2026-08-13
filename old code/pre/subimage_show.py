import os
# 在导入其他库之前设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免一些冲突
import matplotlib.pyplot as plt

class SubImageExtractor(nn.Module):
    """
    子图提取模块
    输入: [B, 1, 512, 512]
    输出: [B, 256, 32, 32]
    """
    def __init__(self, patch_size=16, image_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 32

    def forward(self, image):
        B, C, H, W = image.shape
        p = self.patch_size
        g = self.grid_size

        patches = image.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, g * g, p, p).squeeze(1)
        sub_images = patches.view(B, g, g, p, p).permute(0, 3, 4, 1, 2).contiguous()
        sub_images = sub_images.view(B, p * p, g, g)

        return sub_images  # [B, 256, 32, 32]

# 读取图片
image_path = r"D:\med_data\ai\translate\reverse\train_PNG\ANY_60001_0.png"
img = Image.open(image_path).convert('L')  # 转换为灰度图，得到 [1, 512, 512]

# 调整大小为 512x512（如果需要）
img = img.resize((512, 512))

# 转换为 tensor 并添加 batch 和 channel 维度
img_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]

# 归一化到 [0, 1] 范围
img_tensor = img_tensor / 255.0

# 初始化子图提取器
extractor = SubImageExtractor(patch_size=16, image_size=512)

# 提取子图像
with torch.no_grad():
    sub_images = extractor(img_tensor)  # [1, 256, 32, 32]

# 打印信息
print(f"原始图片形状: {img_tensor.shape}")
print(f"子图像张量形状: {sub_images.shape}")  # 应该是 [1, 256, 32, 32]
print(f"总共有 {sub_images.shape[1]} 个子图像，每个子图像大小为 {sub_images.shape[2]}x{sub_images.shape[3]}\n")

# 打印前10个子图像的信息
print("前10个子图像的信息:")
for i in range(min(10, sub_images.shape[1])):
    sub_img = sub_images[0, i, :, :]  # [32, 32]
    print(f"子图像 {i+1}: shape={sub_img.shape}, "
          f"值范围=[{sub_img.min():.3f}, {sub_img.max():.3f}], "
          f"均值={sub_img.mean():.3f}, "
          f"标准差={sub_img.std():.3f}")

# 保存前10个子图像为图片文件（避免显示问题）
output_dir = r"D:\med_data\ai\translate\reverse\train_PNG\subimages"
os.makedirs(output_dir, exist_ok=True)

for i in range(min(10, sub_images.shape[1])):
    sub_img = sub_images[0, i, :, :].numpy()
    # 反归一化到 0-255
    sub_img_display = (sub_img * 255).astype(np.uint8)
    img_save = Image.fromarray(sub_img_display)
    img_save.save(os.path.join(output_dir, f"subimage_{i+1}.png"))
    print(f"已保存子图像 {i+1} 到: {output_dir}/subimage_{i+1}.png")

print(f"\n所有子图像已保存到: {output_dir}")

# 可选：如果需要显示图像，取消下面的注释
# try:
#     # 尝试显示图像（可能需要交互式后端）
#     fig, axes = plt.subplots(2, 5, figsize=(12, 6))
#     axes = axes.flatten()
#
#     for i in range(10):
#         sub_img = sub_images[0, i, :, :].numpy()
#         axes[i].imshow(sub_img, cmap='gray')
#         axes[i].set_title(f'Sub-image {i+1}')
#         axes[i].axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "subimages_grid.png"), dpi=150)
#     print(f"子图像网格图已保存到: {output_dir}/subimages_grid.png")
#     plt.close()
# except Exception as e:
#     print(f"无法显示图像: {e}")