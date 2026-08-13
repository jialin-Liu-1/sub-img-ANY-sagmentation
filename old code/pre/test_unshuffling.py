import torch
import torch.nn as nn
import time


class SubImageExtractor(nn.Module):
    def __init__(self, patch_size=16, image_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size

    def forward(self, image):
        B, C, H, W = image.shape
        p = self.patch_size
        g = self.grid_size

        patches = image.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, g * g, p, p).squeeze(1)
        sub_images = patches.view(B, g, g, p, p).permute(0, 3, 4, 1, 2).contiguous()
        sub_images = sub_images.view(B, p * p, g, g)
        return sub_images


class CompatiblePixelUnshuffle(nn.Module):
    def __init__(self, patch_size=16, image_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=patch_size)

        # 正确的重排索引
        # PixelUnshuffle 输出: [B, C_in * patch_size^2, H/patch_size, W/patch_size]
        # 这里 C_in = 1，所以通道数 = patch_size^2 = 256
        # 子图数量 = grid_size * grid_size = 1024
        # 但通道只有256，所以不能按子图位置重排！
        # 实际上，PixelUnshuffle 将每个 patch 的像素展开到了通道维度
        # 我们需要重排的是这256个通道的顺序

        # 创建一个 grid_size x grid_size 的索引矩阵（行优先），每个位置的值是 patch 内的像素索引
        # 但这里简化：直接创建列优先的通道顺序
        num_channels = patch_size * patch_size  # 256
        # 行优先顺序
        row_major = torch.arange(num_channels)
        # 转换为列优先（对于 grid_size x grid_size 的网格）
        # 将 0..255 重新排列为列优先
        row_major_2d = row_major.reshape(self.grid_size, self.grid_size)
        col_major_2d = row_major_2d.T
        col_major = col_major_2d.flatten()

        self.register_buffer('reorder_indices', col_major)

    def forward(self, image):
        out = self.unshuffle(image)  # [B, 256, 32, 32]
        return out[:, self.reorder_indices, :, :]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

batch_size = 8
x = torch.randn(batch_size, 1, 512, 512).to(device)

your_model = SubImageExtractor().to(device)
compatible_model = CompatiblePixelUnshuffle().to(device)

# 验证一致性
with torch.no_grad():
    out1 = your_model(x)
    out2 = compatible_model(x)

if torch.allclose(out1, out2, atol=1e-6):
    print("✅ 输出一致\n")
else:
    print(f"❌ 输出不一致，最大差异: {(out1 - out2).abs().max().item()}\n")

# 性能测试
num_runs = 200

torch.cuda.synchronize()
start = time.time()
for _ in range(num_runs):
    _ = your_model(x)
torch.cuda.synchronize()
your_time = (time.time() - start) / num_runs * 1000

torch.cuda.synchronize()
start = time.time()
for _ in range(num_runs):
    _ = compatible_model(x)
torch.cuda.synchronize()
compatible_time = (time.time() - start) / num_runs * 1000

print(f"Batch Size: {batch_size}")
print(f"迭代次数: {num_runs}")
print("-" * 40)
print(f"你的 SubImageExtractor:     {your_time:.3f} ms")
print(f"CompatiblePixelUnshuffle:   {compatible_time:.3f} ms")
print(f"加速比:                     {your_time / compatible_time:.2f}x")