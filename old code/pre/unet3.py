import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# 标准卷积编码器块（替换 MambaEncoderBlock）
class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0.0 else None
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        skip = x  # 保存跳跃连接
        x = self.pool(x)
        return skip, x

# 多尺度卷积块
class MultiScaleConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        # 调整输出通道数的卷积层
        self.adjust_conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(x)))
        fused_features = torch.cat([conv1, conv2], dim=1)  # 特征融合
        fused_features = self.adjust_conv(fused_features)  # 调整通道数
        return fused_features

# 解码器块
class DecoderBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x, skip):
        x = self.up(x)
        # 调整尺寸匹配
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

# 多尺度 U-Net 3D（去掉 Mamba 块）
class MultiScale_UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_filters_start=32, dropout_rate=0.4):
        super().__init__()
        # 编码器：使用标准卷积块
        self.encoder1 = ConvEncoderBlock(in_channels, num_filters_start, dropout_rate)
        self.encoder2 = ConvEncoderBlock(num_filters_start, num_filters_start * 2, dropout_rate)
        self.encoder3 = ConvEncoderBlock(num_filters_start * 2, num_filters_start * 4, dropout_rate)
        self.encoder4 = ConvEncoderBlock(num_filters_start * 4, num_filters_start * 8, dropout_rate)

        # 多尺度卷积块（瓶颈层）
        self.multi_scale_conv = MultiScaleConvBlock1(num_filters_start * 8, num_filters_start * 8)

        # 解码器
        self.decoder1 = DecoderBlock1(num_filters_start * 8, num_filters_start * 8, dropout_rate)
        self.decoder2 = DecoderBlock1(num_filters_start * 8, num_filters_start * 4, dropout_rate)
        self.decoder3 = DecoderBlock1(num_filters_start * 4, num_filters_start * 2, dropout_rate)
        self.decoder4 = DecoderBlock1(num_filters_start * 2, num_filters_start, dropout_rate)

        # 输出层
        self.final_conv = nn.Conv3d(num_filters_start, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器路径
        skip1, x1 = self.encoder1(x)
        skip2, x2 = self.encoder2(x1)
        skip3, x3 = self.encoder3(x2)
        skip4, x4 = self.encoder4(x3)

        # 多尺度卷积块（瓶颈层）
        x5 = self.multi_scale_conv(x4)

        # 解码器路径
        x = self.decoder1(x5, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        # 残差连接
        x = x + skip1

        # 输出层
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class MultiScale_UNet3D_3(nn.Module):
    def __init__(self, in_channels=1, num_filters_start=32, dropout_rate=0.4):
        super().__init__()

        self.encoder1 = ConvEncoderBlock(in_channels, num_filters_start, dropout_rate)
        self.encoder2 = ConvEncoderBlock(num_filters_start, num_filters_start * 2, dropout_rate)
        self.encoder3 = ConvEncoderBlock(num_filters_start * 2, num_filters_start * 4, dropout_rate)

        self.multi_scale_conv = MultiScaleConvBlock1(num_filters_start * 4, num_filters_start * 4)

        self.decoder1 = DecoderBlock1(num_filters_start * 4, num_filters_start * 4, dropout_rate)
        self.decoder2 = DecoderBlock1(num_filters_start * 4, num_filters_start * 2, dropout_rate)
        self.decoder3 = DecoderBlock1(num_filters_start * 2, num_filters_start, dropout_rate)

        self.final_conv = nn.Conv3d(num_filters_start, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        skip1, x1 = self.encoder1(x)
        skip2, x2 = self.encoder2(x1)
        skip3, x3 = self.encoder3(x2)

        x4 = self.multi_scale_conv(x3)

        x = self.decoder1(x4, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)

        x = x + skip1

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class L_EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        # 从原来的2个卷积减少到1个卷积
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0.0 else None
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        # 只有一个卷积操作 + ReLU + BN
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        skip = x  # 保存跳跃连接
        x = self.pool(x)
        return skip, x

class L_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # 从原来的3个卷积减少到2个卷积
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x, skip):
        x = self.up(x)
        # 调整尺寸匹配
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # 移除了第三个卷积
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class LightMultiScale_UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_filters_start=32, dropout_rate=0.4):
        super().__init__()
        # 编码器
        self.encoder1 = L_EncoderBlock(in_channels, num_filters_start, dropout_rate)
        self.encoder2 = L_EncoderBlock(num_filters_start, num_filters_start * 2, dropout_rate)
        self.encoder3 = L_EncoderBlock(num_filters_start * 2, num_filters_start * 4, dropout_rate)
        self.encoder4 = L_EncoderBlock(num_filters_start * 4, num_filters_start * 8, dropout_rate)

        # 瓶颈层
        self.multi_scale_conv = MultiScaleConvBlock1(num_filters_start * 8, num_filters_start * 8)

        # 解码器 - 正确的通道递减
        self.decoder1 = L_DecoderBlock(num_filters_start * 8, num_filters_start * 4, dropout_rate)  # 256→128
        self.decoder2 = L_DecoderBlock(num_filters_start * 4, num_filters_start * 2, dropout_rate)  # 128→64
        self.decoder3 = L_DecoderBlock(num_filters_start * 2, num_filters_start, dropout_rate)      # 64→32
        self.decoder4 = L_DecoderBlock(num_filters_start, num_filters_start, dropout_rate)          # 32→32 (最后一层保持)

        # 输出层
        self.final_conv = nn.Conv3d(num_filters_start, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1, x1 = self.encoder1(x)
        skip2, x2 = self.encoder2(x1)
        skip3, x3 = self.encoder3(x2)
        skip4, x4 = self.encoder4(x3)

        x5 = self.multi_scale_conv(x4)

        x = self.decoder1(x5, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = x + skip1  # 残差连接
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

class LightMultiScale_UNet3D_3(nn.Module):
    def __init__(self, in_channels=1, num_filters_start=32, dropout_rate=0.4):
        super().__init__()
        # 编码器
        self.encoder1 = L_EncoderBlock(in_channels, num_filters_start, dropout_rate)
        self.encoder2 = L_EncoderBlock(num_filters_start, num_filters_start * 2, dropout_rate)
        self.encoder3 = L_EncoderBlock(num_filters_start * 2, num_filters_start * 4, dropout_rate)

        # 瓶颈层
        self.multi_scale_conv = MultiScaleConvBlock1(num_filters_start * 4, num_filters_start * 4)

        # 解码器 - 正确的通道递减
        self.decoder2 = L_DecoderBlock(num_filters_start * 4, num_filters_start * 2, dropout_rate)  # 128→64
        self.decoder3 = L_DecoderBlock(num_filters_start * 2, num_filters_start, dropout_rate)      # 64→32
        self.decoder4 = L_DecoderBlock(num_filters_start, num_filters_start, dropout_rate)          # 32→32

        # 输出层
        self.final_conv = nn.Conv3d(num_filters_start, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1, x1 = self.encoder1(x)
        skip2, x2 = self.encoder2(x1)
        skip3, x3 = self.encoder3(x2)

        x5 = self.multi_scale_conv(x3)

        x = self.decoder2(x5, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = x + skip1  # 残差连接
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x
