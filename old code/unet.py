import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightUNet3(nn.Module):
    """
    轻量级U-Net分割网络
    改进：
    1. 池化方式全部改为平均值池化
    2. 编码器全部使用LeakyReLU激活函数
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 dropout_rate: float = 0.1,
                 negative_slope: float = 0.01):  # LeakyReLU的负斜率
        super().__init__()

        # ========== 编码器部分 ==========
        self.encoder1 = self._make_encoder_block(in_channels, base_channels, dropout_rate, negative_slope)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels * 2, dropout_rate, negative_slope)
        self.encoder3 = self._make_encoder_block(base_channels * 2, base_channels * 4, dropout_rate, negative_slope)

        # ========== 瓶颈层 ==========
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

        # ========== 解码器部分 ==========
        self.decoder3 = self._make_decoder_block(base_channels * 8, base_channels * 2, dropout_rate, negative_slope)
        self.decoder2 = self._make_decoder_block(base_channels * 4, base_channels, dropout_rate, negative_slope)
        self.decoder1 = self._make_decoder_block(base_channels * 2, base_channels, dropout_rate, negative_slope)

        # ========== 输出层 ==========
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _make_encoder_block(self, in_channels, out_channels, dropout_rate, negative_slope):
        """创建编码器块（使用平均值池化和LeakyReLU）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.AvgPool2d(2),  # 改为平均值池化
            nn.Dropout2d(dropout_rate)
        )

    def _make_decoder_block(self, in_channels, out_channels, dropout_rate, negative_slope):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Dropout2d(dropout_rate)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: [B, 1, H, W] 经过注意力mask处理的DSA图像
        返回:
            segmentation: [B, 1, H, W] 动脉瘤分割结果
        """
        # ========== 编码器路径 ==========
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # ========== 瓶颈层 ==========
        bottleneck = self.bottleneck(enc3)

        # ========== 解码器路径 ==========
        dec3 = self.decoder3(torch.cat([bottleneck, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        # ========== 输出分割 ==========
        segmentation = self.output_conv(dec1)

        # 确保输出尺寸与输入一致
        if segmentation.shape[2:] != x.shape[2:]:
            segmentation = F.interpolate(segmentation, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)

        return segmentation

class LightweightUNet4(nn.Module):
    """
    轻量级U-Net分割网络（四层编码器-解码器结构）
    改进：
    1. 池化方式全部改为平均值池化
    2. 编码器全部使用LeakyReLU激活函数
    3. 四层编码器解码器结构，增加网络深度
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 dropout_rate: float = 0.1,
                 negative_slope: float = 0.01):  # LeakyReLU的负斜率
        super().__init__()

        # ========== 编码器部分（4层） ==========
        self.encoder1 = self._make_encoder_block(in_channels, base_channels, dropout_rate, negative_slope)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels * 2, dropout_rate, negative_slope)
        self.encoder3 = self._make_encoder_block(base_channels * 2, base_channels * 4, dropout_rate, negative_slope)
        self.encoder4 = self._make_encoder_block(base_channels * 4, base_channels * 8, dropout_rate, negative_slope)

        # ========== 瓶颈层 ==========
        # 输入来自encoder4的输出 (base_channels*8)
        # 输出保持为 base_channels*8，然后通过跳跃连接传递给decoder4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

        # ========== 解码器部分（4层） ==========
        # decoder4: 输入 = bottleneck (base_channels*8) + encoder4 (base_channels*8) -> 输出 base_channels*4
        self.decoder4 = self._make_decoder_block(base_channels * 16, base_channels * 4, dropout_rate, negative_slope)

        # decoder3: 输入 = decoder4输出 (base_channels*4) + encoder3 (base_channels*4) -> 输出 base_channels*2
        self.decoder3 = self._make_decoder_block(base_channels * 8, base_channels * 2, dropout_rate, negative_slope)

        # decoder2: 输入 = decoder3输出 (base_channels*2) + encoder2 (base_channels*2) -> 输出 base_channels
        self.decoder2 = self._make_decoder_block(base_channels * 4, base_channels, dropout_rate, negative_slope)

        # decoder1: 输入 = decoder2输出 (base_channels) + encoder1 (base_channels) -> 输出 base_channels
        self.decoder1 = self._make_decoder_block(base_channels * 2, base_channels, dropout_rate, negative_slope)

        # ========== 输出层 ==========
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _make_encoder_block(self, in_channels, out_channels, dropout_rate, negative_slope):
        """创建编码器块（使用平均值池化和LeakyReLU）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.AvgPool2d(2),  # 平均值池化
            nn.Dropout2d(dropout_rate)
        )

    def _make_decoder_block(self, in_channels, out_channels, dropout_rate, negative_slope):
        """创建解码器块"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout_rate)
        )

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: [B, 1, H, W] 经过注意力mask处理的DSA图像
        返回:
            segmentation: [B, 1, H, W] 动脉瘤分割结果
        """
        # ========== 编码器路径（4层） ==========
        enc1 = self.encoder1(x)  # [B, base_channels, H/2, W/2]
        enc2 = self.encoder2(enc1)  # [B, base_channels*2, H/4, W/4]
        enc3 = self.encoder3(enc2)  # [B, base_channels*4, H/8, W/8]
        enc4 = self.encoder4(enc3)  # [B, base_channels*8, H/16, W/16]

        # ========== 瓶颈层 ==========
        bottleneck = self.bottleneck(enc4)  # [B, base_channels*8, H/16, W/16]

        # ========== 解码器路径（4层，带跳跃连接） ==========
        # 第4层解码器
        dec4 = self.decoder4(torch.cat([bottleneck, enc4], dim=1))  # [B, base_channels*4, H/8, W/8]

        # 第3层解码器
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))  # [B, base_channels*2, H/4, W/4]

        # 第2层解码器
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))  # [B, base_channels, H/2, W/2]

        # 第1层解码器
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))  # [B, base_channels, H, W]

        # ========== 输出分割 ==========
        segmentation = self.output_conv(dec1)  # [B, out_channels, H, W]

        # 确保输出尺寸与输入一致
        if segmentation.shape[2:] != x.shape[2:]:
            segmentation = F.interpolate(segmentation, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)

        return segmentation


# 为了测试模型结构，可以添加一个简单的测试函数
def test_model():
    """测试四层U-Net模型"""
    # 创建模型
    model = LightweightUNet4(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        dropout_rate=0.1,
        negative_slope=0.01
    )

    # 打印模型结构
    print("四层U-Net模型结构:")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建测试输入
    batch_size = 2
    input_size = (512, 512)
    x = torch.randn(batch_size, 1, *input_size)

    # 前向传播
    output = model(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 打印每层的特征图尺寸（用于调试）
    print("\n特征图尺寸变化:")
    with torch.no_grad():
        enc1 = model.encoder1(x)
        enc2 = model.encoder2(enc1)
        enc3 = model.encoder3(enc2)
        enc4 = model.encoder4(enc3)
        bottleneck = model.bottleneck(enc4)
        dec4 = model.decoder4(torch.cat([bottleneck, enc4], dim=1))
        dec3 = model.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = model.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = model.decoder1(torch.cat([dec2, enc1], dim=1))

        print(f"  enc1: {enc1.shape}")
        print(f"  enc2: {enc2.shape}")
        print(f"  enc3: {enc3.shape}")
        print(f"  enc4: {enc4.shape}")
        print(f"  bottleneck: {bottleneck.shape}")
        print(f"  dec4: {dec4.shape}")
        print(f"  dec3: {dec3.shape}")
        print(f"  dec2: {dec2.shape}")
        print(f"  dec1: {dec1.shape}")
        print(f"  output: {output.shape}")

    return model


if __name__ == "__main__":
    # 运行测试
    model = test_model()