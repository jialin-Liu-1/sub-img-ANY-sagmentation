import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightUNet4(nn.Module):
    """
    Lightweight U-Net Segmentation Network (4-layer Encoder-Decoder structure)
    Improvements:
    1. All pooling methods changed to Average Pooling
    2. Encoder uses LeakyReLU activation functions throughout
    3. 4-layer structure increases network depth
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 dropout_rate: float = 0.0,
                 negative_slope: float = 0.01):
        super().__init__()

        # ========== Encoder Section (4 Layers) ==========
        self.encoder1 = self._make_encoder_block(in_channels, base_channels, dropout_rate, negative_slope)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels * 2, dropout_rate, negative_slope)
        self.encoder3 = self._make_encoder_block(base_channels * 2, base_channels * 4, dropout_rate, negative_slope)
        self.encoder4 = self._make_encoder_block(base_channels * 4, base_channels * 8, dropout_rate, negative_slope)

        # ========== Bottleneck Layer ==========
        # Input comes from encoder4 output (base_channels*8)
        # Output remains base_channels*8, then passed to decoder4 via skip connection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

        # ========== Decoder Section (4 Layers) ==========
        # decoder4: Input = bottleneck (base_channels*8) + encoder4 (base_channels*8) -> Output base_channels*4
        self.decoder4 = self._make_decoder_block(base_channels * 16, base_channels * 4, dropout_rate, negative_slope)

        # decoder3: Input = decoder4 output (base_channels*4) + encoder3 (base_channels*4) -> Output base_channels*2
        self.decoder3 = self._make_decoder_block(base_channels * 8, base_channels * 2, dropout_rate, negative_slope)

        # decoder2: Input = decoder3 output (base_channels*2) + encoder2 (base_channels*2) -> Output base_channels
        self.decoder2 = self._make_decoder_block(base_channels * 4, base_channels, dropout_rate, negative_slope)

        # decoder1: Input = decoder2 output (base_channels) + encoder1 (base_channels) -> Output base_channels
        self.decoder1 = self._make_decoder_block(base_channels * 2, base_channels, dropout_rate, negative_slope)

        # ========== Output Layer ==========
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _make_encoder_block(self, in_channels, out_channels, dropout_rate, negative_slope):
        """Creates an encoder block (using Average Pooling and LeakyReLU)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),

            nn.AvgPool2d(2),  # Average Pooling
            nn.Dropout2d(dropout_rate)
        )

    def _make_decoder_block(self, in_channels, out_channels, dropout_rate, negative_slope):
        """Creates a decoder block"""
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
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass
        Parameters:
            x: [B, 1, H, W] DSA image processed with attention mask
        Returns:
            segmentation: [B, 1, H, W] Aneurysm segmentation result
        """
        # ========== Encoder Path (4 Layers) ==========
        enc1 = self.encoder1(x)  # [B, base_channels, H/2, W/2]
        enc2 = self.encoder2(enc1)  # [B, base_channels*2, H/4, W/4]
        enc3 = self.encoder3(enc2)  # [B, base_channels*4, H/8, W/8]
        enc4 = self.encoder4(enc3)  # [B, base_channels*8, H/16, W/16]

        # ========== Bottleneck ==========
        bottleneck = self.bottleneck(enc4)  # [B, base_channels*8, H/16, W/16]

        # ========== Decoder Path (4 layers, with skip connections) ==========
        # Decoder Level 4
        dec4 = self.decoder4(torch.cat([bottleneck, enc4], dim=1))  # [B, base_channels*4, H/8, W/8]

        # Decoder Level 3
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))  # [B, base_channels*2, H/4, W/4]

        # Decoder Level 2
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))  # [B, base_channels, H/2, W/2]

        # Decoder Level 1
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))  # [B, base_channels, H, W]

        # ========== Output Segmentation ==========
        segmentation = self.output_conv(dec1)  # [B, out_channels, H, W]

        # Ensure output size matches input size
        if segmentation.shape[2:] != x.shape[2:]:
            segmentation = F.interpolate(segmentation, size=x.shape[2:],
                                         mode='bilinear', align_corners=False)

        return segmentation


def test_model4():
    """Tests the 4-layer U-Net model"""
    # Initialize model
    model = LightweightUNet4(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        dropout_rate=0.1,
        negative_slope=0.01
    )

    # Print structure
    print("4-Layer U-Net Model Structure:")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    batch_size = 2
    input_size = (512, 512)
    x = torch.randn(batch_size, 1, *input_size)

    # Forward pass
    output = model(x)

    print(f"\nInput Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")

    # Print feature map sizes for debugging
    print("\nFeature Map Size Progression:")
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
    # Run test
    model4 = test_model4()