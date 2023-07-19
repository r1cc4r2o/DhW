import torch.nn as nn
from _convBlock import ConvBlock

class UnetEncoderBlock(nn.Module):
    """Downsaple the input by a factor 2 raising the number of channels by a factor 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        shape_norm = [out_channels, 26, 189]
        self.convblock = ConvBlock(in_channels, out_channels, shape_norm)
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.convblock(x)
        pooled = self.maxpool(x)
        # print(pooled.shape)
        return x, pooled