import torch
import torch.nn as nn

from _convBlock import ConvBlock

class UnetDecoderBlock(nn.Module):
    """Upsample the input by a factor 2 lowering the number of channels by a factor 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,  dilation=1, kernel_size=3, padding=0, stride=2)#dilation=1,
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  dilation=1, kernel_size=3, padding=1)#dilation=1,
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,  kernel_size=2, stride=2, padding=0, dilation=1)
        shape_norm = [out_channels,  26, 189]
        self.convblock = ConvBlock(in_channels+out_channels, out_channels, shape_norm)

    def forward(self, x, skip):
        skip = self.conv1(skip)
        # print(x.shape, skip.shape)
        # pad x to match skip
        # torch.Size([32, 74, 26, 188]) torch.Size([32, 148, 26, 189])
        skip = nn.functional.pad(skip, (0, 1, 0, 0))
        x = torch.cat([x, skip], dim=1)
        x = self.convblock(x)
        return x