import torch.nn as nn

class ConvBlock(nn.Module):
    """Apply a two steps convolution with normalization and GELU activation"""
    def __init__(self, channel_in, channel_out, shape_norm, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.ln = nn.LayerNorm(shape_norm)
        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x