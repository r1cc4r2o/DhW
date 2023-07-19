import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, n_channels = 74, kernel_size=3, stride=2, padding=0, dilation=1, dropout=0.1, stride_factor_reduction = 2):
        super().__init__()

        self.n_channels = n_channels

        self.channel_reduction = 16
        self.stride_factor_reduction = stride_factor_reduction

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels//self.channel_reduction, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(in_channels=n_channels//self.channel_reduction, out_channels=n_channels//self.channel_reduction, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        self.activation = nn.GELU()

        self.layer_norm_1 = nn.LayerNorm([n_channels//self.channel_reduction, 26//self.stride_factor_reduction-1, 189//self.stride_factor_reduction])
        self.layer_norm_2 = nn.LayerNorm([n_channels//self.channel_reduction, (26//self.stride_factor_reduction-1)//self.stride_factor_reduction-1, (189//self.stride_factor_reduction -1)//self.stride_factor_reduction])

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):

        # feature extraction block 1
        x = self.conv1(x)
        # print(x.shape)
        x = self.activation(x)
        x = self.layer_norm_1(x)
        x = self.dropout(x)

        # feature extraction block 2
        x = self.conv2(x)
        # print(x.shape)
        x = self.activation(x)
        x = self.layer_norm_2(x)
        x = self.dropout(x)

        x = x.flatten(start_dim=2, end_dim=-1)

        return x