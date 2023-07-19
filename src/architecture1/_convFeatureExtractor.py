import torch.nn as nn

class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_channels, lenght_sequence, stride, kernel_size, dilation, bias=True):
        super().__init__()

        self.conv = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias
        )
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(lenght_sequence//2)
        self.dropout = nn.Dropout(0.1, inplace=True)
        self.linear = nn.Linear(lenght_sequence//2, 32, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x