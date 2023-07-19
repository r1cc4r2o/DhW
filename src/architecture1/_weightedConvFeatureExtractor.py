import torch
import torch.nn as nn

from _convFeatureExtractor import ConvFeatureExtractor


class WeightedConvFeatureExtractor(nn.Module):
    def __init__(self, n_channels, lenght_sequence, num_layers_t=1, n_head=2, stride=2, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.convFeatureExtractor = ConvFeatureExtractor(n_channels, lenght_sequence, stride, kernel_size, dilation, bias)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=n_head, batch_first=True, norm_first=True), num_layers=num_layers_t
        )

        self.dropout = nn.Dropout(0.1, inplace=True)
        self.linear_1 = nn.Linear(32*2, 16, bias=bias)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.layer_norm_1 = nn.LayerNorm(16*n_channels)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(16*n_channels, 64, bias=bias)
        self.layer_norm_2 = nn.LayerNorm(64)

    def forward(self, x):
        x_0 = self.convFeatureExtractor(x)
        x = self.transformer_encoder(x_0)

        # dropout
        x = self.dropout(x)
        # concatenate
        x = torch.cat((x_0, x), dim=-1)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.flatten(x)
        x = self.layer_norm_1(x)
        x = self.linear_2(x)
        x = self.layer_norm_2(x)
        return x