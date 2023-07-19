import torch
import torch.nn as nn

from _featureClassifier import FeatureClassifier

class FeatureRegressor(nn.Module):
    def __init__(self, n_channels, lenght_sequence, n_classes, num_layers_t=1, n_head=2, stride=2, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.featureClassifier = FeatureClassifier(
            n_channels, lenght_sequence, n_classes, num_layers_t, n_head, stride, kernel_size, dilation, bias
        )

        self.channel_enhancer = nn.Conv1d(
                                        1,
                                        n_channels,
                                        1,
                                        stride=1,
                                        dilation=1,
                                        bias=True
                                    )

        self.mlp_regression = nn.Sequential(
            nn.Dropout(0.01, inplace=True),
            nn.Linear(64, 256, bias=bias),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.01, inplace=True),
            nn.Linear(256, lenght_sequence, bias=bias)
        )

    def forward(self, x, x_emb):
        logits_c, x = self.featureClassifier(x, x_emb)
        x = self.channel_enhancer(x.unsqueeze(1))
        x = self.mlp_regression(x)
        return logits_c, x