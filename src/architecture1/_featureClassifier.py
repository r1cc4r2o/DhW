import torch.nn as nn

from _weightedConvFeatureExtractor import WeightedConvFeatureExtractor

class FeatureClassifier(nn.Module):
    def __init__(self, n_channels, lenght_sequence, n_classes, num_layers_t=1, n_head=2, stride=2, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.weightedConvFeatureExtractor = WeightedConvFeatureExtractor(
            n_channels, lenght_sequence, num_layers_t, n_head, stride, kernel_size, dilation, bias
        )

        self.dropout = nn.Dropout(0.05, inplace=True)
        self.linear = nn.Linear(512, 64)
        self.bilinear = nn.Bilinear(64, 64, n_classes, bias=bias)

    def forward(self, x, x_emb):
        x = self.weightedConvFeatureExtractor(x)
        x = self.dropout(x)
        x_emb = self.linear(x_emb)
        logits_c = self.bilinear(x, x_emb)
        return logits_c, x