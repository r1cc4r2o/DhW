import torch.nn as nn

from _Unet import Unet
from _featureExtractor import FeatureExtractor
from _featureTransformerBlock import FeatureTransformerBlock

class Architecture(nn.Module):
    """Net architecture"""
    def __init__(self, n_channels = 74, lenght_signal = 189, n_classes = 9):
        super().__init__()

        self.unet = Unet()
        self.feature_extractor = FeatureExtractor(n_channels = 74)
        self.feature_transformer = FeatureTransformerBlock()

        self.up_sampling_channels = nn.Linear(4, n_channels)
        self.mlp_classifier = nn.Sequential(
            nn.Linear(lenght_signal, lenght_signal//8),
            nn.GELU(),
            nn.LayerNorm(lenght_signal//8),
            nn.Linear(lenght_signal//8, n_classes),
        )

    def forward(self, x, t_face):
        # unet
        x = self.unet(x)
        # feature extractor
        x = self.feature_extractor(x)
        # feature transformer
        cls_token, x = self.feature_transformer(x, t_face)

        # up sampling
        x = self.up_sampling_channels(x).permute(0, 2, 1)

        # mlp classifier
        cls_classification = self.mlp_classifier(cls_token)

        return cls_classification, x