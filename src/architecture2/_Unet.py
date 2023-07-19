import torch.nn as nn

from _UnetEncoderBlock import UnetEncoderBlock
from _UnetDecoderBlock import UnetDecoderBlock

class Unet(nn.Module):
    """Unet architecture"""
    def __init__(self, encoder_steps = 2, decoder_steps = 2, n_channels = 74):
        super().__init__()

        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps


        # encoder
        self.encoder_1 = UnetEncoderBlock(n_channels, n_channels*2)
        self.encoder_2 = UnetEncoderBlock(n_channels*2, n_channels*4)
        # decoder
        self.decoder_1 = UnetDecoderBlock(n_channels*4, n_channels*2)
        self.decoder_2 = UnetDecoderBlock(n_channels*2, n_channels)


    def forward(self, x):

        # encoder
        x, pooled_1 = self.encoder_1(x)
        x, pooled_2 = self.encoder_2(x) # bottleneck

        # decoder
        x = self.decoder_1(x, pooled_2)
        x = self.decoder_2(x, pooled_1)

        return x