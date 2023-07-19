import torch
import torch.nn as nn

class FeatureTransformerBlock(nn.Module):
    """Transformer block for the feature extraction"""
    def __init__(self, lenght_signal = 189, num_t_layer = 1, n_head = 3, stride_factor_reduction = 2):
        super().__init__()

        self.num_t_layer = num_t_layer
        self.n_head = n_head

        self.stride_factor_reduction = stride_factor_reduction

        freq = (26//self.stride_factor_reduction-1)//self.stride_factor_reduction-1
        time_dim = (189//self.stride_factor_reduction -1)//self.stride_factor_reduction

        self.linear_transformation = nn.Linear(freq*time_dim, lenght_signal, bias=True)

        # 768 is the dimension of the embedding of BERT
        self.linear_transformation_tface = nn.Linear(768, lenght_signal, bias=True)

        self.te_layer = nn.TransformerEncoderLayer(d_model=lenght_signal, nhead=n_head)
        self.te = nn.TransformerEncoder(self.te_layer, num_layers=num_t_layer)

    def forward(self, x, t_face):
        """Return the cls token and the embedding"""

        # linear transformation
        x = self.linear_transformation(x)

        # add cls token
        # cls_token = t_face.unsqueeze(-1).to(x.device)
        cls_token = t_face.to(x.device)
        cls_token = self.linear_transformation_tface(cls_token).unsqueeze(-2)
        # print(cls_token.shape, x.shape)
        # cls_token = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)

        # concat cls token with the embedding
        x = torch.cat([cls_token, x], dim=1)

        # transpose to match transformer input
        x = x.transpose(0, 1)

        # transformer
        x = self.te(x)

        # transpose back
        x = x.transpose(0, 1)

        # get the cls token
        cls_token = x[:, 0, :]

        # get the embedding
        x = x[:, 1:, :].permute(0, 2, 1)

        return cls_token, x