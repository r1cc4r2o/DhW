import torch

def custom_mse(y_pred, y_true):
    return (torch.sum(y_pred - y_true, dim=-1)**2).mean()