import torch

def custom_regression_loss(x, y_label_signal):
  y_label_signal = y_label_signal[~torch.any(x.isnan(),dim=-1)]
  x = x[~torch.any(x.isnan(),dim=-1)]
  loss = abs((x*1e2-y_label_signal*1e2))
  loss = loss.sum(dim=-1).mean()
  return loss