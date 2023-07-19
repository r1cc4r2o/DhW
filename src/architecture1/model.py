import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from _featureRegressor import FeatureRegressor
from utils import custom_mse


class Net(pl.LightningModule):
    def __init__(self, n_channels, lenght_sequence, n_classes, num_layers_t=1, n_head=2, stride=2, kernel_size=3, dilation=1, bias=True):
        super().__init__()

        self.model = FeatureRegressor(n_channels, lenght_sequence, n_classes, num_layers_t, n_head, stride, kernel_size, dilation, bias)

        self.loss_class = nn.CrossEntropyLoss()
        self.loss_reg = custom_mse

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass",num_classes=n_classes)
        self.precision = torchmetrics.Precision(task="multiclass",num_classes=n_classes)
        self.recall = torchmetrics.Recall(task="multiclass",num_classes=n_classes)


    def forward(self, x, x_emb):
        return self.model(x, x_emb)

    def training_step(self, batch, batch_idx):
        x, x_emb, y_class, y_reg = batch

        y_c, y_r = self(x, x_emb)

        loss_class = self.loss_class(y_c, y_class)
        loss_reg = self.loss_reg(y_r, y_reg)

        loss = loss_class + loss_reg

        y = torch.argmax(y_class, dim=-1)

        a = self.accuracy(y_c, y)
        f = self.f1(y_c, y)
        p = self.precision(y_c, y)
        r = self.recall(y_c, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_class", loss_class, prog_bar=True)
        self.log("train_loss_reg", loss_reg, prog_bar=True)
        self.log("train_acc", a, prog_bar=True)
        self.log("train_f1", f, prog_bar=True)
        self.log("train_precision", p, prog_bar=True)
        self.log("train_recall", r, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_emb, y_class, y_reg = batch

        y_c, y_r = self(x, x_emb)

        loss_class = self.loss_class(y_c, y_class)
        loss_reg = self.loss_reg(y_r, y_reg)

        loss = loss_class + loss_reg

        y = torch.argmax(y_class, dim=-1)

        a = self.accuracy(y_c, y)
        f = self.f1(y_c, y)
        p = self.precision(y_c, y)
        r = self.recall(y_c, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_class", loss_class, prog_bar=True)
        self.log("val_loss_reg", loss_reg, prog_bar=True)
        self.log("val_acc", a, prog_bar=True)
        self.log("val_f1", f, prog_bar=True)
        self.log("val_precision", p, prog_bar=True)
        self.log("val_recall", r, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]