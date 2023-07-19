import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics


from _architecture import Architecture
from utils import custom_regression_loss




class Net(pl.LightningModule):
    def __init__(self, n_channels = 74, lenght_signal = 189, n_classes = 9):
        super().__init__()

        self.model = Architecture(n_channels = n_channels, lenght_signal = lenght_signal, n_classes = n_classes)

        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_regression = custom_regression_loss
        # self.huber_loss = nn.SmoothL1Loss()
        self.classification_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)

    def forward(self, x, t_face):
        return self.model(x, t_face)

    def training_step(self, x, batch_idx):
        # get the data
        x, x_face, y_label_signal, y_label_class = x

        cls_classification, x = self.forward(x, x_face)

        # classification loss
        loss_classification = self.criterion_classification(cls_classification, y_label_class)

        # regression loss
        loss_regression = self.criterion_regression(x, y_label_signal)

        # total loss
        loss = loss_classification + loss_regression

        y_label_class = torch.argmax(y_label_class, dim = -1)

        # classification accuracy
        acc = self.classification_accuracy(cls_classification, y_label_class)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_classification', loss_classification, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_regression', loss_regression, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_huber_loss', huber_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_classification', acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, x, batch_idx):
        # get the data
        x, x_face, y_label_signal, y_label_class = x


        cls_classification, x = self.forward(x, x_face)

        # classification loss
        loss_classification = self.criterion_classification(cls_classification, y_label_class)

        y_label_class = torch.argmax(y_label_class, dim = -1)

        # regression loss
        loss_regression = self.criterion_regression(x, y_label_signal)

        if not loss_regression.isnan():
          # total loss
          loss = loss_classification + loss_regression
          self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
          self.log('val_loss_regression', loss_regression, on_epoch=True, prog_bar=True, logger=True)
        else:
          loss = loss_classification + 4e5


        # classification accuracy
        acc = self.classification_accuracy(cls_classification, y_label_class)

        self.log('val_loss_classification', loss_classification, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_classification', acc, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]

##########################################################################################################
##########################################################################################################
##########################################################################################################

# example

# x = torch.randn(32, 74, 26, 189)
# target_face = torch.randn(32)

# model = Net()

# print(model)
# # number of parameters
# print('Number of parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 'M')
# del model