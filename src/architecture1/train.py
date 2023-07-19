from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
import time
import pytorch_lightning as pl


device = 'cuda'


def get_dataloader(train_idx, test_idx, x, y, emb_y, BATCH_SIZE):

    emb_y_train, emb_y_test = emb_y[train_idx].to(device), emb_y[test_idx].to(device)
    y_train, y_test = y[train_idx].to(device), y[test_idx].to(device)

    x_train = x[train_idx].to(device).to(torch.float32)
    train_dataset = TensorDataset(x_train, emb_y_train, y_train, x_train)
    del x_train, y_train, emb_y_train

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    del train_dataset

    x_test = x[test_idx].to(device).to(torch.float32)
    test_dataset = TensorDataset(x_test, emb_y_test, y_test, x_test)
    del x_test, y_test, emb_y_test

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    del test_dataset

    torch.cuda.empty_cache()

    return train_dataloader, test_dataloader

def get_callbacks(idx_fold):

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/content/drive/MyDrive/Neuro/word_recognition_preprocessed_data/bin/',
        filename=str(idx_fold)+'-model-word_recognition-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=True,
        mode='min'
    )

    return [checkpoint_callback, early_stopping_callback]


def main():
    
    BATCH_SIZE = 128
    LR = 5e-2

    device = 'cuda'
    x = (torch.load(PATH_1)*1e-4)
    y = torch.load(PATH_2).to(torch.float32)
    emb_y = torch.load(PATH_3).to(torch.float32)
    
    print(y.unique())
    num_classes = len(y.unique())
    # one hot encoding
    y = nn.functional.one_hot(y.to(torch.int64), num_classes=num_classes).to(torch.float32)

    srf = KFold(n_splits=2, shuffle=True, random_state=42)

    logged_metrics = []

    for idx_fold, (train_idx, test_idx) in enumerate(srf.split(x)):

        train_dataloader, test_dataloader = get_dataloader(train_idx, test_idx, x, y, emb_y, BATCH_SIZE)

        model = Net(193, 901, num_classes, num_layers_t=1).to(device)

        trainer = pl.Trainer(
            max_epochs=5,
            accelerator='auto',
            callbacks=get_callbacks(idx_fold),
        )

        trainer.fit(model, train_dataloader, test_dataloader)

        logged_metrics.append(trainer.logged_metrics)


        del model, train_dataloader, test_dataloader, trainer
        torch.cuda.empty_cache()
        time.sleep(4)

        def compute_avg_std_metrics_cross_fold(logged_metrics):
            metrics = {}
            for metric in logged_metrics[0]:
                metrics[metric] = []
                for fold in logged_metrics:
                    metrics[metric].append(float(fold[metric].numpy()))


            for metric in metrics:
                metrics[metric] = np.array(metrics[metric])
                print(metric, metrics[metric].mean(), metrics[metric].std())

        compute_avg_std_metrics_cross_fold(logged_metrics)