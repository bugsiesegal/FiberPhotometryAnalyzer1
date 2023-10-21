import math
import os

import data
import torch
import wandb
from JAAEC import AmazingAutoEncoder
import lightning as pl
from pytorch_lightning.loggers import WandbLogger


def train():
    wandb.init(project="JAAEC_Fiberphotometry")

    # Hyperparameters
    learning_rate = wandb.config.learning_rate
    window_size = wandb.config.window_size
    embedding_size = wandb.config.embedding_size
    batch_size = wandb.config.batch_size
    num_workers = wandb.config.num_workers

    datamodule = data.TDTFiberPhotometryDataModule("data", window_size, batch_size=batch_size, num_workers=num_workers)

    autoencoder = AmazingAutoEncoder((1, window_size, 1), (1, embedding_size), learning_rate=learning_rate, num_layers=3)

    torch.set_float32_matmul_precision('medium')

    wandb_logger = WandbLogger(project="JAAEC_Fiberphotometry", log_model=True,
                               save_dir=os.path.join(os.getcwd(), "wandb_logs"))

    trainer = pl.Trainer(logger=wandb_logger, precision='16-mixed', max_time="00:08:00:00",)

    trainer.fit(autoencoder, datamodule=datamodule)


if __name__ == "__main__":
    wandb.init(project="JAAEC_Fiberphotometry")
    wandb.config.learning_rate = 1e-6
    wandb.config.window_size = 1000
    wandb.config.embedding_size = 32
    wandb.config.batch_size = 4
    wandb.config.num_workers = 16

    train()
