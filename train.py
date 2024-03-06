import math
import os

from lightning.pytorch.callbacks import ModelPruning, LearningRateFinder

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
    embedding_size = 2**wandb.config.embedding_size
    batch_size = wandb.config.batch_size
    num_workers = wandb.config.num_workers
    num_heads = wandb.config.num_heads
    num_layers = wandb.config.num_layers
    dropout = wandb.config.dropout

    datamodule = data.TDTFiberPhotometryDataModule("data",
                                                   window_size, batch_size=batch_size, num_workers=num_workers)

    autoencoder = AmazingAutoEncoder((1, window_size, 1), (1, embedding_size),
                                     learning_rate=learning_rate, num_layers=num_layers,
                                     num_heads=num_heads, dropout=dropout)

    torch.set_float32_matmul_precision('medium')

    wandb_logger = WandbLogger(project="JAAEC_Fiberphotometry", log_model=True,
                               save_dir=os.path.join(os.getcwd(), "wandb_logs"))

    trainer = pl.Trainer(logger=wandb_logger,
                         precision='16-mixed',
                         max_time="00:20:00:00",
                         benchmark=True,
                         callbacks=[ModelPruning(pruning_fn="l1_unstructured"), LearningRateFinder(),],
                         gradient_clip_val=0.5,
                         )

    trainer.fit(autoencoder, datamodule=datamodule)


if __name__ == "__main__":
    wandb.init(project="JAAEC_Fiberphotometry")
    wandb.config.learning_rate = 1e-6
    wandb.config.window_size = 1000
    wandb.config.embedding_size = 5
    wandb.config.batch_size = 64
    wandb.config.num_workers = 16
    wandb.config.num_heads = 8
    wandb.config.num_layers = 3
    wandb.config.dropout = 0.1

    train()
