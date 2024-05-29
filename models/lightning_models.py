import math
from abc import ABC, abstractmethod
from typing import Type, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from lightning.pytorch import LightningModule
from torch import Tensor

from config import Config
from models.base_model import BaseAutoencoder
from models.fft_transformer_model import FFTTransformerAutoencoder_1
from models.transformer_model_v1 import TransformerAutoencoder_1
from models.transformer_model_v2 import TransformerAutoencoder_2


class BaseAutoencoderModule(LightningModule, ABC):
    model: BaseAutoencoder

    def __init__(self, config: Config):
        super(BaseAutoencoderModule, self).__init__()
        self.config = config
        self.model = BaseAutoencoder
        self.loss = nn.MSELoss()
        self.learning_rate = config.learning_rate

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def _common_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        # Convert loss to float32 to avoid overflow
        self.log('train_loss', loss.to(torch.float32).cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss.to(torch.float32).cpu())

        if batch_idx == 0 and self.logger is not None:
            x = batch
            x_hat = self(x)
            for i in range(x.shape[2]):
                plt.plot(x[0, :, i].cpu().detach().numpy(), label='Original')
                plt.plot(x_hat[0, :, i].cpu().detach().numpy(), label='Reconstructed')
                plt.legend()
                loss = nn.functional.l1_loss(target=x, input=x_hat, reduction='none')
                self.logger.experiment.log({f"Reconstruction Error {i}": wandb.Histogram(loss.cpu(), num_bins=100), f"Plot {i}": plt})
                plt.clf()
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss.to(torch.float32).cpu())
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.config.lr_factor,
                                                               patience=self.config.lr_patience, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.config.monitor,
                'frequency': self.config.scheduler_frequency,
                'interval': self.config.scheduler_interval,
            }
        }

    def on_after_backward(self):
        if self.trainer.global_step % 25 == 0:  # log every 25 steps
            self.logger.experiment.log({"Learning Rate": self.trainer.optimizers[0].param_groups[0]['lr']})


class TransformerAutoencoderModule_1(BaseAutoencoderModule):
    """A module for a transformer autoencoder with linear layer for compression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder module.
        :param config: The configuration object
        """
        super(TransformerAutoencoderModule_1, self).__init__(config)
        self.model = TransformerAutoencoder_1(config)
        self.loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.
        :param batch: Batch of data
        :param batch_idx: Batch index
        :return: Loss
        """
        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        return loss


class TransformerAutoencoderModule_2(BaseAutoencoderModule):
    """A module for a transformer autoencoder with cut off for compression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder module.
        :param config: The configuration object
        """
        super(TransformerAutoencoderModule_2, self).__init__(config)
        self.model = TransformerAutoencoder_2(config)
        self.loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.
        :param batch: Batch of data
        :param batch_idx: Batch index
        :return: Loss
        """
        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        return loss


class FFTAutoencoderModule_V1(BaseAutoencoderModule):
    """A module for a transformer autoencoder using FFT and a linear layer for compression."""

    def __init__(self, config: Config):
        """
        Initializes the FFT autoencoder module.
        :param config: The configuration object
        """
        super(FFTAutoencoderModule_V1, self).__init__(config)
        self.model = FFTTransformerAutoencoder_1(config)
        self.loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.
        :param batch: Batch of data
        :param batch_idx: Batch index
        :return: Loss
        """
        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        return loss
