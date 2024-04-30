import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from lightning.pytorch import LightningModule
from torch import Tensor

from config import Config


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super(TransformerEncoder, self).__init__()
        self.config = config

        self.input_layer = nn.Linear(config.input_features, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout, config.window_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        self.encoder_compression = nn.Linear(config.window_dim * config.d_model, config.latent_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(1, 2)
        x = self.encoder_compression(x)
        x = F.sigmoid(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: Config):
        super(TransformerDecoder, self).__init__()
        self.config = config

        self.decoder_compression = nn.Linear(config.latent_dim, config.window_dim * config.d_model)
        # I am using a encoder despite the name.
        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        self.output_layer = nn.Linear(config.d_model, config.input_features)

    def forward(self, x):
        x = self.decoder_compression(x).swapaxes(1, 2)
        x = self.transformer_decoder(x)
        x = self.output_layer(x)
        return x


class TransformerAutoencoder(nn.Module):
    def __init__(self, config: Config):
        super(TransformerAutoencoder, self).__init__()
        self.config = config

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FFTTransformerEncoder(TransformerAutoencoder):
    def forward(self, x):
        x = torch.fft.fft(x, dim=1)
        x = torch.concat((x.real, x.imag), dim=2)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.fft.ifft(torch.complex(x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]), dim=1).real
        x = F.sigmoid(x)
        return x


class AutoencoderModule(LightningModule):
    def __init__(self, config: Config):
        super(AutoencoderModule, self).__init__()
        self.config = config
        if config.use_fft:
            self.model = FFTTransformerEncoder(config)
        else:
            self.model = TransformerAutoencoder(config)
        self.loss = nn.MSELoss()
        self.learning_rate = config.learning_rate

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)

        if batch_idx == 0:
            x = batch
            x_hat = self(x)
            fig, axs = plt.subplots(x.shape[2], 1, figsize=(10, 10))
            for i in range(x.shape[2]):
                axs[i].plot(x[0, :, i].cpu().detach().numpy(), label='Original')
                axs[i].plot(x_hat[0, :, i].cpu().detach().numpy(), label='Reconstructed')

            self.logger.experiment.log({'Plot': fig})
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.config.lr_factor,
                                                               patience=self.config.lr_patience, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'frequency': 100,
                'interval': 'step',
            }
        }
