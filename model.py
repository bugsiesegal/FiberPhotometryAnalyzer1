import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from lightning.pytorch import LightningModule

from config import Config


class Autoencoder(nn.Module):
    def __init__(self, config: Config):
        super(Autoencoder, self).__init__()
        self.config = config
        self.window_dim = config.window_dim
        self.nhead = config.nhead
        self.dim_feedforward = config.dim_feedforward
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.activation = config.activation
        self.latent_dim = config.latent_dim
        self.d_model = config.d_model

        # encoder
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
            ),
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        self.encoder_fc = nn.Linear(self.window_dim * self.d_model, self.latent_dim)
        # decoder
        self.decoder_fc = nn.Linear(self.latent_dim, self.window_dim * self.d_model)
        self.decoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True
            ),
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        self.decoder_final = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        x = self.encoder_transformer(x).swapaxes(1, 2)
        x = self.encoder_fc(x.reshape(-1, self.window_dim * self.d_model))
        x = self.decoder_fc(x).reshape(-1, self.window_dim, self.d_model)
        x = self.decoder_transformer(x)
        x = self.decoder_final(x)
        x = F.tanh(x)
        return x


class AutoencoderModule(LightningModule):
    def __init__(self, config: Config):
        super(AutoencoderModule, self).__init__()
        self.config = config
        self.model = Autoencoder(config)
        self.lr = config.learning_rate

        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, config.window_dim, config.d_model)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        y_hat = self.model(batch)
        loss = F.mse_loss(y_hat, batch)
        return loss

    def _log_prediction(self, batch, y_hat):
        y_hat = torch.nan_to_num(y_hat, nan=0.0)
        # Plot input and output using matplotlib
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(batch[0, :, 0].detach().cpu().numpy(), label='Input')
        ax[1].plot(y_hat[0, :, 0].detach().cpu().numpy(), label='Output')
        # Log using wandb
        self.logger.experiment.log({'prediction': wandb.Plotly(plt)})

    def training_step(self, batch, batch_idx):
        if self.config.normalize:
            batch = (batch - batch.mean(dim=1)) / batch.std(dim=1)

        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        if batch_idx == 0:
            self._log_prediction(batch, self.model(batch))
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.config.lr_factor, patience=self.config.lr_patience, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'train_loss',
                'frequency': 1000,
                'interval': 'step',
            }
        }
