import torch
import torch.nn as nn
import torch.nn.functional as F

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
                batch_first=True
            ),
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        self.encoder_fc = nn.Linear(self.window_dim, self.latent_dim)
        # decoder
        self.decoder_fc = nn.Linear(self.latent_dim, self.window_dim)
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

    def forward(self, x):
        x = self.encoder_transformer(x).swapaxes(1, 2)
        x = self.encoder_fc(x)
        x = self.decoder_fc(x).swapaxes(1, 2)
        x = self.decoder_transformer(x)
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
        self.logger.experiment.log({'input': batch[0]})
        self.logger.experiment.log({'output': y_hat[0]})

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        self._log_prediction(batch, self.model(batch))
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
