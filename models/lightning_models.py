import math
from abc import ABC, abstractmethod
from typing import Type, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from hilbertcurve.hilbertcurve import HilbertCurve

from lightning.pytorch import LightningModule
from matplotlib.colors import ListedColormap
from torch import Tensor

from config import Config
from models.base_model import BaseAutoencoder
from models.fft_transformer_model import FFTTransformerAutoencoder_1
from models.sparse_model import SparseTransformerAutoencoder
from models.transformer_model_v1 import TransformerAutoencoder_1
from models.transformer_model_v2 import TransformerAutoencoder_2
from models.transformer_model_v3 import TransformerAutoencoder_3
from models.transformer_model_v4 import TransformerAutoencoder_4
from typing import Union, Dict


# import pdb; pdb.set_trace()

class BaseAutoencoderModule(LightningModule, ABC):
    model: BaseAutoencoder

    def __init__(self, config: Union[Config, Dict]):
        super(BaseAutoencoderModule, self).__init__()
        if isinstance(config, Dict):
            config = Config.from_dict(config)
        self.config = config
        self.model = BaseAutoencoder
        self.loss = nn.MSELoss()
        self.learning_rate = config.learning_rate
        self.save_hyperparameters(config.to_dict())

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
                self.logger.experiment.log(
                    {f"Reconstruction Error {i}": wandb.Histogram(loss.cpu(), num_bins=100), f"Plot {i}": plt})
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
        if self.trainer.global_step % 25 == 0 and self.logger is not None:  # log every 25 steps
            self.logger.experiment.log({"Learning Rate": self.trainer.optimizers[0].param_groups[0]['lr']})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_hat = self.model.encoder(batch)
        return x_hat


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


class TransformerAutoencoderModule_3(BaseAutoencoderModule):
    """A module for a transformer autoencoder with cut off for compression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder module.
        :param config: The configuration object
        """
        super(TransformerAutoencoderModule_3, self).__init__(config)
        self.model = TransformerAutoencoder_3(config)
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


class TransformerAutoencoderModule_4(BaseAutoencoderModule):
    """A module for a transformer autoencoder with linear compression and normalization."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder module.
        :param config: The configuration object
        """
        super(TransformerAutoencoderModule_4, self).__init__(config)
        self.model = TransformerAutoencoder_4(config)
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


class SparseAutoencoderModule(BaseAutoencoderModule):
    """A module for a sparse transformer autoencoder."""

    def __init__(self, config: Config):
        """
        Initializes the sparse autoencoder module.
        :param config: The configuration object
        """
        super(SparseAutoencoderModule, self).__init__(config)
        self.model = SparseTransformerAutoencoder(config)
        self.loss = nn.MSELoss()

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your device configuration.")

        self.sparsity_weight = torch.tensor(config.sparsity_weight, device="cuda")
        self.current_sparsity_weight = nn.Parameter(self.sparsity_weight.clone(), requires_grad=True)
        self.register_parameter('current_sparsity_weight', self.current_sparsity_weight)

        # Initialize the optimizer for sparsity weight
        self.sparsity_optimizer = torch.optim.Adam([self.current_sparsity_weight], lr=config.sparsity_lr)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def calculate_loss(self, x, hidden_activation, x_hat):
        reconstruction_loss = self.loss(x, x_hat)
        sparsity_loss = torch.mean(torch.abs(hidden_activation))
        total_loss = reconstruction_loss + F.relu(self.current_sparsity_weight) * sparsity_loss
        return total_loss, reconstruction_loss, sparsity_loss

    def _common_step(self, batch, batch_idx):
        x = batch
        hidden_activation = self.model.encoder(x)
        x_hat = self.model.decoder(hidden_activation)
        total_loss, reconstruction_loss, sparsity_loss = self.calculate_loss(x, hidden_activation, x_hat)

        self.log('sparsity_weight', self.current_sparsity_weight)

        # Log the average activation percentage
        avg_activation_percentage = torch.mean((hidden_activation > 0).float()).item()
        self.log('avg_activation_percentage', avg_activation_percentage)
        return total_loss, reconstruction_loss, sparsity_loss

    def plot_hidden_activations(self, hidden_activation):
        """
        Plots the hidden activations as squares with values either 1 or 0.
        :param hidden_activation: The hidden activations (expected to be a 2D tensor [batch_size, units])
        """
        hidden_activation = hidden_activation.cpu().detach().numpy()

        # Ensure hidden_activation is a 2D array [batch_size, units]
        if len(hidden_activation.shape) != 2:
            raise ValueError("Expected hidden_activation to be a 2D tensor of shape [batch_size, units]")

        # Compute the mean activation across the batch
        mean_hidden_activation = np.mean(hidden_activation, axis=0)

        # Binarize activations
        binarized_hidden_activation = (mean_hidden_activation > 0).astype(int)

        # Plot using Hilbert curve
        n = len(binarized_hidden_activation)
        dim = int(np.ceil(np.log2(n) / 2))
        side_length = 2 ** dim

        # Pad the binarized_hidden_activation array to fit in a square
        padded_activation = np.zeros(side_length ** 2)
        padded_activation[:n] = binarized_hidden_activation

        hilbert_curve = HilbertCurve(dim, 2)
        points = hilbert_curve.points_from_distances(range(side_length ** 2))

        # Create an empty grid
        grid = np.zeros((side_length, side_length))

        for i, point in enumerate(points):
            x, y = point
            grid[x, y] = padded_activation[i]

        # Plot the grid
        cmap = ListedColormap(['white', 'black'])
        plt.imshow(grid, cmap=cmap, origin='lower')
        plt.title('Hidden Activations')
        return plt

    def adjust_sparsity_weight(self, sparsity_loss):
        """
        Adjusts the sparsity weight based on the current sparsity loss.
        This function can be refined to use more sophisticated adaptive techniques.
        """
        self.sparsity_optimizer.zero_grad()
        sparsity_loss.backward(retain_graph=True)
        self.sparsity_optimizer.step()

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        loss, reconstruction_loss, sparsity_loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss.to(torch.float32).cpu())

        if batch_idx == 0 and self.logger is not None:
            x = batch
            hidden_activation = self.model.encoder(x)
            x_hat = self.model.decoder(hidden_activation)
            self.logger.experiment.log({"Hidden Layer": self.plot_hidden_activations(hidden_activation)})
            plt.clf()
            for i in range(x.shape[2]):
                plt.plot(x[0, :, i].cpu().detach().numpy(), label='Original')
                plt.plot(x_hat[0, :, i].cpu().detach().numpy(), label='Reconstructed')
                plt.legend()
                loss = nn.functional.l1_loss(target=x, input=x_hat, reduction='none')
                self.logger.experiment.log(
                    {f"Reconstruction Error {i}": wandb.Histogram(loss.cpu(), num_bins=100), f"Plot {i}": plt})
                plt.clf()
        return loss

    def training_step(self, batch, batch_idx):
        total_loss, reconstruction_loss, sparsity_loss = self._common_step(batch, batch_idx)
        self.log('reconstruction_loss', reconstruction_loss.to(torch.float32).cpu())
        self.log('sparsity_loss', sparsity_loss.to(torch.float32).cpu())
        # Adjust sparsity weight should use the sparsity loss, not total_loss
        self.adjust_sparsity_weight(sparsity_loss)
        self.log('train_loss', total_loss.to(torch.float32).cpu())
        return total_loss
