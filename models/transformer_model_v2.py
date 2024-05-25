from torch import nn, Tensor
import torch.nn.functional as F

from config import Config
from models.base_model import BaseEncoder, BaseAutoencoder, BaseDecoder
from models.transformer_model_v1 import PositionalEncoding, _get_activation


class TransformerEncoder(BaseEncoder):
    """A transformer encoder module. Uses cut off for compression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer encoder.
        :param config: The configuration object
        """
        super(TransformerEncoder, self).__init__(config)
        self.input_layer = nn.Linear(config.input_features, config.d_model)
        if self.config.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(config.d_model, config.dropout, config.window_dim)
        else:
            self.positional_encoding = nn.Identity()
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

        self.output_activation = _get_activation(config.activation)

    def forward(self, x):
        """Forward pass through the transformer encoder. Returns the latent representation."""
        x = self.input_layer(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        # Set all values to zero after the latent dimension
        x[:, self.config.latent_dim:] = 0
        x = self.output_activation(x)
        return x


class TransformerDecoder(BaseDecoder):
    """A transformer decoder module. Uses a cut off for compression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer decoder.
        :param config: The configuration object
        """
        super(TransformerDecoder, self).__init__(config)
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

        self.output_activation = _get_activation(config.activation)

    def forward(self, x):
        """Forward pass through the transformer decoder. Returns the reconstructed input."""
        x = self.transformer_decoder(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x


class TransformerAutoencoder_2(BaseAutoencoder):
    """A transformer autoencoder module. Uses a transformer encoder and decoder."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder.
        :param config: The configuration object
        """
        super(TransformerAutoencoder_2, self).__init__(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, x: Tensor):
        """Forward pass through the transformer autoencoder. Returns the reconstructed input."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
