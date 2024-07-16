import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from config import Config
from models.base_model import BaseEncoder, BaseDecoder, BaseAutoencoder
from models.transformer_model_v1 import PositionalEncoding, _get_activation


class TransformerEncoder(BaseEncoder):
    """A transformer encoder module with a linear layer for compression."""

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.encoder_compression = nn.Linear(config.window_dim * config.d_model, config.latent_dim)
        self.layer_norm = nn.LayerNorm(config.latent_dim)

        self.output_activation = _get_activation(config.activation)

    def forward(self, x):
        """Forward pass through the transformer encoder. Returns the latent representation."""
        x = self.input_layer(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(1, 2).reshape(x.shape[0], -1)
        x = self.encoder_compression(x)
        x = self.layer_norm(x)
        x = self.output_activation(x)
        return x


class TransformerDecoder(BaseDecoder):
    """A transformer decoder module with a linear layer for decompression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer decoder.
        :param config: The configuration object
        """
        super(TransformerDecoder, self).__init__(config)
        self.decoder_compression = nn.Linear(config.latent_dim, config.window_dim * config.d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)

        self.output_layer = nn.Linear(config.d_model, config.input_features)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.output_activation = _get_activation(config.activation)

    def forward(self, x):
        """Forward pass through the transformer decoder. Returns the reconstructed input."""
        x = self.decoder_compression(x).reshape(x.shape[0], -1, self.config.d_model)
        x = self.transformer_decoder(x)
        x = self.layer_norm(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x


class TransformerAutoencoder_4(BaseAutoencoder):
    """A transformer autoencoder module using a transformer encoder and decoder."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder.
        :param config: The configuration object
        """
        super(TransformerAutoencoder_4, self).__init__(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, x: Tensor):
        """Forward pass through the transformer autoencoder. Returns the reconstructed input."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
