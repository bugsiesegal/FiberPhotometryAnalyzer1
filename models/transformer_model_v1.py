import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from config import Config
from models.base_model import BaseEncoder, BaseDecoder, BaseAutoencoder


def _get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f"Activation function {activation} not supported.")


class PositionalEncoding(nn.Module):
    """A positional encoding module."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the positional encoding module.
        :param d_model: The model dimensionality
        :param dropout: The dropout rate
        :param max_len: The maximum length of the input sequence
        """
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


class TransformerEncoder(BaseEncoder):
    """A transformer encoder module. Uses a linear layer for compression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer encoder.
        :param config: The configuration object
        """
        super(TransformerEncoder, self).__init__(config)
        self.input_layer = nn.Linear(config.input_features, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout, config.window_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        self.encoder_compression = nn.Linear(config.window_dim * config.d_model, config.latent_dim)

        self.output_activation = _get_activation(config.activation)

    def forward(self, x):
        """Forward pass through the transformer encoder. Returns the latent representation."""
        x = self.input_layer(x)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(1, 2).reshape(x.shape[0], -1)
        x = self.encoder_compression(x)
        x = self.output_activation(x)
        return x


class TransformerDecoder(BaseDecoder):
    """A transformer decoder module. Uses a linear layer for decompression."""

    def __init__(self, config: Config):
        """
        Initializes the transformer decoder.
        :param config: The configuration object
        """
        super(TransformerDecoder, self).__init__(config)
        self.decoder_compression = nn.Linear(config.latent_dim, config.window_dim * config.d_model)
        # I am using a encoder despite the name.
        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        self.output_layer = nn.Linear(config.d_model, config.input_features)

        self.output_activation = _get_activation(config.activation)

    def forward(self, x):
        """Forward pass through the transformer decoder. Returns the reconstructed input."""
        x = self.decoder_compression(x).reshape(x.shape[0], -1, self.config.d_model)
        x = self.transformer_decoder(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x


class TransformerAutoencoder_1(BaseAutoencoder):
    """A transformer autoencoder module. Uses a transformer encoder and decoder."""

    def __init__(self, config: Config):
        """
        Initializes the transformer autoencoder.
        :param config: The configuration object
        """
        super(TransformerAutoencoder_1, self).__init__(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, x: Tensor):
        """Forward pass through the transformer autoencoder. Returns the reconstructed input."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
