import torch
import torch.nn.functional as F

from models.transformer_model_v1 import TransformerAutoencoder_1, TransformerEncoder, TransformerDecoder


class FFTTransformerAutoencoder_1(TransformerAutoencoder_1):
    """The FFT Transformer Autoencoder model. Uses FFT to transform the input data. Compresses the data using a
    linear layer."""

    def __init__(self, config):
        """
        Initializes the FFT Transformer Autoencoder model.
        :param config: The configuration object
        """
        config.input_features = config.input_features * 2
        super(FFTTransformerAutoencoder_1, self).__init__(config)

    def forward(self, x):
        """The forward pass"""
        x = torch.fft.fft(x, dim=1)
        x = torch.concat((x.real, x.imag), dim=2)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.fft.ifft(torch.complex(x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]), dim=1).real
        x = F.sigmoid(x)
        return x
