from torch import nn, Tensor
from abc import ABC, abstractmethod

from config import Config


class BaseEncoder(nn.Module, ABC):
    def __init__(self, config: Config):
        super(BaseEncoder, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: Tensor):
        pass


class BaseDecoder(nn.Module, ABC):
    def __init__(self, config: Config):
        super(BaseDecoder, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: Tensor):
        pass


class BaseAutoencoder(nn.Module, ABC):
    encoder: BaseEncoder
    decoder: BaseDecoder

    def __init__(self, config: Config):
        super(BaseAutoencoder, self).__init__()
        self.config = config
        self.encoder = BaseEncoder(config)
        self.decoder = BaseDecoder(config)

    @abstractmethod
    def forward(self, x: Tensor):
        self.encoder(x)
        self.decoder(x)
        return x
