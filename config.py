from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    # Basic model configuration
    d_model: int = 64
    window_dim: int = 1000
    nhead: int = 8
    dim_feedforward: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = 'relu'
    latent_dim: int = 128

    # Model selection
    model: str = 'transformer_v1'

    # Learning parameters
    learning_rate: float = 1e-3
    lr_factor: float = 0.1
    lr_patience: int = 5

    # Scheduling parameters
    monitor: str = 'val_loss'
    scheduler_frequency: int = 1
    scheduler_interval: str = 'epoch'

    # Input configuration
    input_features: int = 52
    use_fft: bool = False

    # Data handling
    data_dir: str = field(default_factory=str)
    batch_size: int = 32
    num_workers: int = 4

    # Feature toggles
    use_fiber: bool = True
    use_tracking: bool = True

    # Fiber specific configuration
    fiber_channel_name: str = 'Analog In. | Ch.1 AIn-1 - Dem (AOut-2)'
    control_channel_name: str = 'Analog In. | Ch.1 AIn-1 - Dem (AOut-3)'
    normalize: bool = True

    # Training configuration
    max_epochs: int = 100
    precision: str = '32'
    max_time: str = '1h'

    def to_yaml(self, path: str):
        with open(path, 'w') as file:
            yaml.dump(self.__dict__, file)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)
            return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def to_json(self, path: str):
        import json
        with open(path, 'w') as file:
            json.dump(self.__dict__, file)

    @classmethod
    def from_json(cls, path: str):
        import json
        with open(path, 'r') as file:
            config_dict = json.load(file)
            return cls(**config_dict)

    def to_py(self, path: str):
        with open(path, 'w') as file:
            file.write(f'config = {self.__repr__()}')

    @classmethod
    def from_py(cls, path: str):
        namespace = {}
        with open(path, 'r') as file:
            exec(file.read(), globals(), namespace)

        # Fetch the config from the namespace
        config = namespace.get('config', None)

        if config is None or not isinstance(config, cls):
            raise ValueError("Python file does not define a valid 'config' of type Config")

        return config
