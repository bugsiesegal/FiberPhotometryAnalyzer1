from dataclasses import dataclass, field


@dataclass
class Config:
    # Model
    d_model: int = 52
    window_dim: int = 1000
    nhead: int = 8
    dim_feedforward: int = 2048
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = 'relu'
    latent_dim: int = 128
    learning_rate: float = 1e-3
    lr_factor: float = 0.1
    lr_patience: int = 5

    # Data
    data_dir: str = field(default_factory=str)
    batch_size: int = 32
    num_workers: int = 4
    use_fiber: bool = True
    use_tracking: bool = True

    # Fiber
    fiber_channel_name: str = 'Analog In. | Ch.1 AIn-1 - Dem (AOut-2)'
    control_channel_name: str = 'Analog In. | Ch.1 AIn-1 - Dem (AOut-3)'
    normalize: bool = True
    