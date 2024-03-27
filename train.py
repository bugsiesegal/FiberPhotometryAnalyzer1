from model import AutoencoderModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder, BatchSizeFinder
from config import Config
from data import FiberTrackingDataModule
import wandb


def train():
    wandb.init(project='fiber-tracking')
    # Create a config object
    config = Config()
    config.data_dir = '/workspace/data/December 2023/'
    config.batch_size = 16
    try:
        config.nhead = wandb.config.nhead
        config.num_layers = wandb.config.num_layers
        config.latent_dim = wandb.config.latent_dim
        config.dropout = wandb.config.dropout
        config.window_dim = wandb.config.window_dim
    except AttributeError:
        config.nhead = 26
        config.num_layers = 6
        config.latent_dim = 16
        config.dropout = 0.2
        config.window_dim = 1000
    config.num_workers = 4

    # Create a model
    model = AutoencoderModule(config)

    # Create a data module
    data_module = FiberTrackingDataModule(config)

    # Create a trainer
    trainer = Trainer(
        logger=[WandbLogger(project='fiber-tracking', log_model="all")],
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),
            BatchSizeFinder()
        ],
        max_time={'hours': 10},
        precision="16-mixed",
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the model
    trainer.save_checkpoint('model.ckpt')

    # Test the model
    trainer.test(model, data_module)


if __name__ == '__main__':
    train()
