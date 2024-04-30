import torch

from model import AutoencoderModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder, BatchSizeFinder, ModelSummary
from config import Config
from data import FiberTrackingDataModule
import wandb


def train():
    run = wandb.init(project='fiber-tracking')
    # Create a config object
    config = Config()
    config.data_dir = '/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/datafiles'
    config.batch_size = 4
    try:
        config.nhead = wandb.config.nhead
        config.num_layers = wandb.config.num_layers
        config.latent_dim = wandb.config.latent_dim
        config.dropout = wandb.config.dropout
        config.window_dim = wandb.config.window_dim
    except AttributeError:
        config.input_features = 104
        config.d_model = 256
        config.nhead = 4
        config.num_layers = 2
        config.latent_dim = 64
        config.dropout = 0.1
        config.window_dim = 1024
        config.normalize = True
        config.activation = 'relu'
        config.learning_rate = 1e-3
        config.lr_factor = 0.1
        config.lr_patience = 5
        config.use_fiber = True
        config.use_tracking = True
        config.use_fft = True
    config.num_workers = 4

    # Create a model
    model = AutoencoderModule(config)

    # state_dict = torch.load('model.ckpt')
    #
    # model.load_state_dict(state_dict['state_dict'])

    # Create a data module
    data_module = FiberTrackingDataModule(config)

    # Create a trainer
    trainer = Trainer(
        logger=[WandbLogger(project='fiber-tracking', log_model="all")],
        callbacks=[
            ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='model-{epoch:02d}-{val_loss:.2f}'),
            LearningRateFinder(num_training_steps=100),
            BatchSizeFinder(),
            ModelSummary(max_depth=3)
        ],
        max_time={'hours': 10},
        precision="16-mixed",
        val_check_interval=0.25,
        # gradient_clip_val=1.0
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the model
    trainer.save_checkpoint('model.ckpt')

    # Test the model
    trainer.test(model, data_module)


if __name__ == '__main__':
    train()
