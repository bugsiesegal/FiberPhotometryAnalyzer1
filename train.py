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
    config.data_dir = '/workspace/datafiles/December 2023/'
    config.batch_size = 16
    try:
        config.nhead = wandb.config.nhead
        config.num_layers = wandb.config.num_layers
        config.latent_dim = wandb.config.latent_dim
        config.dropout = wandb.config.dropout
        config.window_dim = wandb.config.window_dim
    except AttributeError:
        config.d_model = 52
        config.nhead = 26
        config.num_layers = 6
        config.latent_dim = 16
        config.dropout = 0.2
        config.window_dim = 1024
        config.normalize = True
        config.activation = 'gelu'
        config.learning_rate = 1e-3
        config.lr_factor = 0.1
        config.lr_patience = 5
        config.use_fiber = True
        config.use_tracking = False
    config.num_workers = 4

    # Load last checkpoint
    # artifact = run.use_artifact('bugsiesegal/fiber-tracking/model-irc0xjbn:v1', type='model')
    # artifact_dir = artifact.download()

    # Create a model
    model = AutoencoderModule(config)

    # Load the model
    # model = AutoencoderModule.load_from_checkpoint(artifact_dir + '/model.ckpt')

    # Create a data module
    data_module = FiberTrackingDataModule(config)

    # Create a trainer
    trainer = Trainer(
        logger=[WandbLogger(project='fiber-tracking', log_model="all")],
        callbacks=[
            ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='model-{epoch:02d}-{val_loss:.2f}'),
            LearningRateFinder(),
            BatchSizeFinder(),
            ModelSummary(max_depth=3)
        ],
        max_time={'hours': 10},
        precision="16-mixed",
        val_check_interval=0.25,
        gradient_clip_val=1.0
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the model
    trainer.save_checkpoint('model.ckpt')

    # Test the model
    trainer.test(model, data_module)


if __name__ == '__main__':
    train()
