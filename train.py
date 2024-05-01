from models.lightning_models import *
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder, BatchSizeFinder, \
    ModelSummary
from config import Config
from data import FiberTrackingDataModule
import wandb
import click


# Define the command line interface
@click.command()
@click.option('--config', '-c', default='config.yaml', help='Path to the configuration file.')
@click.option('--wandb_enabled', '-w', default=False, is_flag=True, help='Enable Weights & Biases logging.')
def main(config, wandb_enabled):
    # Load the configuration
    if config.endswith('.json'):
        config = Config.from_json(config)
    elif config.endswith('.yaml'):
        config = Config.from_yaml(config)
    elif config.endswith('.py'):
        config = Config.from_py(config)

    # Initialize Weights & Biases
    if wandb_enabled:
        wandb.init(project='fiber-tracking', config=config)

    # Initialize the data module
    data_module = FiberTrackingDataModule(config)

    # Initialize the model
    if config.model == 'transformer_v1':
        model = TransformerAutoencoderModule_1(config)
    elif config.model == 'fft_transformer_v1':
        model = FFTAutoencoderModule_V1(config)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    # Initialize the trainer
    trainer = Trainer(
        logger=WandbLogger(project='fiber-tracking', config=config) if wandb_enabled else None,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', save_top_k=1),
            EarlyStopping(monitor='val_loss', patience=config.lr_patience),
            LearningRateFinder(),
            BatchSizeFinder(),
            ModelSummary(max_depth=3)
        ],
        max_epochs=config.max_epochs,
        precision=config.precision,
        max_time=config.max_time
    )

    # Fit the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    # Close Weights & Biases
    if wandb_enabled:
        wandb.finish()
