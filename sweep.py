from lightning.pytorch.tuner import Tuner

from models.lightning_models import *
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder, BatchSizeFinder, \
    ModelSummary
from config import Config
from data import FiberTrackingDataModule
import wandb
import click

def main():
    # Initialize Weights & Biases
    wandb.init(project='fiber-tracking')

    config = wandb.config
    # Initialize the data module
    data_module = FiberTrackingDataModule(config)

    # Initialize the model
    if config.model == 'transformer_v1':
        model = TransformerAutoencoderModule_1(config)
    elif config.model == 'fft_transformer_v1':
        model = FFTAutoencoderModule_V1(config)
    elif config.model == 'transformer_v2':
        model = TransformerAutoencoderModule_2(config)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    # Initialize the trainer
    trainer = Trainer(
        logger=[WandbLogger(project='fiber-tracking', config=config)],
        callbacks=[
            ModelCheckpoint(monitor='val_loss', save_top_k=1),
            EarlyStopping(monitor='val_loss', patience=config.lr_patience),
            BatchSizeFinder(),
            ModelSummary(max_depth=3)
        ],
        max_epochs=config.max_epochs,
        precision=config.precision,
        max_time=str(config.max_time),
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value',
    )

    # Initialize the tuner
    tuner = Tuner(trainer)

    # Find the optimal learning rate
    lr_finder = tuner.lr_find(model, data_module, min_lr=1e-14, max_lr=1e+4, num_training=1000, mode='exponential')

    # Log learning rate graph
    wandb.log({'Learning Rate Finder': lr_finder.plot(suggest=True)})
    plt.clf()

    # Pick the optimal learning rate
    new_lr = lr_finder.suggestion()
    model.learning_rate = new_lr

    # Configure the optimizers learning rate
    trainer.optimizers[0].param_groups[0]['lr'] = new_lr

    # Fit the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    # Close Weights & Biases
    wandb.finish()


if __name__ == '__main__':
    main()
