import os

import matplotlib.pyplot as plt
import torch

from models.lightning_models import *
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder, BatchSizeFinder, \
    ModelSummary
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
from config import Config
from data import FiberTrackingDataModule
import wandb
import click


# Define the command line interface
@click.command()
@click.option('--config', '-c', default='config.yaml', help='Path to the configuration file.')
@click.option('--wandb_enabled', '-w', default=False, is_flag=True, help='Enable Weights & Biases logging.')
@click.option('--debug_run', '-d', default=False, is_flag=True, help='Run the model in debug mode.')
@click.option('--checkpoint', '-cp', default=None, help='Path to the model checkpoint to load.')
@click.option('--profile', '-p', default=False, is_flag=True, help='Enable PyTorch profiler.')
@click.option('--lr_finder', '-lr', default=False, is_flag=True, help='Enable learning rate finder.')
@click.option('--batch_finder', '-b', default=False, is_flag=True, help='Enable batch size finder.')
def main(config, wandb_enabled, debug_run, checkpoint, profile, lr_finder, batch_finder):
    # Load the configuration
    if config.endswith('.json'):
        config = Config.from_json(config)
    elif config.endswith('.yaml'):
        config = Config.from_yaml(config)
    elif config.endswith('.py'):
        config = Config.from_py(config)

    # Initialize the data module
    data_module = FiberTrackingDataModule(config)

    # Initialize the model
    if config.model == 'transformer_v1':
        if checkpoint is not None:
            model = TransformerAutoencoderModule_1.load_from_checkpoint(checkpoint, config=config)
        else:
            model = TransformerAutoencoderModule_1(config)
    elif config.model == 'fft_transformer_v1':
        if checkpoint is not None:
            model = FFTAutoencoderModule_V1.load_from_checkpoint(checkpoint, config=config)
        else:
            model = FFTAutoencoderModule_V1(config)
    elif config.model == 'transformer_v2':
        if checkpoint is not None:
            model = TransformerAutoencoderModule_2.load_from_checkpoint(checkpoint, config=config)
        else:
            model = TransformerAutoencoderModule_2(config)
    elif config.model == 'transformer_v3':
        if checkpoint is not None:
            model = TransformerAutoencoderModule_3.load_from_checkpoint(checkpoint, config=config)
        else:
            model = TransformerAutoencoderModule_3(config)
    elif config.model == 'transformer_v4':
        if checkpoint is not None:
            model = TransformerAutoencoderModule_4.load_from_checkpoint(checkpoint, config=config)
        else:
            model = TransformerAutoencoderModule_4(config)
    elif config.model == 'sparse':
        if checkpoint is not None:
            model = SparseAutoencoderModule.load_from_checkpoint(checkpoint, config=config)
        else:
            model = SparseAutoencoderModule(config)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    # Initialize the trainer
    trainer = Trainer(
        logger=[WandbLogger(project='fiber-tracking', config=config, log_model=True),
                TensorBoardLogger(save_dir=os.path.join(os.getcwd(), "logs"), log_graph=True)
                ] if wandb_enabled and not debug_run else [],
        callbacks=[
            ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath=os.path.join(os.getcwd(), "checkpoints")),
            # EarlyStopping(monitor='val_loss', patience=config.lr_patience),
            BatchSizeFinder(mode="binsearch"),
            ModelSummary(max_depth=3)
        ] if batch_finder else [
            ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath=os.path.join(os.getcwd(), "checkpoints")),
            # EarlyStopping(monitor='val_loss', patience=config.lr_patience),
            ModelSummary(max_depth=3)
        ],
        max_epochs=config.max_epochs,
        precision=config.precision,
        max_time=str(config.max_time) if not debug_run else '00:00:05:00',
        limit_train_batches=5 if debug_run else 1.0,
        limit_val_batches=5 if debug_run else 1.0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value',
        profiler=PyTorchProfiler(
            dirpath=os.path.join(os.getcwd(), 'profiler'),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(os.getcwd(), 'logs', 'lightning_logs', 'profiler0')),
            schedule=torch.profiler.schedule(
                skip_first=50,
                wait=3,
                warmup=3,
                active=20,
                repeat=4
            ),
            profile_memory=True
        ) if profile else None
    )

    if lr_finder:
        # Initialize the tuner
        tuner = Tuner(trainer)

        # Find the optimal learning rate
        lr_finder = tuner.lr_find(model, data_module, min_lr=1e-8, max_lr=1e+4, num_training=1000, mode='exponential')

        # Log learning rate graph
        if wandb_enabled:
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

    # Encode the data
    encoded_data = trainer.predict(model, data_module)

    # Save the encoded data to disk as a numpy array
    np.save(os.path.join(os.getcwd(), 'encoded_data.npy'), encoded_data)

    # Close Weights & Biases
    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    main()
