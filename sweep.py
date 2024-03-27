from train import train
import wandb

swep_config = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "nhead": {"values": [2, 4, 13, 26, 52]},
        "num_layers": {"values": [2, 4, 6, 8, 10]},
        "latent_dim": {"values": [32, 64, 128, 256]},
        "dropout": {"values": [0.1, 0.2, 0.3, 0.4]},
        "window_dim": {"values": [500, 1000, 2000, 4000, 10000]},
    }
}

wandb.sweep(swep_config, project='fiber-tracking', entity='bugsie')
wandb.agent('sweep_id', function=train)