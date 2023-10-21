import wandb
from train import train

wandb.login()

sweep_config = {
    'method': 'bayes',  # Here we are using Bayesian optimization
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   # You might want to change the metric and goal as per your requirement
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.000001,
            'max': 0.1
        },
        'window_size': {
            'value': 100
        },
        'embedding_size': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 8
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 2,  # Adjust this range based on your requirements
            'max': 12
        },
        'num_workers': {
            'value': 16
        },
    }
}

if __name__=="__main__":
    sweep_id = wandb.sweep(sweep_config, project="JAAEC_Fiberphotometry")
    print(sweep_id)
    wandb.agent(sweep_id, function=train, count=100)
