import wandb

wandb.init(
    project="5ARIP10-T3G3",
    entity="luciano-deben",
    config={
        "learning_rate": 0.001,
        "batch_size": 4,
        "num_epochs": 10,
        "val_frac": 0.2,
        "seed": 42,
    },
)

config = wandb.config
