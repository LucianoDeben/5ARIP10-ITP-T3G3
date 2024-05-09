import wandb

wandb.init(
    project="5ARIP10-T3G3",
    entity="luciano-deben",
    config={
        "learning_rate": 3e-5,
        "batch_size": 1,
        "num_workers": 0,
        "num_epochs": 50,
        "num_classes": 5,
        "grad_accum_steps": 1,
        "val_frac": 0.2,
        "seed": 42,
        "warmup_scheduler": {"warmup_epochs": 1},
        "contrast_value": 1000,
        "resize_shape": [256, 256, 48],
    },
)

config = wandb.config
