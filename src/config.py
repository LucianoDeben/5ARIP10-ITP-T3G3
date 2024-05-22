import wandb

wandb.init(
    project="5ARIP10-T3G3",
    entity="luciano-deben",
    config={
        "learning_rate": 3e-5,
        "batch_size": 1,
        "num_workers": 0,
        "num_epochs": 2,
        "num_classes": 1,
        "seg_task": "binary",
        "grad_accum_steps": 1,
        "val_frac": 0.0,
        "seed": 42,
        "weight_decay": 0.01,
        "warmup_scheduler": {"warmup_epochs": 0},
        "contrast_value": 1000,
        "resize_shape": [512, 512, 96],
    },
)

config = wandb.config
