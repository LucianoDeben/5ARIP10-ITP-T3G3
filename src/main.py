import sys

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from torch.optim.lr_scheduler import LambdaLR

import wandb
from config import config
from model import SegFormer3D
from preprocessing import get_dataloaders, get_datasets, get_transforms
from train_validate import train, validate

sys.path.append("../src")


def lr_lambda(epoch):
    if config.num_epochs == config.warmup_scheduler["warmup_epochs"]:
        raise ValueError(
            "config.num_epochs must be greater than config.warmup_scheduler['warmup_epochs']"
        )
    if epoch < config.warmup_scheduler["warmup_epochs"]:
        return epoch / config.warmup_scheduler["warmup_epochs"]
    else:
        return (
            1
            - (epoch - config.warmup_scheduler["warmup_epochs"])
            / (config.num_epochs - config.warmup_scheduler["warmup_epochs"])
        ) ** 0.9


def main():

    # Get the transforms
    transform = get_transforms(
        resize_shape=config["resize_shape"], contrast_value=config["contrast_value"]
    )

    # Get the datasets
    train_dataset, val_dataset = get_datasets(
        root_dir="../data",
        collection="HCC-TACE-Seg",
        transform=transform,
        download=True,
        download_len=2,
        seg_type="SEG",
        val_frac=config["val_frac"],
        seed=config["seed"],
    )

    # Get the dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = SegFormer3D().to(device)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for training.")
        model = nn.parallel.DistributedDataParallel(model)

    # Initialize the criterion
    criterion = DiceCELoss(include_background=False, sigmoid=True, to_onehot_y=False)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    wandb.watch(model, log_freq=100)

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Setup the model training loop
    for epoch in range(config["num_epochs"]):

        # Train the model
        train_loss, train_acc, train_dice, train_iou = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=config["num_classes"],
            grad_accum_steps=config["grad_accum_steps"],
        )

        # Validate the model
        val_loss, val_acc, val_dice, val_iou = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            num_classes=config["num_classes"],
        )

        print(train_dice, val_dice)

        # Log the results with the wandb logger
        wandb.log(
            {
                "epcoh": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
        )

        # Step the learning rate scheduler
        scheduler.step()

    # Save model
    torch.save(model.state_dict(), "../models/model.pth")


if __name__ == "__main__":
    main()
