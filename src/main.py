import sys

import torch
import torch.nn as nn

import wandb
from config import config
from model import SegFormer3D
from preprocessing import get_dataloaders, get_datasets, get_transforms
from train_validate import train, validate

sys.path.append("..")


# Define main function
def main():

    # Get the transforms
    transform = get_transforms(resize_shape=[256, 256, 48], contrast_value=1000)

    # Get the datasets
    train_dataset, val_dataset = get_datasets(
        root_dir="../data",
        collection="HCC-TACE-Seg",
        transform=transform,
        download=True,
        download_len=2,
        seg_type="SEG",
        val_frac=config.val_frac,
        seed=config.seed,
    )

    # Get the dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset, batch_size=1, num_workers=0
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
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, log_freq=100)

    # Setup the model training loop
    for epoch in range(config.num_epochs):

        # Train the model
        train_loss, train_acc, train_dice, train_iou = train(
            train_loader, model, criterion, optimizer, device
        )

        # Validate the model
        val_loss, val_acc, val_dice, val_iou = validate(
            val_loader, model, criterion, device
        )

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

    # Save model
    torch.save(model.state_dict(), "../models/model.pth")


if __name__ == "__main__":
    main()
