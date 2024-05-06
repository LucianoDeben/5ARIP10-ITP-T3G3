import sys

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from config import config
from preprocessing import get_dataloaders, get_datasets, get_transforms
from src.train_validate import train, validate

sys.path.append("..")


# Define main function
def main():

    # Get the transforms
    transform = get_transforms(contrast_value=1000)

    # Get the datasets
    train_dataset, val_dataset = get_datasets(
        root_dir="../data",
        collection="HCC-TACE-Seg",
        transform=transform,
        download=True,
        download_len=5,
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
    model = UNet(
        in_channels=1,
        out_channels=20,
        spatial_dims=(512, 512, 96),
        channels=5,
        strides=1,
    ).to(device)

    # Initialize the criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initialize the best validation loss
    best_val_loss = float("inf")

    # Train the model
    for epoch in range(config.num_epochs):
        train_loss, train_acc, train_dice, train_iou = train(
            train_loader, model, criterion, optimizer, device
        )
        val_loss, val_acc, val_dice, val_iou = validate(
            val_loader, model, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Train Dice: {train_dice:.4f}, "
            f"Train IoU: {train_iou:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Val Dice: {val_dice:.4f}, "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")


if __name__ == "__main__":
    main()
