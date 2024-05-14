import sys

import torch
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_jaccard_index,
    dice,
)
from tqdm import tqdm

sys.path.append("..")


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    grad_accum_steps=1,
):
    """
    Train the model for one epoch

    Args:
        train_dataloader: The training dataloader
        model: The model to train
        criterion: The loss function
        optimizer: The optimizer
        device: The device to use for training
        grad_accum_steps: The number of gradient accumulation steps

    Returns:
        train_loss: The average training loss
        train_acc: The average training accuracy
        train_dice: The average training dice score
        train_iou: The average training IoU score
    """
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    train_iou = 0.0

    scaler = GradScaler()

    for i, inputs in enumerate(tqdm(train_loader)):
        volumes, targets = inputs["image"], inputs["seg"]
        volumes, targets = volumes.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(volumes)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0 or i + 1 == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        outputs_max = (torch.sigmoid(outputs) > 0.5).float()

        with torch.no_grad():
            train_acc += binary_accuracy(
                outputs_max,
                targets,
            ).item()
            train_dice += dice(
                outputs_max,
                targets,
            ).item()
            train_iou += binary_jaccard_index(outputs_max, targets).item()

    num_batches = len(train_loader)
    return (
        train_loss / num_batches,
        train_acc / num_batches,
        train_dice / num_batches,
        train_iou / num_batches,
    )


def validate(val_loader, model, criterion, device):
    """
    Validate the model

    Args:
        val_loader: The validation dataloader
        model: The model to validate
        criterion: The loss function
        device: The device to use for validation

    Returns:
        val_loss: The average validation loss
        val_acc: The average validation accuracy
        val_dice: The average validation dice score
        val_iou: The average validation IoU score
    """
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for inputs in tqdm(val_loader):
            volumes, targets = inputs["image"], inputs["seg"]
            volumes, targets = volumes.to(device), targets.to(device)

            # Use autocast to enable mixed precision
            with autocast():
                outputs = model(volumes)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            outputs_max = (torch.sigmoid(outputs) > 0.5).float()

            val_acc += binary_accuracy(
                outputs_max,
                targets,
            ).item()
            val_dice += dice(
                outputs_max,
                targets,
            ).item()
            val_iou += binary_jaccard_index(outputs_max, targets).item()

    num_batches = len(val_loader)
    return (
        val_loss / num_batches,
        val_acc / num_batches,
        val_dice / num_batches,
        val_iou / num_batches,
    )
