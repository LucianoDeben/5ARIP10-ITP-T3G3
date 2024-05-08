import sys

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.functional.classification import (
    accuracy,
    dice,
    multiclass_jaccard_index,
)
from tqdm import tqdm

sys.path.append("..")


from config import config


def train(train_dataloader, model, criterion, optimizer, device, grad_accum_steps=1):
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

    for i, inputs in enumerate(tqdm(train_dataloader)):
        volumes, targets = inputs["image"], inputs["seg"]
        targets = targets.squeeze(dim=1).long()
        volumes, targets = volumes.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(volumes)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0 or i + 1 == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        outputs_max = torch.argmax(outputs, dim=1)

        with torch.no_grad():
            train_acc += accuracy(
                outputs_max, targets, task="multiclass", num_classes=5, ignore_index=4
            ).item()
            train_dice += dice(outputs, targets, ignore_index=4).item()
            train_iou += multiclass_jaccard_index(
                outputs, targets, num_classes=5, ignore_index=4
            ).item()

    num_batches = len(train_dataloader)
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
            targets = targets.squeeze(dim=1).long()
            volumes, targets = volumes.to(device), targets.to(device)

            # Use autocast to enable mixed precision
            with autocast():
                outputs = model(volumes)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            outputs_max = torch.argmax(outputs, dim=1)

            val_acc += accuracy(
                outputs_max, targets, task="multiclass", num_classes=5, ignore_index=4
            ).item()
            val_dice += dice(outputs, targets, ignore_index=4).item()
            val_iou += multiclass_jaccard_index(
                outputs, targets, num_classes=5, ignore_index=4
            ).item()

    num_batches = len(val_loader)
    return (
        val_loss / num_batches,
        val_acc / num_batches,
        val_dice / num_batches,
        val_iou / num_batches,
    )
