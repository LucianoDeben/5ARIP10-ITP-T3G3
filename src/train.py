import sys

import numpy as np
import torch
import torch.nn as nn
import tqdm
from monai.networks.nets import UNet
from torchmetrics.functional.classification import (
    accuracy,
    dice,
    multiclass_jaccard_index,
)

from preprocessing import get_dataloaders, get_datasets, get_transforms

sys.path.append("..")

from config import config


# Define the training loop
def train(train_dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    train_iou = 0.0

    for inputs, targets in tqdm(train_dataloader):
        targets = targets.long().squeeze(dim=1)
        inputs, targets = inputs.to(device), targets.to(device)

        print(inputs.shape)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach()
        outputs_max = torch.argmax(outputs, dim=1)

        train_acc += accuracy(
            outputs_max, targets, task="multiclass", num_classes=20, ignore_index=19
        ).detach()
        train_dice += dice(outputs, targets, ignore_index=19).detach()
        train_iou += multiclass_jaccard_index(
            outputs, targets, num_classes=20, ignore_index=19
        ).detach()

    num_batches = len(train_dataloader)
    return (
        (train_loss / num_batches).item(),
        (train_acc / num_batches).item(),
        (train_dice / num_batches).item(),
        (train_iou / num_batches).item(),
    )


# Define the validation loop
def validate(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    val_iou = 0.0

    for inputs, targets in tqdm(val_loader):
        targets = targets.long().squeeze(dim=1)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.detach()
        outputs_max = torch.argmax(outputs, dim=1)

        val_acc += accuracy(
            outputs_max, targets, task="multiclass", num_classes=20, ignore_index=19
        ).detach()
        val_dice += dice(outputs, targets, ignore_index=19).detach()
        val_iou += multiclass_jaccard_index(
            outputs, targets, num_classes=20, ignore_index=19
        ).detach()

    num_batches = len(val_loader)
    return (
        (val_loss / num_batches).item(),
        (val_acc / num_batches).item(),
        (val_dice / num_batches).item(),
        (val_iou / num_batches).item(),
    )
