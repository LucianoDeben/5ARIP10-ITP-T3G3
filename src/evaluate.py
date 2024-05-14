import torch
from torch.utils.data import DataLoader

from model import SegFormer3D
from preprocessing import get_dataloaders, get_datasets, get_transforms


def evaluate(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    return accuracy


def main():
    model = SegFormer3D()
    model.load_state_dict(torch.load("../models/model.pth"))

    transform = get_transforms()

    train_dataset, val_dataset = get_datasets(transform)

    _, val_loader = get_dataloaders(train_dataset, val_dataset)

    accuracy = evaluate(model, val_loader)
    print(f"Validation accuracy: {accuracy}")


if __name__ == "__main__":
    main()
