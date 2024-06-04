import sys

from monai.apps import TciaDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    DataStatsd,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    LoadImaged,
    Rand3DElasticd,
    RandFlipd,
    RandGaussianNoised,
    RandGridDistortiond,
    RandRotated,
    RandScaleIntensityd,
    Resized,
    ResizeWithPadOrCrop,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
)

from custom_transforms import (
    AddBackgroundChannel,
    AddVesselContrast,
    ClassIsolation,
    ConvertToSingleChannel,
    RemoveDualImage,
    RemoveNecrosisChannel,
)


def get_transforms(resize_shape=[256, 256, 48], is_train=True):
    """
    Create a composed transform for the data preprocessing of mask and image data

    Args:
        resize_shape: The shape to which the images are resized

    Returns:
        transform: The composed transform
    """

    if is_train:

        # Create a composed transform
        train_transform = Compose(
            [
                LoadImaged(reader="PydicomReader", keys=["image", "seg"]),
                EnsureChannelFirstd(keys=["image", "seg"]),
                ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=resize_shape),
                ClassIsolation(keys=["seg"], class_index=2),
            ],
            lazy=False,
        )
        return train_transform
    else:
        val_transform = Compose(
            [
                LoadImaged(reader="PydicomReader", keys=["image", "seg"]),
                EnsureChannelFirstd(keys=["image", "seg"]),
                ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=resize_shape),
                RemoveNecrosisChannel(keys=["seg"]),
                AddBackgroundChannel(keys=["seg"]),
                ClassIsolation(keys=["seg"], class_index=2),
            ],
            lazy=False,
        )
        return val_transform


def get_eval_transforms(resize_shape=[256, 256, 48], contrast_value=1000):
    transform = ResizeWithPadOrCrop(resize_shape, lazy=False)
    return transform


def get_datasets(
    root_dir="../data",
    collection="HCC-TACE-Seg",
    seg_type="SEG",
    transform=None,
    download=True,
    download_len=-1,
    val_frac=0.2,
    seed=0,
):
    """
    Create the training and validation datasets for the given TCIA collection

    Args:
        root_dir: The root directory where the data is stored
        collection: The TCIA collection to use
        seg_type: The segmentation type to use
        transform: The transform to apply to the data
        download: Whether to download the data
        download_len: The number of files to download
        val_frac: The fraction of the data to use for validation
        seed: The seed for the random number generator

    Returns:
        train_dataset: The training dataset
        val_dataset: The validation dataset
    """

    # Create a dataset for the training with a validation split
    train_dataset = TciaDataset(
        root_dir=root_dir,
        collection=collection,
        section="training",
        transform=transform,
        download=download,
        download_len=download_len,
        seg_type=seg_type,
        progress=True,
        cache_rate=0.0,
        val_frac=val_frac,
        seed=seed,
    )

    # Create the corresponding validation dataset
    val_dataset = TciaDataset(
        root_dir="../data",
        collection=collection,
        section="validation",
        transform=transform,
        download=download,
        download_len=download_len,
        seg_type=seg_type,
        progress=True,
        cache_rate=0.0,
        val_frac=val_frac,
        seed=seed,
    )

    return train_dataset, val_dataset


def get_dataloaders(train_dataset, val_dataset, batch_size=1, num_workers=0):
    """
    Create the training and validation dataloaders for the given datasets

    Args:
        train_dataset: The training dataset
        val_dataset: The validation dataset
        batch_size: The batch size
        num_workers: The number of workers to use for loading the data

    Returns:
        train_loader: The training dataloader
        val_loader: The validation dataloader
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def add_vessel_contrast(image, seg, contrast_value):
    """
    Increase the contrast of vessels in an image using the vessel segmentation mask.

    Parameters:
    image (torch.Tensor): The input image.
    seg (torch.Tensor): The segmentation mask.
    contrast_value (float): The value to add to the vessel pixels in the image.

    Returns:
    torch.Tensor: The image with increased vessel contrast.
    """
    # Check the number of dimensions in the segmentation mask
    if len(seg.shape) == 5:  # Batched input
        # Check the number of channels in the segmentation mask
        if seg.shape[1] == 1:
            # If there's only one channel, the vessels are at index 0
            vessel_mask = seg[:, 0]
        else:
            # Otherwise, the vessels are at index 3
            vessel_mask = seg[:, 3]
        # Add an extra dimension for the channels
        vessel_mask = vessel_mask.unsqueeze(1)
    else:  # Non-batched input
        # Check the number of channels in the segmentation mask
        if seg.shape[0] == 1:
            # If there's only one channel, the vessels are at index 0
            vessel_mask = seg[0]
        else:
            # Otherwise, the vessels are at index 3
            vessel_mask = seg[3]
        # Add an extra dimension for the channels
        vessel_mask = vessel_mask.unsqueeze(0)

    # Add the vessel contrast to the image
    image[vessel_mask == 1] += contrast_value
    return image
