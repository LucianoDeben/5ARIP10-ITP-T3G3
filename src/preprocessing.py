from monai.apps import TciaDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    DataStatsd,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
)

from src.custom_transforms import (
    AddBackgroundChannel,
    AddVesselContrast,
    RemoveNecrosisChannel,
)


def get_transforms(resize_shape=[512, 512, 64]):
    """
    Create a composed transform for the data preprocessing of mask and image data

    Args:
        resize_shape: The shape to which the images are resized

    Returns:
        transform: The composed transform
    """
    # Create a composed transform
    transform = Compose(
        [
            LoadImaged(reader="PydicomReader", keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=[512, 512, 64]),
            RemoveNecrosisChannel(keys=["seg"]),
            Resized(keys=["image", "seg"], spatial_size=resize_shape),
            AddBackgroundChannel(keys=["seg"]),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-2048,
            #     a_max=1455,
            #     b_min=-170,
            #     b_max=230,
            #     clip=True,
            # ),
            AddVesselContrast(keys=["image", "seg"], contrast_value=200),
            DataStatsd(keys=["image", "seg"], data_shape=True),
        ],
        lazy=False,
    )
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
