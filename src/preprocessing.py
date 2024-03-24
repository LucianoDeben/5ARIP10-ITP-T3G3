from monai.apps import TciaDataset
from monai.data import DataLoader
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              Resized, ResizeWithPadOrCropd)

from src.custom_transforms import AddBackgroundChannel, RemoveNecrosisChannel


def get_transforms():
    # Create a composed transform
    transform = Compose(
        [
            LoadImaged(reader="PydicomReader", keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=[512, 512, 64]),        
            RemoveNecrosisChannel(keys=["seg"]),
            AddBackgroundChannel(keys=["seg"]),
            Resized(keys=["image", "seg"], spatial_size=[64, 64, 64])
        ]
    )
    return transform


def get_datasets(root_dir="../data", collection = "HCC-TACE-Seg", seg_type="SEG", transform=None, download=True, download_len=-1, val_frac=0.2, seed=0):
    
    # Create a dataset for the training with a validation split
    train_dataset = TciaDataset(
        root_dir= root_dir,
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
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader