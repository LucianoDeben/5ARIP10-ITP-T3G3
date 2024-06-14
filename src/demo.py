
import sys
sys.path.append("..")
from src.preprocessing import get_transforms, get_datasets, get_dataloaders, add_vessel_contrast
from src.model import TACEnet
import torch
from diffdrr.data import read, transform_hu_to_density
import torchio as tio
from src.drr import create_drr
from src.vizualization import plot_results


def demonstration(rotation, ef, deformation = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Get transforms
    transform = get_transforms(resize_shape= [512, 512, 96], contrast_value=1000)

    # Get datasets
    train_ds, val_ds = get_datasets(
        root_dir="../data081",
        collection="HCC-TACE-Seg",
        transform=transform,
        download=False,
        val_frac=0.0,
        download_len=1,
        seed=42
    )

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=1)

    # Sample a batch of data from the dataloader
    batch = next(iter(train_loader))

    # Separate the image and segmentation from the batch
    volumes, targets = batch["image"], batch["seg"]
    volumes = add_vessel_contrast(volumes, targets, contrast_value=4000)

    model = TACEnet().to(device)
    
    if deformation:
        model.load_state_dict(torch.load("../models/TACEnet_vessel_enhancement_deformations_06062024.pth"))
    else:
        model.load_state_dict(torch.load("../models/TACEnet_vessel_enhancement_29052024.pth"))
    
    subject = read(tensor=volumes[0],
                label_tensor=targets[0],
                orientation="AP",
                bone_attenuation_multiplier=5.0,
                )
    
    if deformation:
        deform = tio.RandomElasticDeformation(p=1.0, num_control_points=7, max_displacement=50)

                # Apply the transform to the subject
        deformed_subject = deform(subject)

        deformed_subject.density = transform_hu_to_density(
                deformed_subject.volume.data, 5.0
                )
        subject = deformed_subject
    
    drr_raw = create_drr(
                subject,
                sdd=1020,
                height=256,
                width=256,
                rotations=torch.tensor([[rotation, 0.0, 0.0]]),
                translations=torch.tensor([[0.0, 850.0, 0.0]]),
                mask_to_channels=True,
                device="cpu",
                )
    
    drr_body = drr_raw[:, 0]
    drr_vessels = drr_raw[:, 1]
    drr_combined_low_enhancement = (drr_body + ef * drr_vessels).unsqueeze(0)
    drr_combined_low_enhancement = drr_combined_low_enhancement.to(device)
    drr_combined_target = drr_vessels.to(device)

    prediction, latent_representation = model(targets.to(device), drr_combined_low_enhancement)

    plot_results(
            drr_combined_low_enhancement, drr_combined_target, prediction,latent_representation, vmax=25
        )
    
    return