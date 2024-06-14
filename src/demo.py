import sys

sys.path.append("..")
import torch
import torchio as tio

from diffdrr.data import read, transform_hu_to_density
from src.drr import create_drr
from src.model import TACEnet
from src.preprocessing import (
    add_vessel_contrast,
    get_dataloaders,
    get_datasets,
    get_transforms,
)
from src.vizualization import plot_results


def load_model(deformation, device):
    model_path = (
        "../models/TACEnet_vessel_enhancement_deformations_06062024.pth"
        if deformation
        else "../models/TACEnet_vessel_enhancement_29052024.pth"
    )
    model = TACEnet().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def apply_deformation(subject):
    deform = tio.RandomElasticDeformation(
        p=1.0, num_control_points=7, max_displacement=50
    )
    subject = deform(subject)
    subject.density = transform_hu_to_density(subject.volume.data, 5.0)
    return subject


def generate_drr(subject, rotation, ef, device):
    drr_raw = create_drr(
        subject,
        sdd=1020,
        height=256,
        width=256,
        rotations=torch.tensor([[rotation, 0.0, 0.0]]),
        translations=torch.tensor([[0.0, 850.0, 0.0]]),
        mask_to_channels=True,
        device=device,
    )
    drr_body, drr_vessels = drr_raw[:, 0], drr_raw[:, 1]
    drr_combined = (drr_body + ef * drr_vessels).unsqueeze(0).to(device)
    return drr_combined, drr_vessels.to(device)


def demonstration(rotation, ef, deformation=False, initial_contrast=4000, device="cpu"):
    transform = get_transforms(
        resize_shape=[512, 512, 96], contrast_value=initial_contrast
    )
    train_ds, _ = get_datasets(
        root_dir="../data081",
        collection="HCC-TACE-Seg",
        seg_type="SEG",
        transform=transform,
        download=False,
        download_len=1,
        val_frac=0.2,
        seed=42,
    )
    train_loader, _ = get_dataloaders(train_ds, _, batch_size=1)
    batch = next(iter(train_loader))
    volumes, targets = batch["image"], batch["seg"]
    volumes = add_vessel_contrast(volumes, targets, contrast_value=initial_contrast)

    model = load_model(deformation, device)
    subject = read(
        tensor=volumes[0],
        label_tensor=targets[0],
        orientation="AP",
        bone_attenuation_multiplier=5.0,
    )

    if deformation:
        subject = apply_deformation(subject)
    drr_combined, drr_target = generate_drr(subject, rotation, ef, device)

    prediction, latent = model(targets.to(device), drr_combined)
    plot_results(drr_combined, drr_target, prediction, latent, vmax=25)
