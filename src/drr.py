# Import libraries
import sys

sys.path.append("..")

import torch

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert


def create_drr(
    volume,
    segmentation,
    bone_attenuation_multiplier=5.0,
    sdd=1020,
    height=200,
    width=200,
    delx=2.0,
    dely=2.0,
    x0=0,
    y0=0,
    p_subsample=None,
    reshape=True,
    reverse_x_axis=True,
    patch_size=None,
    renderer="siddon",
    rotations=torch.tensor([[0.0, 0.0, 0.0]]),
    rotations_degrees=True,
    translations=torch.tensor([[0.0, 850.0, 0.0]]),
    mask_to_channels=True,
    device="cpu",
):

    # Read the image and segmentation subject
    subject = read(
        tensor=volume,
        label_tensor=segmentation,
        orientation="AP",
        bone_attenuation_multiplier=bone_attenuation_multiplier,
    )

    # Create a DRR object
    drr = DRR(
        subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
        sdd=sdd,  # Source-to-detector distance (i.e., the C-arm's focal length)
        height=height,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        width=width,  # Width of the DRR
        delx=delx,  # Pixel spacing (in mm)
        dely=dely,  # Pixel spacing (in mm)
        x0=x0,  # # Principal point X-offset
        y0=y0,  # Principal point Y-offset
        p_subsample=p_subsample,  # Proportion of pixels to randomly subsample
        reshape=reshape,  # Return DRR with shape (b, 1, h, w)
        reverse_x_axis=reverse_x_axis,  # If True, obey radiologic convention (e.g., heart on right)
        patch_size=patch_size,  # Render patches of the DRR in series
        renderer=renderer,  # Rendering backend, either "siddon" or "trilinear"
    ).to(device)

    # Ensure rotations are in radians
    if rotations_degrees:
        rotations = torch.deg2rad(rotations)

    zero = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    pose1 = convert(
        zero, translations, parameterization="euler_angles", convention="ZXY"
    )
    pose2 = convert(rotations, zero, parameterization="euler_angles", convention="ZXY")
    pose = pose1.compose(pose2)

    img = drr(pose, mask_to_channels=mask_to_channels)
    return img
