# Import libraries
import sys

sys.path.append("..")

import torch

from diffdrr.drr import DRR, convert


def create_drr(
    subject,
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
    """
    Create a digitally reconstructed radiograph (DRR) from a CT volume and segmentation

    Args:
        volume (torch.Tensor): The CT volume
        segmentation (torch.Tensor): The segmentation
        bone_attenuation_multiplier (float): The bone attenuation multiplier
        sdd (int): The source-to-detector distance
        height (int): The height of the DRR
        width (int): The width of the DRR
        delx (float): The pixel spacing in the X-direction
        dely (float): The pixel spacing in the Y-direction
        x0 (int): The principal point X-offset
        y0 (int): The principal point Y-offset
        p_subsample (float): The proportion of pixels to randomly subsample
        reshape (bool): Whether to reshape the DRR to (b, 1, h, w)
        reverse_x_axis (bool): Whether to reverse the X-axis
        patch_size (int): The size of the patches to render
        renderer (str): The rendering backend
        rotations (torch.Tensor): The rotations to apply
        rotations_degrees (bool): Whether the rotations are in degrees
        translations (torch.Tensor): The translations to apply
        mask_to_channels (bool): Whether to convert the mask to channels
        device (str): The device to use

    Returns:
        img (torch.Tensor): The DRR
    """

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

    if rotations_degrees:
        rotations = torch.deg2rad(rotations)

    # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
    zero = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations.to(device)
    rotations.to(device)

    # Convert the rotations and translations to a pose matrix
    pose1 = convert(
        zero, translations, parameterization="euler_angles", convention="ZXY"
    ).to(device)
    pose2 = convert(
        rotations, zero, parameterization="euler_angles", convention="ZXY"
    ).to(device)
    pose = pose1.compose(pose2)

    # Create the DRR image tensor object
    img = drr(
        pose,
        mask_to_channels=mask_to_channels,
    )
    return img
