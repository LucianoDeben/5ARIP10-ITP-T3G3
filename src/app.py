import random

# Import libraries
import sys

import numpy as np
import streamlit as st
import torch
import torchio as tio
from PIL import Image
from torchvision.transforms import ToTensor

from drr import create_drr
from model import TACEnet

sys.path.append("..")

import matplotlib.pyplot as plt
import pyvista
import torch
from monai.losses import DiceCELoss
from torchvision import transforms
from tqdm import tqdm

from diffdrr.data import load_example_ct, read, transform_hu_to_density
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.visualization import drr_to_mesh, img_to_mesh, plot_drr, plot_mask
from preprocessing import (
    add_vessel_contrast,
    get_dataloaders,
    get_datasets,
    get_transforms,
)
from vizualization import plot_drr_enhancement, plot_results

# Load example CT data
subject = load_example_ct(bone_attenuation_multiplier=1.0)

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drr = DRR(
    subject,  # A torchio.Subject object storing the CT volume, origin, and voxel spacing
    sdd=1020,  # Source-to-detector distance (i.e., the C-arm's focal length)
    height=200,  # Height of the DRR (if width is not seperately provided, the generated image is square)
    delx=2.0,  # Pixel spacing (in mm)
).to(device)

# Specify the C-arm pose with a rotation (yaw, pitch, roll) and orientation (x, y, z)
rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
xyz = torch.tensor([[0.0, 850.0, 0.0]], device=device)
img = drr(rot, xyz, parameterization="euler_angles", convention="ZXY")
# Display the DRR
axs = plot_drr(img, ticks=False)
fig = axs[0].figure
st.pyplot(fig)

# Load your model
model = TACEnet()
model.load_state_dict(torch.load("models/TACEnet_vessel_enhancement_1_0.pth"))
model.eval()

# Set the demo title
st.title("DRR Enhancement Model Demo")

rotation = st.slider(
    label="Select rotation",
    min_value=-45,
    max_value=45,
    value=0,
    step=1,
    key="rotation_slider",
    help="Select the rotation angle in degrees",
    on_change=None,
    label_visibility="visible",
)

prediction = model(subject)
