# Import libraries
import sys

import streamlit as st
import torch

from model import TACEnet

sys.path.append("..")
import torch

from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from preprocessing import get_eval_transforms

# Load example CT data
subject = load_example_ct(bone_attenuation_multiplier=1.0)

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drr = DRR(
    subject,
    sdd=1020,
    height=200,
    delx=2.0,
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
model.load_state_dict(
    torch.load("models/TACEnet_vessel_enhancement_deformations_30052024.pth")
)
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

# Transform the subject to match model input size
eval_transform = get_eval_transforms(resize_shape=[512, 512, 96])
input = eval_transform(subject.volume.data)

prediction = model(input.unsqueeze(0), drr)
