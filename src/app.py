import os
import sys

import streamlit as st
import torch

# Now you can import from the submodule
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from drr import create_drr
from model import TACEnet
from training import loadData, sampleVolume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the page configuration
st.set_page_config(page_title="DRR Enhancement Model Demo", layout="wide")


# Caching the model loading to avoid reloading the model on each interaction
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    model = TACEnet()
    model.load_state_dict(
        torch.load("../models/TACEnet_vessel_enhancement_deformations_30052024.pth")
    )
    model.eval()
    return model


# Caching the DRR generation to avoid redundant computations
@st.cache_data(show_spinner="Generating DRR...")
def cached_create_drr(
    _train_loader, contrast_value, height=256, width=256, sdd=1020, rotation=0
):
    _volume, _target = sampleVolume(_train_loader, contrast_value=contrast_value)
    return (
        create_drr(
            _volume[0],
            _target[0],
            device="cpu",
            height=height,
            width=width,
            sdd=sdd,
            mask_to_channels=False,
            rotations=torch.tensor([[rotation, 0.0, 0.0]]),
        ),
        _volume,
        _target,
    )


@st.cache_data
def cached_loadData():
    return loadData()


@st.cache_data
def cached_sampleVolume(_train_loader, contrast_value):
    return sampleVolume(_train_loader, contrast_value=contrast_value)


# Show dialog
@st.experimental_dialog("Accept terms and conditions")
def terms_conditions():
    st.write(
        """
        By using this application, you agree to the following terms and conditions:

        - This application is for demonstration purposes only.
        - The results generated by this application are not intended for clinical use.
        - The model used in this application is trained on synthetic data and may not generalize well to real-world data.
        - The authors of this application are not responsible for any damages or losses resulting from the use of this application.
        """
    )
    st.write("Do you agree to the terms and conditions?")
    if st.button("I agree"):
        st.session_state.terms_conditions = True
        st.rerun()


if "terms_conditions" not in st.session_state:
    terms_conditions()

st.logo("https://avatars.githubusercontent.com/u/8323854?s=200&v=4")

# Display the DRR and enhanced DRR in the main layout
st.title("DRR Enhancement Model Demo")

st.divider()

with st.sidebar:
    st.header("Settings")

    "---"

    st.slider(
        label="Select rotation",
        min_value=-45,
        max_value=45,
        value=0,
        step=5,
        key="rotation_slider",
        help="Select the rotation angle in degrees",
        on_change=None,
        label_visibility="visible",
    )

    st.slider(
        label="Select initial contrast",
        min_value=0,
        max_value=4000,
        value=0,
        step=500,
        key="contrast_slider",
        help="Select the intital contrast",
        on_change=None,
        label_visibility="visible",
    )

    "---"

# Load your model
model = load_model()
model.to(device)

# Load CT data
train_loader, val_loader = cached_loadData()
# volume, target = cached_sampleVolume(
#     train_loader, contrast_value=st.session_state.contrast_slider
# )

# Initialize the DRR module for generating synthetic X-rays
drr, volume, target = cached_create_drr(
    train_loader,
    contrast_value=st.session_state.contrast_slider,
    rotation=st.session_state.rotation_slider,
)


col1, col2, col3 = st.columns(spec=[0.4, 0.2, 0.4])

with col1:
    st.header("Original DRR")
    axs = plot_drr(drr, ticks=False)
    fig = axs[0].figure
    fig.set_size_inches(2, 2)  # Adjust the figure size
    st.pyplot(fig)

with col3:
    st.header("Enhanced DRR")
    enhanced_placeholder = st.empty()  # Placeholder for enhanced image

with col2:
    # Add an "Enhance" button
    if st.button("Enhance"):
        with st.spinner("Enhancing DRR..."):
            prediction, latent = model(
                volume[0].unsqueeze(0).to(device), drr.to(device)
            )
            axs = plot_drr(prediction, ticks=False)
            fig = axs[0].figure
            fig.set_size_inches(2, 2)  # Adjust the figure size
            enhanced_placeholder.pyplot(fig)

col4, col5, col6 = st.columns(3)
col4.metric("Temperature", "70 °F", "1.2 °F")
col5.metric("Wind", "9 mph", "-8%")
col6.metric("Humidity", "86%", "4%")
