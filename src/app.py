import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import torch

from demo import demonstration, get_demo_data, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the page configuration
st.set_page_config(page_title="Tacetastic", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def cached_load_model(deformation, device):
    return load_model(deformation, device)


@st.cache_data(show_spinner="Collecting the data...")
def cached_get_demo_data():
    return get_demo_data()


@st.cache_data(show_spinner="Generating the results...")
def cached_demonstration(
    _volumes, _targets, _model, rotation, ef, deformation, initial_contrast, device
):
    # Your existing code to generate the figure
    fig = demonstration(
        _volumes, _targets, _model, rotation, ef, deformation, initial_contrast, device
    )
    return fig


st.session_state.deformation_checkbox = False

# Load the model
model = cached_load_model(
    deformation=st.session_state.deformation_checkbox, device=device
)

# Load the demo data
volumes, targets = cached_get_demo_data()


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
        label="Select contrast agent reduction ratio",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        key="contrast_slider",
        help="Select the contrast reduction ratio for the vessels",
        on_change=None,
        label_visibility="visible",
    )

    st.checkbox(
        label="Enable deformation",
        value=st.session_state.deformation_checkbox,
        key="deformation_checkbox",
        help="Enable deformation of the subject",
    )

    "---"

if st.sidebar.button("Generate"):
    # Assuming demonstration now returns a figure directly
    fig = cached_demonstration(
        _volumes=volumes,
        _targets=targets,
        _model=model,
        rotation=st.session_state.rotation_slider,
        ef=st.session_state.contrast_slider,
        deformation=st.session_state.deformation_checkbox,
        initial_contrast=4000,
        device=device,
    )

    # Display the figure in Streamlit's main view
    st.pyplot(fig)
