import random
import time

import streamlit as st

st.set_page_config(page_title="Surgeon Demo", page_icon="👋")

# Mock Patient Details
patient_info = {
    "Name": "John Doe",
    "Age": 34,
    "Gender": "Male",
    "MRN": "123456789",
    "Admission Date": "2023-04-01",
    "Diagnosis": "Appendicitis",
    "Surgeon Assigned": "Dr. Jane Smith",
    "Procedure": "TACE",
}

# Vital Signs (Mock, could be dynamically updated)
vital_signs = {
    "SpO2": "98%",
    "Blood Pressure": "120/80 mmHg",
    "Respiratory Rate": "16 breaths/min",
}

# Display Patient Information in Sidebar
st.sidebar.title("Patient Information")
for key, value in patient_info.items():
    st.sidebar.text(f"{key}: {value}")

st.sidebar.title("Vital Signs")
for key, value in vital_signs.items():
    st.sidebar.text(f"{key}: {value}")

# Initialize session state variables if they don't exist
if "rotation" not in st.session_state:
    st.session_state.rotation = -45  # Start at 0 degrees
if "rotation_direction" not in st.session_state:
    st.session_state.rotation_direction = 1  # Start moving towards -45
if "confidence" not in st.session_state:
    st.session_state.confidence = 98  # Start with high confidence
if "spo2" not in st.session_state:
    st.session_state.spo2 = random.randint(
        80, 88
    )  # Random SpO2 value between 94% and 98%

st.image(r"D:\Programming\AI\5ARIP10-ITP-T3G3\src\pages\xray_demo.gif", width=400)


# Placeholder for dynamic metrics
col1, col2, col3 = st.columns(3)
rotation_placeholder = col1.empty()
confidence_placeholder = col2.empty()
spo2_placeholder = col3.empty()

while True:
    # Update rotation
    if st.session_state.rotation_direction == 1:
        if st.session_state.rotation + 10 <= 45:
            st.session_state.rotation += 10
        else:
            st.session_state.rotation = 45  # Correct to max 45 if overshooting
            st.session_state.rotation_direction = -1
    else:
        if st.session_state.rotation - 10 >= -45:
            st.session_state.rotation -= 10
        else:
            st.session_state.rotation = -45  # Correct to min -45 if overshooting
            st.session_state.rotation_direction = 1

    # Update confidence based on rotation
    angle_from_zero = abs(st.session_state.rotation)
    st.session_state.confidence = round(
        98 - (15 * (angle_from_zero / 45)), 2
    )  # Decreases from 98% to 65%

    # Randomly vary SpO2 value
    st.session_state.spo2 = random.randint(80, 88)

    # Display updated metrics
    rotation_placeholder.metric(
        "Rotation", f"{st.session_state.rotation} °C", delta=None
    )
    confidence_placeholder.metric(
        "Confidence", f"{st.session_state.confidence}%", delta=None
    )
    spo2_placeholder.metric("Heart Rate", f"{st.session_state.spo2} BPM", delta=None)

    # Sleep for the duration of the GIF's rotation cycle or any desired update interval
    time.sleep(0.157)
