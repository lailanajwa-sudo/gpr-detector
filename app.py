import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="GPR-X Analysis Suite", layout="wide", page_icon="🛰️")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("🛰️ GPR-X Intelligence Portal")
st.write("Professional Ground Penetrating Radar Classification & Active Learning System")

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("Model file 'best.pt' not found.")

# --- 4. UPLOAD ---
uploaded_file = st.file_uploader("Upload B-Scan Radargram", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # --- AI INFERENCE ---
    with st.spinner("Analyzing Subsurface Signatures..."):
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # --- SIDE BY SIDE RESULTS ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Data")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("AI Classification")
        st.image(res_rgb, use_container_width=True)

    # --- 5. INTERACTIVE APEX CORRECTION ---
    st.divider()
    st.subheader("🎯 Precision Apex Correction")
    st.write("If the AI missed a hyperbola, define the **Apex (Peak)** below to log a new training box.")

    # User-friendly coordinate selection
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        st.write("**Fine-Tune X**")
        target_x = st.slider("X Axis (Width)", 0, img.width, img.width // 2)
    with c2:
        st.write("**Fine-Tune Y**")
        target_y = st.slider("Y Axis (Height)", 0, img.height, img.height // 2)
        
    with c3:
        # MAGNIFIED VIEW: Shows the user exactly where they are pointing
        # Crop a 100x100 area around the selected point
        left = max(0, target_x - 50)
        top = max(0, target_y - 50)
        right = min(img.width, target_x + 50)
        bottom = min(img.height, target_y + 50)
        
        zoom_img = img.crop((left, top, right, bottom))
        st.image(zoom_img, width=200, caption="Magnified Apex View")

    # --- 6. DATA LOGGING ---
    st.write("**Correction Details**")
    log_col1, log_col2, log_col3 = st.columns(3)
    
    with log_col1:
        obj_type = st.selectbox("Identify Object", ["Metal Pipe", "Cavity", "Stone/Brick"])
    with log_col2:
        box_size = st.slider("Target Box Size", 20, 100, 50, help="Adjust the size of the training box around the apex.")
    with log_col3:
        st.write("") # Spacer
        if st.button("Log Apex for Retraining"):
            # This generates the bounding box coordinates automatically
            new_box = {
                "x_center": target_x,
                "y_center": target_y,
                "width": box_size,
                "height": box_size,
                "label": obj_type
            }
            st.session_state.last_log = new_box
            st.success(f"Successfully logged {obj_type} at {target_x}, {target_y}!")
            st.balloons()

    if "last_log" in st.session_state:
        st.info(f"Retraining Queue: {st.session_state.last_log}")

else:
    st.info("Please upload a radargram to initialize the GPR-X engine.")
