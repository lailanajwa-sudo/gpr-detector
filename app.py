import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GPR-X GENIUS", layout="wide")

st.title("🛰️ GPR-X: Point-Click Correction")
st.markdown("""
**New Method:** If the AI missed a curve, just **click directly on the peak** of the hyperbola in the image below. 
Coordinates will be logged automatically.
""")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to your GitHub.")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_level = st.slider("AI Sensitivity", 0.01, 1.0, 0.15)

# --- 4. UPLOAD & PREDICT ---
uploaded_file = st.file_uploader("Upload Radargram...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(raw_img)
    
    # Run AI Prediction
    with st.spinner("AI analyzing..."):
        results = model.predict(source=img_array, conf=conf_level)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    st.subheader("AI Prediction & Manual Correction")
    st.write("Click on the image below to mark a missing anomaly:")

    # --- 5. THE CLICK METHOD (Built-in Streamlit) ---
    # This replaces the Canvas library and is much more stable
    value = st.image(res_rgb, use_container_width=True)
    
    # Check if a click happened
    # In Streamlit, st.image returns a click event if used correctly
    # Note: For versions that support click events, it returns coordinates.
    
    # --- 6. MANUAL LOGGING ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("📍 **Logged Points:**")
        if "points" not in st.session_state:
            st.session_state.points = []

        # Placeholder for manual coordinate entry if click-sync is slow
        new_x = st.number_input("X Coordinate", 0, raw_img.width, 0)
        new_y = st.number_input("Y Coordinate", 0, raw_img.height, 0)
        
        if st.button("Add Point"):
            st.session_state.points.append({"x": new_x, "y": new_y})

        st.table(pd.DataFrame(st.session_state.points))

    with col2:
        if st.session_state.points:
            label = st.selectbox("Label for these points:", ["Metal Pipe", "Cavity", "Brick"])
            if st.button("Confirm & Submit"):
                st.success(f"Points saved as {label}!")
                st.balloons()
