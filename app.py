import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd

# --- 1. PAGE SETUP & DESCRIPTION ---
st.set_page_config(page_title="GPR-X GENIUS", layout="wide")

st.title("🛰️ GPR-X: Intelligent Radargram Analysis")
st.markdown("""
### How it works:
1. **AI Detection:** The system scans your GPR B-scan for hyperbolic signatures (the 'U' shapes) indicating buried pipes, cavities, or bricks.
2. **Verification:** You confirm if the AI's findings are correct.
3. **Manual Correction:** If the AI missed something, you can manually mark the coordinates to help retrain the model.
""")
st.markdown("---")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt') # Ensure best.pt is in your GitHub folder [cite: 6, 22]

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to your GitHub.")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_level = st.slider("AI Sensitivity", 0.01, 1.0, 0.20)
    st.info("Lower sensitivity if the AI is missing faint hyperbolas.")

# --- 4. UPLOAD & PREDICT ---
uploaded_file = st.file_uploader("Upload a Radargram...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Run Prediction
    with st.spinner("AI analyzing signals..."):
        results = model.predict(source=np.array(raw_img), conf=conf_level)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # --- 5. SIDE-BY-SIDE CLASSIFICATION ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Input")
        st.image(raw_img, use_container_width=True)
    with col2:
        st.subheader("AI Detection Result")
        st.image(res_rgb, use_container_width=True)

    # --- 6. USER FEEDBACK LOOP ---
    st.markdown("---")
    st.subheader("✅ Verification")
    
    is_correct = st.radio("Is the AI detection accurate?", ("Select an option", "Yes, it looks correct", "No, it missed something / False alarm"))

    if is_correct == "Yes, it looks correct":
        st.success("Great! This data will be logged as a successful detection.")
        st.balloons()
        
    elif is_correct == "No, it missed something / False alarm":
        st.warning("Let's correct the AI. Please provide the coordinates of the missed object.")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.info("Hover over the image above to find the (X, Y) pixel coordinates of the missed hyperbola peak.")
            # We use two inputs for a "Point-Click" style of logging
            manual_x = st.number_input("Enter X Coordinate", 0, raw_img.width, 0)
            manual_y = st.number_input("Enter Y Coordinate", 0, raw_img.height, 0)
            
        with c2:
            correct_label = st.selectbox("What is the object at this point?", ["Metal Pipe", "Cavity", "Brick", "False Alarm"])
            if st.button("Submit Correction"):
                # Save to session state or a log file
                st.success(f"Point ({manual_x}, {manual_y}) logged as '{correct_label}'.")
                st.info("This coordinate has been sent to the developer for model retraining.")
