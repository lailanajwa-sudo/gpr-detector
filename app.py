import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="GPR-X PRECISION", layout="wide")
st.title("🛰️ GPR-X PRECISION DETECTION")
st.markdown("---")

# 2. Sidebar Settings
with st.sidebar:
    st.header("🔍 Detection Settings")
    conf_level = st.slider("Sensitivity (Confidence)", 0.01, 1.0, 0.15, 
                           help="Lower this to 0.10 if the AI is missing clear curves.")
    st.write("---")
    st.info("Upload a GPR Radargram to begin analysis.")

# 3. Load YOLOv8 Model
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in your GitHub folder
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 4. Image Upload
uploaded_file = st.file_uploader("Choose a GPR image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- STEP A: Pre-processing ---
    raw_image = Image.open(uploaded_file)
    proc_image = ImageOps.grayscale(raw_image).convert('RGB')
    img_width, img_height = raw_image.size
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Radargram")
        st.image(raw_image, use_container_width=True)

    # --- STEP B: AI Prediction ---
    with st.spinner('AI Searching...'):
        img_array = np.array(proc_image)
        results = model.predict(source=img_array, conf=conf_level, imgsz=640, augment=True)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("AI Detection Result")
        st.image(res_rgb, use_container_width=True)

    # --- STEP C: The Correction System (The "Genius" Learning Part) ---
    st.markdown("---")
    st.subheader("🎯 Visual Correction Tool")
    st.write("If the AI missed a curve, use the sliders to place a marker on the **Apex (Top)** of the curve.")

    # Interaction columns
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        # User defines X/Y as percentage of image size
        x_pct = st.slider("Move Marker Horizontal (X)", 0, 100, 50)
    with c2:
        y_pct = st.slider("Move Marker Vertical (Y)", 0, 100, 50)
    with c3:
        correct_obj = st.selectbox("What is at this location?", 
                                   ["Select...", "Metal Pipe", "Cavity", "Brick", "False Alarm"])

    # Calculate real pixel coordinates
    real_x = int((x_pct / 100) * img_width)
    real_y = int((y_pct / 100) * img_height)

    # Create a preview of the correction
    preview_img = np.array(raw_image.copy())
    cv2.drawMarker(preview_img, (real_x, real_y), (255, 0, 0), cv2.MARKER_CROSS, 40, 5)
    
    st.image(preview_img, caption="Red Cross = Your Correction Point", width=500)

    if st.button("📥 Save Correction for Training"):
        if correct_obj != "Select...":
            # Save data to a local CSV log
            new_data = {
                "timestamp": [datetime.now()],
                "filename": [uploaded_file.name],
                "x_pixel": [real_x],
                "y_pixel": [real_y],
                "label": [correct_obj]
            }
            df = pd.DataFrame(new_data)
            
            # Append to a file
            file_path = "correction_logs.csv"
            if not os.path.isfile(file_path):
                df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, mode='a', header=False, index=False)
            
            st.success(f"Successfully logged! Marked {correct_obj} at ({real_x}, {real_y}).")
            st.info("Download 'correction_logs.csv' later to update your Colab dataset.")
        else:
            st.warning("Please select the correct object type first.")
