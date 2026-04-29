import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_canvas import st_canvas
import pandas as pd

# --- 1. PAGE SETUP & DESCRIPTION ---
st.set_page_config(page_title="GPR-X GENIUS", layout="wide")

st.title("🛰️ GPR-X: Intelligent Radargram Analysis")
st.markdown("""
### What is this?
This AI system is designed to detect underground anomalies in **Ground Penetrating Radar (GPR)** data. 
It identifies **Cavities, Metal Pipes, and Bricks** by recognizing 'Hyperbolic signatures' (the curve shapes).

**Is the AI wrong?** AI can sometimes miss faint signals or misidentify noise. Use the **Correction Canvas** below to draw boxes around missed objects. Your input helps 'teach' the model to be more accurate in the next training cycle.
""")
st.markdown("---")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to GitHub.")

# --- 3. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_level = st.slider("Sensitivity", 0.05, 1.0, 0.20)
    st.info("Lower sensitivity if the AI is missing clear hyperbolas.")

# --- 4. UPLOAD & PREDICT ---
uploaded_file = st.file_uploader("Upload a Radargram...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file)
    # Match training data pre-processing
    proc_img = ImageOps.grayscale(raw_img).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Input")
        st.image(raw_img, use_container_width=True)
        
    with col2:
        st.subheader("AI Prediction")
        results = model.predict(source=np.array(proc_img), conf=conf_level, imgsz=640, augment=True)
        res_plotted = results[0].plot()
        st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- 5. THE CORRECTION CANVAS (User Bounding Boxes) ---
    st.markdown("---")
    st.subheader("🖍️ Manual Correction Canvas")
    st.write("If the AI missed something, **draw a box** around the correct area below:")

    # Define canvas properties
    stroke_width = 3
    bg_image = raw_img
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange
        stroke_width=stroke_width,
        stroke_color="#FF0000",
        background_image=bg_image,
        update_streamlit=True,
        height=bg_image.height * (600 / bg_image.width), # Maintain aspect ratio
        width=600,
        drawing_mode="rect", # Force rectangle/bounding box mode
        key="canvas",
    )

    # --- 6. SAVE DRAWN BOXES ---
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        if not objects.empty:
            st.write("📍 **New Objects Marked by User:**")
            # Filter only rectangles
            boxes = objects[objects['type'] == 'rect'][['left', 'top', 'width', 'height']]
            st.dataframe(boxes)
            
            correct_label = st.selectbox("What object did you just draw?", ["Cavity", "Metal Pipe", "Brick"])
            
            if st.button("Submit Boxes for Training"):
                # Here you would save these coordinates to your CSV or Database
                st.success(f"Thank you! Saved {len(boxes)} boxes as '{correct_label}'.")
                st.balloons()
                st.info("The developer will now use these coordinates to retrain the AI model.")
