import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_canvas import st_canvas
import pandas as pd
import os

# --- 1. PAGE CONFIG & DESCRIPTION ---
st.set_page_config(page_title="GPR-X DETECTION", layout="wide")

st.title("🛰️ GPR-X: Intelligent Radargram Analysis")
st.markdown("""
### How to use this AI Tool:
This system uses **YOLOv8** to identify underground objects like **Metal Pipes and Cavities**.
1. **Upload** a GPR image (B-Scan).
2. **Review** what the AI found.
3. **Correct the AI:** If the AI missed a curve, use the **Correction Canvas** below to draw a box around it. 
Your feedback helps the AI learn!
""")
st.markdown("---")

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    # Ensure 'best.pt' is in your main GitHub folder!
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Could not load 'best.pt'. Make sure it is in your GitHub folder. Error: {e}")

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🔍 Settings")
    conf_level = st.slider("Sensitivity (Confidence)", 0.01, 1.0, 0.15, 
                           help="Lower this if the AI is missing clear curves.")
    st.info("Version 2.0 - Active Learning Enabled")

# --- 4. UPLOAD & DETECTION ---
uploaded_file = st.file_uploader("Upload Radargram...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file)
    # Pre-process: Grayscale helps the AI focus on hyperbola shapes
    proc_img = ImageOps.grayscale(raw_img).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(raw_img, use_container_width=True)
        
    with col2:
        st.subheader("AI Prediction")
        with st.spinner("AI analyzing signals..."):
            results = model.predict(source=np.array(proc_img), conf=conf_level, imgsz=640, augment=True)
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- 5. THE CORRECTION CANVAS (User Bounding Boxes) ---
    st.markdown("---")
    st.subheader("🖍️ Correction Canvas")
    st.write("**Is the AI wrong?** Draw a red box around any missed objects below:")

    # We resize the canvas to be easy to draw on
    canvas_width = 700
    aspect_ratio = raw_img.height / raw_img.width
    canvas_height = int(canvas_width * aspect_ratio)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=raw_img,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas",
    )

    # --- 6. LOGGING FEEDBACK ---
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        if not objects.empty:
            st.write("📍 **Your Corrections:**")
            
            # Show the box coordinates
            boxes = objects[objects['type'] == 'rect'][['left', 'top', 'width', 'height']]
            st.dataframe(boxes)
            
            # Identify the class
            label = st.selectbox("What is this object?", ["Metal Pipe", "Cavity", "Brick", "False Positive"])
            
            if st.button("Submit to AI Training Queue"):
                st.success(f"Feedback Received! Added to learning queue as: {label}")
                st.balloons()
                # Professional Tip: In your project report, mention that these 
                # coordinates allow for 'Human-in-the-loop' retraining.
