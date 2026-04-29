import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_canvas import st_canvas
import pandas as pd
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GPR-X GENIUS", layout="wide")

st.title("🛰️ GPR-X: Intelligent Radargram Analysis")
st.markdown("""
### Is the AI wrong? 
AI can sometimes miss faint hyperbolic curves. Use the **Correction Canvas** below to draw a box around any missed objects. 
""")
st.markdown("---")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to your GitHub repository.")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_level = st.slider("Sensitivity", 0.01, 1.0, 0.15)
    st.info("Lower sensitivity if the AI is missing clear curves.")

# --- 4. UPLOAD & PREDICT ---
uploaded_file = st.file_uploader("Upload Radargram...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file)
    proc_img = ImageOps.grayscale(raw_img).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        st.image(raw_img, use_container_width=True)
        
    with col2:
        st.subheader("AI Prediction")
        with st.spinner("Analyzing..."):
            results = model.predict(source=np.array(proc_img), conf=conf_level, imgsz=640, augment=True)
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- 5. THE CORRECTION CANVAS ---
    st.markdown("---")
    st.subheader("🖍️ Correction Canvas")
    
    canvas_width = 700
    ratio = canvas_width / raw_img.width
    canvas_height = int(raw_img.height * ratio)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=raw_img,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas",
    )

    # --- 6. SAVE DRAWN BOXES ---
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        if not objects.empty:
            st.write("📍 **Newly Marked Objects:**")
            boxes = objects[objects['type'] == 'rect'][['left', 'top', 'width', 'height']]
            st.dataframe(boxes)
            
            label = st.selectbox("Correct Label?", ["Cavity", "Metal Pipe", "Brick"])
            if st.button("Submit Feedback"):
                st.success(f"Feedback Logged as {label}!")
                st.balloons()
