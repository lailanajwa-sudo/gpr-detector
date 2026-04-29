import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2

# Page Setup
st.set_page_config(page_title="GPR Genius AI", layout="wide")

st.title("🛰️ High-Accuracy GPR Detector")
st.write("This AI uses auto-processing to find Cavities, Pipes, and Bricks with high precision.")

# 1. Load Model
@st.cache_resource
def load_model():
    # After retraining in Colab, upload your new 'best.pt' to GitHub
    return YOLO('best.pt')

model = load_model()

# 2. Upload
uploaded_file = st.file_uploader("Upload Radargram", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # --- AUTO-GENIUS PRE-PROCESSING ---
    # 1. Open and fix orientation
    raw_img = Image.open(uploaded_file)
    
    # 2. Convert to Grayscale & back to RGB 
    # (Matches training data patterns and removes color noise)
    proc_img = ImageOps.grayscale(raw_img).convert('RGB')
    img_array = np.array(proc_img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Radar Data")
        st.image(raw_img, use_container_width=True)

    # 3. Genius Prediction
    with st.spinner('AI analyzing subsurface...'):
        # We set 'augment=True' here. This is the "Genius" part.
        # The AI will look at the image 3 different ways and combine the results.
        results = model.predict(
            source=img_array, 
            conf=0.20,      # Capture even faint hyperbolas
            iou=0.45,       # Clean up overlapping boxes
            imgsz=640, 
            augment=True    # CRITICAL: Increases accuracy on new images
        )
        
        # Plotting
        res_plotted = results[0].plot(line_width=2, font_size=1)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("AI Identification")
        st.image(res_rgb, use_container_width=True)

    # 4. Result Dashboard
    st.markdown("---")
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.success(f"✅ AI Analysis Complete: {len(boxes)} targets identified.")
        # Group results for the user
        for box in boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"📍 **{name.upper()}** - Confidence: `{conf:.2%}`")
    else:
        st.warning("No targets identified. The area appears clear or the signal is too weak.")
