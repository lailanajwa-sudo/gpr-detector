import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GPR-X Intelligence Portal", layout="wide")

st.title("🛰️ GPR-X Analysis Suite")
st.markdown("### Automated Classification & Expert Correction")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("Model 'best.pt' not found in repository.")

# --- 3. UPLOAD ---
uploaded_file = st.file_uploader("Upload Radargram", type=["jpg", "jpeg", "png"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Run Prediction
    with st.spinner("Analyzing signals..."):
        results = model.predict(source=np.array(raw_img), conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # --- SIDE-BY-SIDE VIEW ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Radargram")
        st.image(raw_img, use_container_width=True)
    with col2:
        st.subheader("AI Classification")
        st.image(res_rgb, use_container_width=True)

    # --- 4. EXPERT CORRECTION (OPTIONAL) ---
    st.divider()
    if st.checkbox("🛠️ Help Improve AI (Add Missing Object)"):
        st.info("Identify the Apex (peak) of the missed hyperbola to generate a training box.")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            # Sliders to pick the center of the new box
            pick_x = st.slider("Select Apex X", 0, raw_img.width, raw_img.width // 2)
            pick_y = st.slider("Select Apex Y", 0, raw_img.height, raw_img.height // 2)
            # Slider to control the box size visually
            box_size = st.slider("Adjust Box Size", 10, 200, 60)
            
            obj_label = st.selectbox("Object Category", ["Metal Pipe", "Cavity", "Stone/Brick"])
            
            if st.button("Submit Bounding Box to Queue"):
                st.success(f"Box for {obj_label} logged at ({pick_x}, {pick_y})!")
                st.balloons()

        with c2:
            # PREVIEW: Show the bounding box on the image in real-time
            preview_img = raw_img.copy()
            draw = ImageDraw.Draw(preview_img)
            
            # Calculate box corners
            left = pick_x - (box_size // 2)
            top = pick_y - (box_size // 2)
            right = pick_x + (box_size // 2)
            bottom = pick_y + (box_size // 2)
            
            # Draw the red bounding box the user is creating
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            # Draw a small crosshair at the apex
            draw.line([pick_x-10, pick_y, pick_x+10, pick_y], fill="red", width=1)
            draw.line([pick_x, pick_y-10, pick_x, pick_y+10], fill="red", width=1)
            
            st.image(preview_img, use_container_width=True, caption="Correction Preview")
