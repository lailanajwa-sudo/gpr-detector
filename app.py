import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="GPR-X Analysis Suite", layout="wide", page_icon="🛰️")

# Custom UI Styling
st.markdown("""
    <style>
    .stAlert { border-radius: 10px; }
    .stButton>button { border-radius: 20px; border: 1px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("🛰️ GPR-X Intelligence Portal")
st.write("Automated Subsurface Anomaly Detection & Classification")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("Model weights 'best.pt' not found. Please upload it to your GitHub.")

# --- 4. UPLOAD WORKFLOW ---
uploaded_file = st.file_uploader("📂 Upload Radargram (B-Scan)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # Inference
    with st.spinner("AI analyzing radargram signatures..."):
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        detections = len(results[0].boxes)

    # --- SIDE-BY-SIDE CLASSIFICATION (MAIN GOAL) ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Data")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("AI Identification")
        st.image(res_rgb, use_container_width=True)
        st.metric("Total Anomalies Detected", detections)

    # --- 5. OPTIONAL CORRECTION MODE ---
    st.divider()
    st.subheader("🛠️ Quality Control")
    
    # Ask first
    wants_to_correct = st.checkbox("Enable Expert Feedback Mode (If AI missed an object)")

    if wants_to_correct:
        st.info("🎯 **Apex Precision Tool:** Use the sliders to align the crosshairs with the peak (apex) of the hyperbola.")
        
        tool_col1, tool_col2 = st.columns([1, 1])
        
        with tool_col1:
            # Precise sliders for X and Y
            target_x = st.slider("Align X-Axis", 0, img.width, img.width // 2)
            target_y = st.slider("Align Y-Axis", 0, img.height, img.height // 2)
            
            obj_label = st.selectbox("Assign Correct Category", ["Metal Pipe", "Cavity", "Stone/Brick"])
            if st.button("Submit Correction to Training Queue"):
                st.success(f"Apex logged at X:{target_x}, Y:{target_y} as {obj_label}.")
                st.balloons()
        
        with tool_col2:
            # MAGNIFIER: Shows the user exactly where they are pointing
            # Crop a small window around the target
            window = 60
            left = max(0, target_x - window)
            top = max(0, target_y - window)
            right = min(img.width, target_x + window)
            bottom = min(img.height, target_y + window)
            
            zoom_img = img.crop((left, top, right, bottom))
            # Draw a crosshair on the zoomed image
            zoom_np = np.array(zoom_img)
            h, w, _ = zoom_np.shape
            cv2.line(zoom_np, (w//2, 0), (w//2, h), (255, 0, 0), 1)
            cv2.line(zoom_np, (0, h//2), (w, h//2), (255, 0, 0), 1)
            
            st.image(zoom_np, width=300, caption="Magnified Precision View")

else:
    st.divider()
    st.info("Welcome. Please upload a GPR image to begin classification.")
