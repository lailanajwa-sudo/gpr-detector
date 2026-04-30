import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="GPR-X Analysis Suite", layout="wide", page_icon="🛰️")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("🛰️ GPR-X Intelligence Portal")
st.write("Automated Subsurface Anomaly Detection & Classification System")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure 'best.pt' is in your main GitHub repository folder
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("🚨 System Error: Model weights ('best.pt') not detected. Please upload the file to your GitHub.")

# --- 4. UPLOAD WORKFLOW ---
st.divider()
uploaded_file = st.file_uploader("📂 Upload Radargram B-Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Process Image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    with st.spinner("AI analyzing subsurface signatures..."):
        # Run YOLOv8 Inference
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Count detections
        num_detections = len(results[0].boxes)

    # --- 5. SIDE-BY-SIDE RESULTS ---
    st.markdown("### 📊 Classification Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🖼️ **Original Input**")
        st.image(img, use_container_width=True)
        
    with col2:
        st.success(f"🤖 **AI Classification Result**")
        st.image(res_rgb, use_container_width=True)
        
    # Summary Metric
    st.divider()
    st.metric(label="Total Anomalies Identified", value=num_detections)
    st.caption("Note: Detections are based on hyperbolic signature patterns in the GPR data.")

else:
    # Default landing state
    st.warning("Please upload a radargram file to begin the automated classification process.")
    st.info("This system is optimized for identifying metal pipes, cavities, and masonry structures.")
