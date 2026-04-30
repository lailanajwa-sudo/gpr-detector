import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GPR-X Intelligence Portal", layout="wide")

st.title("🛰️ GPR-X Analysis Suite")
st.write("Automated Subsurface Anomaly Detection & Classification")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best.pt' not found. Please upload it to your GitHub repository.")

# --- 3. UPLOAD WORKFLOW ---
st.divider()
uploaded_file = st.file_uploader("Upload Radargram B-Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    with st.spinner("AI analyzing radargram..."):
        # Run AI Inference
        results = model.predict(source=img_array, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Count number of boxes detected
        num_detections = len(results[0].boxes)

    # --- 4. DISPLAY RESULTS ---
    st.markdown("### 📊 Classification Results")
    
    # Simple Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Data**")
        st.image(img, use_container_width=True)
    with col2:
        st.write("**AI Predictions**")
        st.image(res_rgb, use_container_width=True)

    # --- 5. NUMERICAL RESULTS (Replaces White Box) ---
    st.divider()
    
    # We use a clean columns layout instead of custom CSS cards to avoid the "white box" issue
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Summary")
        st.write(f"✅ **Total Anomalies Identified:** {num_detections}")
        st.write(f"The model successfully classified **{num_detections}** targets in this scan.")
    
    with res_col2:
        st.info("💡 **Note:** These detections represent hyperbolic signatures typically associated with buried utilities or cavities.")

else:
    st.info("Please upload a radargram to begin classification.")
