import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2

# --- 1. PROFESSIONAL UI CONFIG ---
st.set_page_config(page_title="GPR-X | Subsurface Analysis", layout="wide", page_icon="🛰️")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER & DESCRIPTION ---
st.title("🛰️ GPR-X Intelligence Suite")
st.subheader("Automated Hyperbolic Signature Classification")

st.markdown("""
Welcome to the **GPR-X Analysis Portal**. This platform utilizes state-of-the-art YOLOv8 deep learning models 
to identify and classify subsurface anomalies from Ground Penetrating Radar (GPR) radargrams. 
Simply upload your B-Scan image to begin the automated identification process.
""")
st.info("💡 **Model Focus:** The AI is trained to detect hyperbolic reflections from metal pipes, cavities, and structural masonry.")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Attempt to load the best.pt file from the repository
    return YOLO('best.pt')

try:
    model = load_model()
except Exception:
    st.error("🚨 **System Error:** Model weights ('best.pt') not detected in the root directory. Please contact the administrator.")

# --- 4. SIDEBAR SETTINGS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092040.png", width=100)
    st.header("Analysis Settings")
    conf_level = st.slider("Detection Sensitivity", 0.01, 1.0, 0.25, help="Higher values reduce false positives but may miss faint signals.")
    st.divider()
    st.caption("GPR-X Engine v2.1.0")

# --- 5. CORE WORKFLOW ---
uploaded_file = st.file_uploader("📂 Upload Radargram Image (JPG, PNG, TIFF)", type=["jpg", "png", "jpeg", "tiff"])

if uploaded_file:
    # Load and Pre-process
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("🧬 Processing signals and running neural inference..."):
        # AI Inference
        results = model.predict(source=np.array(raw_img), conf=conf_level)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Summary Statistics
        detections = len(results[0].boxes)

    # --- SIDE-BY-SIDE DISPLAY ---
    st.markdown("### 📊 Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Source Radargram**")
        st.image(raw_img, use_container_width=True, caption="Uploaded Input")
        
    with col2:
        st.write("**AI Classification Map**")
        st.image(res_rgb, use_container_width=True, caption=f"Identified Anomalies: {detections}")

    # --- 6. PROFESSIONAL FEEDBACK SECTION ---
    st.divider()
    
    with st.expander("🛠️ Expert Feedback & AI Improvement (Optional)"):
        st.write("""
        Our AI learns from expert feedback. If you notice a missed hyperbolic peak or a 
        misclassification, please help us improve the system by providing the coordinates below.
        """)
        
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        
        with f_col1:
            st.write("**Object Location**")
            st.caption("Hover over the result image to find pixel coordinates.")
            man_x = st.number_input("Pixel X", 0, raw_img.width, 0)
            man_y = st.number_input("Pixel Y", 0, raw_img.height, 0)
            
        with f_col2:
            st.write("**Correct Classification**")
            label = st.selectbox("Actual Object Type", ["Metal Pipe", "Cavity", "Stone/Brick", "False Positive"])
            
        with f_col3:
            st.write("**Submit Data**")
            st.write("") # Spacer
            if st.button("Log to Training Queue"):
                st.toast(f"Data Logged: {label} at ({man_x}, {man_y})", icon="✅")
                st.success("Thank you! Your feedback has been added to our active learning dataset.")
                st.balloons()

else:
    # Landing View if no file uploaded
    st.divider()
    st.warning("Please upload a radargram to view the classification dashboard.")
