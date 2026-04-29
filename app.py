import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2

# 1. Page Config & Professional UI
st.set_page_config(page_title="GPR-X DETECTOR", layout="wide")

st.title("🛰️ GPR-X DETECTOR")
st.markdown("""
**Model Details:** Trained to detect **Cavities, Metal Pipes, and Bricks**.  
*If confidence is low, try adjusting the sensitivity slider in the sidebar.*
""")
st.markdown("---")

# 2. Sidebar Configuration
with st.sidebar:
    st.header("🔍 Detection Settings")
    # Slider to help with "Low Confidence" issues
    conf_level = st.slider("Sensitivity (Confidence)", 0.05, 1.0, 0.20, help="Lower this if objects aren't being detected.")
    st.write("---")
    st.info("Upload a GPR Radargram (B-Scan) to begin analysis.")

# 3. Load YOLOv8 Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'best.pt': {e}. Ensure the file is in your GitHub repository.")

# 4. Image Upload & Processing
uploaded_file = st.file_uploader("Choose a GPR image file...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # --- STEP A: Pre-processing to match training data ---
    # GPR models often perform better if we force them to grayscale first
    raw_image = Image.open(uploaded_file)
    
    # 1. Convert to Grayscale then back to RGB (Removes 'color noise')
    proc_image = ImageOps.grayscale(raw_image).convert('RGB')
    
    # Create columns for side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Radargram")
        st.image(raw_image, use_container_width=True)

    # --- STEP B: Run AI Prediction ---
    with st.spinner('AI is searching for hyperbolas...'):
        img_array = np.array(proc_image)
        
        # 'augment=True' helps with low confidence by looking at the image 
        # from different scales and orientations automatically.
        results = model.predict(
            source=img_array, 
            conf=conf_level, 
            imgsz=640, 
            augment=True
        )
        
        # Draw results
        res_plotted = results[0].plot() 
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("AI Detection Result")
        st.image(res_rgb, use_container_width=True)

    # --- STEP C: Data Summary ---
    st.markdown("### 📊 Detection Summary")
    boxes = results[0].boxes
    if len(boxes) > 0:
        # Create a clean list of findings
        counts = {}
        for box in boxes:
            label = model.names[int(box.cls[0])]
            counts[label] = counts.get(label, 0) + 1
        
        # Display as metric cards
        m_col1, m_col2, m_col3 = st.columns(3)
        cols = [m_col1, m_col2, m_col3]
        for i, (name, count) in enumerate(counts.items()):
            cols[i % 3].metric(label=name.upper(), value=count)
    else:
        st.warning("No targets found. Try lowering the 'Sensitivity' slider in the sidebar.")
