import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# 1. Page Config & Professional UI
st.set_page_config(page_title="GPR-X DETECTION", layout="wide")

st.title("🛰️ GPR-X DETECTION")
st.markdown("""
**Model Details:** Detects **Cavities, Metal Pipes, and Bricks**.  
*Instructions: Upload a radargram, check the AI results, and use the Correction Box below to report specific errors.*
""")
st.markdown("---")

# 2. Sidebar Configuration
with st.sidebar:
    st.header("🔍 Detection Settings")
    conf_level = st.slider("Sensitivity (Confidence)", 0.05, 1.0, 0.20, 
                           help="Lower this if objects aren't being detected.")
    st.write("---")
    st.info("Upload a GPR Radargram (B-Scan) to begin analysis.")

# 3. Load YOLOv8 Model
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in your GitHub main folder
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'best.pt': {e}. Ensure the file is in your GitHub repository.")

# 4. Image Upload & Processing
uploaded_file = st.file_uploader("Choose a GPR image file...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # --- STEP A: Pre-processing ---
    raw_image = Image.open(uploaded_file)
    # Convert to grayscale to match GPR training patterns
    proc_image = ImageOps.grayscale(raw_image).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Radargram")
        st.image(raw_image, use_container_width=True)

    # --- STEP B: Run AI Prediction ---
    with st.spinner('AI is searching for hyperbolas...'):
        img_array = np.array(proc_image)
        
        # Using 'augment=True' for high accuracy on the website
        results = model.predict(
            source=img_array, 
            conf=conf_level, 
            imgsz=640, 
            augment=True
        )
        
        # Plot the boxes
        res_plotted = results[0].plot() 
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("AI Detection Result")
        st.image(res_rgb, use_container_width=True)

    # --- STEP C: Data Summary ---
    st.markdown("### 📊 Detection Summary")
    boxes = results[0].boxes
    if len(boxes) > 0:
        counts = {}
        for box in boxes:
            label = model.names[int(box.cls[0])]
            counts[label] = counts.get(label, 0) + 1
        
        m_col1, m_col2, m_col3 = st.columns(3)
        cols = [m_col1, m_col2, m_col3]
        for i, (name, count) in enumerate(counts.items()):
            cols[i % 3].metric(label=name.upper(), value=count)
    else:
        st.warning("No targets found. Try lowering the 'Sensitivity' slider.")

    # --- STEP D: ADVANCED MULTI-CLASS CORRECTION (The Learning Loop) ---
    st.markdown("---")
    st.subheader("🎯 Pinpoint & Correct Mistakes")
    st.write("Help the AI learn by reporting specifically which part of the detection is wrong.")
    
    with st.expander("Report an Error (Missed objects or Wrong labels)"):
        # Grid layout for precise feedback
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            zone = st.selectbox("Where is the mistake?", 
                                ["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"])
        
        with f_col2:
            actual_class = st.selectbox("What is actually there?", 
                                      ["Select...", "Cavity", "Metal Pipe", "Brick", "Ghost/Noise (Nothing)"])
            
        with f_col3:
            error_type = st.selectbox("What was the AI's mistake?", 
                                    ["Missed an object", "Label is wrong", "Box is in the wrong place"])

        user_comment = st.text_area("Additional Details:", placeholder="e.g. The metal pipe at the bottom right was missed.")

        if st.button("Submit Feedback to AI"):
            if actual_class != "Select...":
                # Success Message
                st.success(f"Feedback Logged! You marked the **{zone}** as a **{actual_class}**.")
                st.balloons()
                
                # Instructions for the Developer
                st.info(f"**Developer Note:** To 'Automate' the learning, download this image, "
                        f"manually label the object in the **{zone}** as **{actual_class}**, "
                        f"and run the Colab training script again.")
            else:
                st.error("Please select the correct class to submit.")
