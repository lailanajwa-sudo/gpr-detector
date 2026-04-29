import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Page Configuration (The "Design" part)
st.set_page_config(page_title="GPR-X DETECTOR", layout="wide")

st.title("🛰️ GPR Radargram Object Detection")
st.markdown("""
### Deep Learning for Subsurface Analysis
Welcome to the **GPR AI Scanner**. This tool uses a trained YOLOv8 model to automatically 
identify underground features from radargram images.")
st.markdown("---")

# 2. Sidebar Settings
with st.sidebar:
    st.header("Settings")
    st.write("Adjust how strict the AI should be:")
    # A slider so users can control the detection sensitivity
    conf_level = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    st.info("Tip: If you see too many 'fake' boxes, increase this value.")

# 3. Load the Model
@st.cache_resource # This makes the website fast!
def load_model():
    # Make sure 'best.pt' is uploaded to the same GitHub folder
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}. Check if 'best.pt' is in your GitHub repo.")

# 4. File Uploader
uploaded_file = st.file_uploader("Upload a GPR Radargram image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # --- FIX FOR YOUR ERROR: Convert to RGB and Resize ---
    # This prevents the 'conv2d' error with random Google images
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns for a clean look
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # 5. Run Detection
    with st.spinner('Analyzing Radargram...'):
        # Convert to numpy for YOLO
        img_array = np.array(image)
        
        # We use imgsz=640 to match your training in Colab
        results = model.predict(source=img_array, conf=conf_level, imgsz=640)
        
        # Plot the boxes on the image
        res_plotted = results[0].plot() 
        # Convert BGR (OpenCV style) to RGB (Streamlit style)
        res_rgb = res_plotted[:, :, ::-1]

    with col2:
        st.subheader("Detection Result")
        st.image(res_rgb, use_container_width=True)

    # 6. Summary of Results
    st.markdown("### Detection Details")
    boxes = results[0].boxes
    if len(boxes) > 0:
        # Create a list to show exactly what was found
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            prob = float(box.conf[0])
            st.success(f"✔️ Found: **{label.upper()}** (Confidence: {prob:.2f})")
    else:
        st.warning("No objects (cavity, pipe, or brick) detected with current settings.")
