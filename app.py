import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Configuration
st.set_page_config(page_title="Fruit Quality Control", page_icon="🍎")

st.title("🍎 Factory AI Inspector")
st.write("Visual Quality Control System (PoC)")

# 2. Load the Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_keras_model():
    # This forces Python to look in the exact current directory
    file_path = os.path.abspath("keras_model.h5")
    model = load_model(file_path, compile=False)
    return model

# Load class names
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

try:
    model = load_keras_model()
except:
    st.error("Could not find 'keras_model.h5'. Please upload it to your project folder.")
    st.stop()

# 3. The Camera Interface
st.write("### 📸 Scan Product")
# This opens the camera (Works on Phone & Laptop)
img_file_buffer = st.camera_input("Take a picture of the apple")

if img_file_buffer is not None:
    # 4. Pre-process the image to fit Teachable Machine's requirements
    image = Image.open(img_file_buffer)
    
    # Resize to 224x224 (The size Teachable Machine expects)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image (Teachable Machine standard: -1 to 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Create the batch (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 5. Make Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 6. Display Result
    st.divider()
    
    # Extract just the name (removing the number "0" or "1" if present in labels)
    clean_name = class_name[2:] if class_name[0].isdigit() else class_name
    
    st.subheader(f"Result: {clean_name}")
    st.metric(label="Confidence Level", value=f"{confidence_score:.2%}")

    # Visual Logic for the Factory Worker
    if "Rotten" in clean_name or "Reject" in clean_name:
        st.error("⚠️ DEFECT DETECTED - REMOVE ITEM")
    elif "Fresh" in clean_name or "Pass" in clean_name:
        st.success("✅ PASSED INSPECTION")
    else:

        st.info("System Ready")
