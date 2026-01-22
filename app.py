import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Configuration
st.set_page_config(page_title="Fruit Quality Control", page_icon="🍎")

st.title("🍎 Factory AI Inspector")
st.write("Visual Quality Control System (PoC)")

# 2. Robust Model Loader
@st.cache_resource
def load_model_from_file():
    # Force absolute path to avoid confusion
    model_path = os.path.abspath("keras_model.h5")
    
    # Load the model using TensorFlow's direct Keras loader
    # compile=False is crucial for Teachable Machine models
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Load Labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("❌ Error: 'labels.txt' is missing. Please upload it.")
    st.stop()

# Load Model (Without hiding errors)
try:
    model = load_model_from_file()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Tip: This often happens if the 'keras_model.h5' is corrupted or incompatible.")
    st.stop()

# 3. The Camera Interface
st.write("### 📸 Scan Product")
img_file_buffer = st.camera_input("Take a picture of the apple")

if img_file_buffer is not None:
    # 4. Pre-process
    image = Image.open(img_file_buffer)
    
    # Resize to 224x224 (Standard for Teachable Machine)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert and Normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 5. Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 6. Result Display
    st.divider()
    
    # Clean up class name (Remove "0 ", "1 " prefix)
    clean_name = class_name[2:] if class_name[0].isdigit() else class_name
    
    st.subheader(f"Result: {clean_name}")
    st.metric(label="Confidence", value=f"{confidence_score:.2%}")

    if "Rotten" in clean_name or "Reject" in clean_name:
        st.error("⚠️ DEFECT DETECTED - REJECT")
    elif "Fresh" in clean_name or "Pass" in clean_name:
        st.success("✅ QUALITY CHECK PASSED")
    else:
        st.info(f"Detected: {clean_name}")
