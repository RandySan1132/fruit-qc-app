import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import tf_keras
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# 1. Configuration
st.set_page_config(page_title="Fruit QC Live", page_icon="🍎")
st.title("🍎 Live Factory Scanner")
st.write("Point camera at object for real-time analysis")

# 2. Load Model & Labels
@st.cache_resource
def load_model_and_labels():
    # Force absolute path
    model_path = os.path.abspath("keras_model.h5")
    model = tf_keras.models.load_model(model_path, compile=False)
    
    # Load labels
    try:
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except:
        class_names = ["Error Loading Labels"]
        
    return model, class_names

try:
    model, class_names = load_model_and_labels()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Define the Live Processor
# This function runs for EVERY single video frame
class VideoProcessor:
    def recv(self, frame):
        # Convert video frame to an image we can process
        img = frame.to_ndarray(format="bgr24")
        
        # --- AI PREDICTION LOGIC ---
        # 1. Resize to 224x224 (Model Requirement)
        # We use OpenCV here because it's faster for video than Pillow
        img_resized = cv2.resize(img, (224, 224))
        
        # 2. Normalize (same as before: -1 to 1)
        img_array = np.asarray(img_resized)
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        
        # 3. Shape for Model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # 4. Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # --- VISUALIZATION LOGIC ---
        # Clean the name (remove "0 ", "1 ")
        clean_name = class_name[2:] if class_name[0].isdigit() else class_name
        
        # Decide Color: Green for Fresh, Red for Rotten
        if "Rotten" in clean_name or "Reject" in clean_name:
            color = (0, 0, 255) # Red (BGR format)
            status = "REJECT"
        elif "Fresh" in clean_name or "Pass" in clean_name:
            color = (0, 255, 0) # Green
            status = "PASS"
        else:
            color = (255, 0, 0) # Blue
            status = "WAITING"

        # Draw the Rectangle and Text on the video frame
        # (x, y) coordinates for text
        cv2.putText(img, f"{status}: {int(confidence_score*100)}%", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, clean_name, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Return the annotated frame back to the screen
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. The Live Camera Component
# "STUN Server" is needed to make it work on Mobile Phones over 4G/5G
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="fruit-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.divider()
st.info("💡 Note: If the video is laggy, it is because the free cloud server is processing every frame.")
