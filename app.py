import streamlit as st
import os

st.title("Debug Mode 🐞")

# 1. PRINT THE CURRENT WORKING DIRECTORY
st.write("Current Directory:", os.getcwd())

# 2. LIST ALL FILES IN THIS DIRECTORY
files = os.listdir()
st.write("Files found here:", files)

# 3. CHECK IF MODEL EXISTS SPECIFICALLY
if "keras_model.h5" in files:
    st.success("✅ The file 'keras_model.h5' is FOUND!")
    
    # Try to verify size
    size = os.path.getsize("keras_model.h5")
    st.write(f"File size: {size} bytes")
    
    if size < 1000:
        st.error("⚠️ The file is too small! It might be corrupted or a Git Pointer.")
    else:
        st.info("File size looks normal.")
        
else:
    st.error("❌ The file 'keras_model.h5' is MISSING.")

# --- STOP HERE IF TESTING ---
# (Once you see the green success message, you can paste the real code back in)
