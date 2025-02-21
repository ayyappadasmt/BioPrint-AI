import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
MODEL_PATH = "bloodtype_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size
img_size = (128, 128)

# Blood type labels (ensure they match your training class order)
class_labels = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

# Streamlit UI Design
st.set_page_config(page_title="Blood Type Predictor", page_icon="🧬", layout="centered")

# Custom CSS for hacker-style theme
st.markdown(
    """
    <style>
        /* Hacker Theme - Pastel Green Background */
        body {
            background-color: #0d1117 !important; /* Dark hacker green */
            color: #33ff33 !important; /* Neon Green text */
            text-align: center;
            font-family: 'Courier New', monospace;
        }
        
        .stApp {
            background-color: #0d1117 !important;
        }

        h1, h2, h3 {
            color: #33ff33 !important; /* Neon Green Headers */
            text-shadow: 0 0 5px #33ff33, 0 0 10px #00ff00; /* Glowing effect */
            text-align: center;
        }

        .stButton>button {
            background-color: #33ff33 !important;
            color: black !important;
            font-weight: bold;
            border-radius: 10px;
            border: 2px solid #00ff00;
            text-shadow: 0 0 5px #00ff00;
        }

        .stTextInput>div>div>input {
            background-color: #0d1117 !important;
            color: #33ff33 !important;
            border: 2px solid #33ff33;
        }

        .stFileUploader {
            border: 2px solid #33ff33;
            color: #33ff33 !important;
        }

        .stSuccess {
            background-color: rgba(0, 255, 0, 0.1) !important;
            border: 2px solid #33ff33 !important;
            color: #33ff33 !important;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title(" Blood Type Predictor")
st.markdown("### Upload a fingerprint image to predict the blood type ")

# File uploader
uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_blood_type(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_labels[class_index]

# Display prediction
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Fingerprint", width=250)
    st.write("🔍 **Processing the fingerprint...**")
    
    # Predict blood type
    prediction = predict_blood_type(uploaded_file, model)
    
    st.success(f"**Predicted Blood Type: {prediction}**")

# Run using: `streamlit run your_script.py`
