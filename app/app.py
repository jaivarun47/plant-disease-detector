import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("../model/plant_disease_model.keras")

with open("../model/class_names.json") as f:
    class_names = json.load(f)

st.title("🌿 Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    confidence = np.max(preds)

    st.image(img, use_column_width=True)
    st.write(f"**Prediction:** {class_names[idx]}")
    st.write(f"**Confidence:** {confidence:.2f}")
