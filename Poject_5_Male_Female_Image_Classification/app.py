import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Male vs Female Image Classifier",
    page_icon="🧑",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
try:
    model = joblib.load("male_female_model.pkl")
except FileNotFoundError:
    st.error("Model file 'male_female_model.pkl' not found.")
    st.stop()

IMG_SIZE = 64

# -------------------------
# Title
# -------------------------
st.title("🧑 Male vs Female Image Classifier")
st.write("Upload a face image to predict whether it is Male or Female.")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to NumPy array
    image = np.array(image)

    # Normalize if your training code normalized images
    image = image.astype("float32") / 255.0

    # Flatten image
    image = image.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(image)[0]

    # Prediction probabilities
    probability = model.predict_proba(image)[0]

    st.divider()

    if prediction == 0:
        st.success("👩 Prediction: FEMALE")
    else:
        st.success("👨 Prediction: MALE")

    st.subheader("Prediction Confidence")

    st.progress(float(probability[prediction]))

    st.write(f"👩 Female : **{probability[0] * 100:.2f}%**")
    st.write(f"👨 Male : **{probability[1] * 100:.2f}%**")
