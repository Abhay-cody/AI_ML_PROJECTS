import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🩻",
    layout="wide"
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("model.keras")

model = load_cnn()

IMG_SIZE = (299, 299)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
### Abhay Gupta

🎓 B.Tech CSE Student

🌐 **GitHub**

https://github.com/Abhay-cody

💼 **LinkedIn**

https://www.linkedin.com/in/abhay-gupta-41095828a/
""")

# ---------------------------
# Main Title
# ---------------------------
st.title("🩻 COVID-19 Detection from Chest X-Ray")

st.markdown("""
This application uses a **Convolutional Neural Network (CNN)** trained on Chest X-Ray images
to detect whether the uploaded X-Ray is **COVID-19 Positive** or **Normal**.
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # Image Preprocessing
    img = image.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    # Since your training labels are:
    # covid = 0
    # normal = 1
    # (If your class_indices are opposite, simply swap the labels below.)

    if prediction < 0.5:
        result = "🦠 COVID-19 Detected"
        confidence = (1 - prediction) * 100

        with col2:
            st.error(result)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(confidence / 100)

    else:
        result = "✅ Normal"
        confidence = prediction * 100

        with col2:
            st.success(result)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(confidence / 100)

    st.divider()

    st.subheader("Model Output")

    st.write(f"Raw Prediction Score : **{prediction:.4f}**")

st.divider()

st.markdown("""
## About Project

This project is based on a **Convolutional Neural Network (CNN)** trained to classify Chest X-Ray images into:

- 🦠 COVID-19
- ✅ Normal

The uploaded image is automatically resized to **299 × 299 pixels**, normalized, and passed through the trained deep learning model for prediction.

---
### ⚠ Disclaimer

This application is intended **only for educational and research purposes**.
It should **not** be used as a substitute for professional medical diagnosis.
""")
