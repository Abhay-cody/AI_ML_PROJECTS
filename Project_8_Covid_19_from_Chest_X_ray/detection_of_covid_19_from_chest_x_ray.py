import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- Page Configuration ---------------- #
st.set_page_config(
    page_title="COVID-19 Detection from Chest X-Ray",
    page_icon="🩻",
    layout="wide"
)

# ---------------- Load Model ---------------- #
@st.cache_resource
def load_cnn():
    try:
        model = load_model("model.keras")
    except:
        model = load_model("model.h5")
    return model

model = load_cnn()

IMG_SIZE = (299, 299)

# ---------------- Sidebar ---------------- #
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
### **Abhay Gupta**

🎓 B.Tech CSE Student

🔗 **GitHub**

https://github.com/Abhay-cody

💼 **LinkedIn**

https://www.linkedin.com/in/abhay-gupta-41095828a/

---
Built with ❤️ using Streamlit & TensorFlow.
""")

# ---------------- Main Title ---------------- #
st.title("🩻 COVID-19 Detection from Chest X-Ray")

st.write("""
Upload a **Chest X-Ray image** and the trained CNN model will predict whether the patient is:

- ✅ Normal
- ⚠️ COVID-19
""")

uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Prediction ---------------- #
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        label = "🦠 COVID-19"
        confidence = prediction * 100
        color = "error"
    else:
        label = "✅ NORMAL"
        confidence = (1 - prediction) * 100
        color = "success"

    with col2:

        st.subheader("Prediction")

        if color == "success":
            st.success(label)
        else:
            st.error(label)

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(float(confidence / 100))

        st.write("Raw Model Output:", round(float(prediction), 4))

st.markdown("---")

st.markdown("""
### About the Project

This project uses a **Convolutional Neural Network (CNN)** trained on Chest X-Ray images to classify:

- COVID-19
- Normal

The uploaded image is resized to **299 × 299 pixels**, normalized, and passed through the trained CNN model.

---

### ⚠️ Disclaimer

This application is intended **only for educational and research purposes** and **must not** be used for medical diagnosis.
""")
