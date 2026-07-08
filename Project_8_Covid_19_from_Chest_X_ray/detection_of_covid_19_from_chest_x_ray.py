import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🩺",
    layout="centered"
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("model.keras")

model = load_cnn()

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(img):

    img = img.resize((299,299))

    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:,:,:3]

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ------------------------------
# Header
# ------------------------------
st.title("🩺 COVID-19 Chest X-Ray Detection")

st.write(
    "Upload a Chest X-Ray image to predict whether it is **COVID** or **NORMAL**."
)

# ------------------------------
# Upload Image
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess_image(image)

    if st.button("Predict"):

        prediction = model.predict(img)[0][0]

        confidence = prediction if prediction > 0.5 else 1-prediction

        if prediction > 0.5:

            st.error("## Prediction : COVID")

        else:

            st.success("## Prediction : NORMAL")

        st.info(f"Confidence : **{confidence*100:.2f}%**")

        st.progress(float(confidence))

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("About")

st.sidebar.info("""
Deep Learning CNN Model

Dataset:
COVID-19 Chest X-Ray Images

Framework:
TensorFlow + Streamlit
""")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.markdown(
"""
### 👨‍💻 Developed by Abhay Kumar Gupta

🔗 **LinkedIn**

https://www.linkedin.com/in/abhay-kumar-gupta-104a18397/

💻 **GitHub**

https://github.com/Abhay-cody
"""
)
