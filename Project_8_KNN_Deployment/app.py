import streamlit as st
import numpy as np
import joblib

# Page Config
st.set_page_config(
    page_title="🌸 Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load Model
try:
    model = joblib.load("knn_model.pkl")
except Exception as e:
    st.error(f"Error Loading Model: {e}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align:center;
    color:#4CAF50;
    font-size:40px;
    font-weight:bold;
}
.subtitle {
    text-align:center;
    color:#666;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🌸 Iris Flower Classification AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Iris Species using K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Input Features")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 4.0, 8.0, 5.8
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 2.0, 5.0, 3.0
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", 1.0, 7.0, 4.0
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", 0.1, 3.0, 1.3
)

input_data = np.array([
    [sepal_length, sepal_width, petal_length, petal_width]
])

# Prediction
if st.button("🔍 Predict Species", use_container_width=True):

    prediction = model.predict(input_data)[0]

    species = {
        0: "🌼 Setosa",
        1: "🌺 Versicolor",
        2: "🌷 Virginica"
    }

    st.success(f"### Prediction: {species[prediction]}")

    st.markdown("### 📋 Input Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sepal Length", f"{sepal_length} cm")
        st.metric("Sepal Width", f"{sepal_width} cm")

    with col2:
        st.metric("Petal Length", f"{petal_length} cm")
        st.metric("Petal Width", f"{petal_width} cm")

# Footer
st.markdown("---")
st.markdown(
    """
    ### 🚀 About Project
    This Machine Learning application uses the **K-Nearest Neighbors (KNN)** algorithm
    to classify Iris flowers into:

    - 🌼 Setosa
    - 🌺 Versicolor
    - 🌷 Virginica

    Developed using **Python, Scikit-Learn, and Streamlit**.
    """
)

st.markdown("---")
st.markdown(
    "👨‍💻 Created by **Richeek Pandey** | "
    "🔗 GitHub:  https://github.com/richeekpandey07 "
    " 💼 LinkedIn: www.linkedin.com/in/richeek-pandey-9954783a9 "
)
