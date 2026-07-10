import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🌸 Iris Flower Prediction",
    page_icon="🌼",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("knn_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load iris dataset
iris = load_iris()
species = iris.target_names

# -----------------------------
# Header
# -----------------------------
st.title("🌸 Iris Flower Species Prediction")
st.markdown("### Predict Iris flower species using K-Nearest Neighbors (KNN)")
st.write("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(iris.data[:,0].min()),
    float(iris.data[:,0].max()),
    5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(iris.data[:,1].min()),
    float(iris.data[:,1].max()),
    3.5
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(iris.data[:,2].min()),
    float(iris.data[:,2].max()),
    1.4
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(iris.data[:,3].min()),
    float(iris.data[:,3].max()),
    0.2
)

# -----------------------------
# Prediction
# -----------------------------
features = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

if st.button("Predict Species"):

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)

    st.success(f"### Predicted Species: **{species[prediction].title()}**")

    st.subheader("Prediction Probability")

    for i, cls in enumerate(species):
        st.write(f"**{cls.title()} : {probability[0][i]*100:.2f}%**")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.markdown("### 📌 Features Used")
st.write("""
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

st.write("---")
st.markdown("### 👨‍💻 Developed by Abhay Kumar Gupta")

col1, col2 = st.columns(2)

with col1:
    st.markdown("[🔗 GitHub](https://github.com/Abhay-cody)")

with col2:
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/abhay-kumar-gupta-104a18397)")