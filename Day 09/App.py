import streamlit as st
import pickle
import pandas as pd
from sklearn.datasets import load_iris

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# ----------------------------------------------------
# Load Trained Model
# ----------------------------------------------------
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Load Iris Dataset (Only for labels and preview)
iris = load_iris()

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.title("🌸 Iris Flower Classification System")

st.write(
    "Predict the species of an Iris flower using a trained Logistic Regression model."
)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.header("🌼 Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    2.0,
    4.5,
    3.5
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    1.4
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    0.1,
    2.5,
    0.2
)

# ----------------------------------------------------
# Display User Input
# ----------------------------------------------------
st.subheader("📋 Entered Values")

input_data = pd.DataFrame({
    "Feature": [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ],
    "Value": [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
})

st.table(input_data)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
sample = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

if st.button("🔍 Predict Species"):

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    species = iris.target_names[prediction[0]]

    st.success(f"🌸 Predicted Species: **{species.title()}**")

    st.subheader("Prediction Confidence")

    for i, flower in enumerate(iris.target_names):
        st.write(f"**{flower.title()} : {probability[0][i]*100:.2f}%**")

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------
st.markdown("---")

st.subheader("📊 Iris Dataset Preview")

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = [
    iris.target_names[i]
    for i in iris.target
]

st.dataframe(df.head())

# ----------------------------------------------------
# About Dataset
# ----------------------------------------------------
with st.expander("ℹ️ About the Iris Dataset"):

    st.write("""
The Iris dataset is one of the most famous datasets in Machine Learning.

It contains **150 flower samples** belonging to **3 different species**:

- Setosa
- Versicolor
- Virginica

Each flower has four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

A Logistic Regression model is trained to predict the species based on these measurements.
""")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")

st.caption(
    "Developed by Shahzaib Salamat | Machine Learning Internship - Day 9"
)
