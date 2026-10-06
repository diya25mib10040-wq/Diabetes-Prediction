import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# --------------------------------------------------
# Load Saved Model and Preprocessing Files
# --------------------------------------------------

from pathlib import Path

# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
MEDIAN_PATH = BASE_DIR / "models" / "median_values.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
median_values = joblib.load(MEDIAN_PATH)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩺 Early Diabetes Prediction")

st.write(
    "Enter the patient's health information below "
    "to generate a machine-learning-based prediction."
)

st.info(
    "This application is for educational and research purposes "
    "and should not be used as a medical diagnosis."
)


# --------------------------------------------------
# Input Fields
# --------------------------------------------------

st.subheader("Patient Information")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    max_value=250.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=150.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Diabetes Risk"):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Replace invalid zero values with
    # the medians calculated during preprocessing

    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in zero_columns:
        if input_data[column].iloc[0] == 0:
            input_data[column] = median_values[column]

    # Determine whether the model requires scaling
    model_name = type(model).__name__

    if model_name == "GaussianNB":
        input_processed = scaler.transform(input_data)
    else:
        input_processed = input_data

    prediction = model.predict(input_processed)[0]

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "⚠️ The model predicts a higher likelihood of diabetes."
        )
    else:
        st.success(
            "✅ The model predicts a lower likelihood of diabetes."
        )

    # Display prediction probability if supported

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_processed
        )[0][1]

        st.write(
            f"Estimated model probability: "
            f"**{probability * 100:.2f}%**"
        )