from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ANN Salary Predictor",
    page_icon="💼",
    layout="centered",
)

st.title("💼 ANN Salary Predictor")
st.caption("A classroom ANN regression project built from the Bank Churn dataset.")

MODEL_PATH = BASE_DIR / "salary_model.keras"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

required_files = [MODEL_PATH, ENCODER_PATH, SCALER_PATH]
missing_files = [p.name for p in required_files if not p.exists()]

if missing_files:
    st.error(
        "The model files are missing: "
        + ", ".join(missing_files)
        + ". Run `python train_model.py` after adding Churn_Modelling.csv."
    )
    st.stop()


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH, compile=False)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, encoder, scaler


model, encoder, scaler = load_artifacts()

st.markdown(
    "Enter the customer's information below. The model estimates the "
    "customer's salary as a continuous numeric value."
)

with st.form("salary_form"):
    st.subheader("Customer details")

    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=650, step=1
    )
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
        num_products = st.number_input(
            "Number of Products", min_value=1, max_value=4, value=1, step=1
        )

    with col2:
        balance = st.number_input(
            "Account Balance", min_value=0.0, max_value=300000.0,
            value=75000.0, step=1000.0
        )
        has_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
        active_member = st.selectbox(
            "Is Active Member", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

    submitted = st.form_submit_button("Predict Estimated Salary", use_container_width=True)

if submitted:
    raw_input = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "Geography": geography,
        "Gender": gender,
    }])

    numeric_columns = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember"
    ]
    categorical_columns = ["Geography", "Gender"]

    numeric_part = scaler.transform(raw_input[numeric_columns])
    categorical_part = encoder.transform(raw_input[categorical_columns])
    final_input = np.hstack([numeric_part, categorical_part]).astype("float32")

    prediction = float(model.predict(final_input, verbose=0).ravel()[0])
    prediction = max(0.0, prediction)

    st.success("Prediction generated successfully.")
    st.metric("Estimated Salary", f"${prediction:,.2f}")

 

st.divider()
st.caption("ANN Salary Regression • TensorFlow/Keras • Streamlit")
