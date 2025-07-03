
import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("Scaling.joblib")
model = joblib.load("Model.joblib")

# Page title
st.title("Credit Card Fraud Detection")

st.markdown("Enter transaction details below and click **Predict** to see if it's fraud or not.")

# Feature inputs
features = []
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Input fields for each column
for col in columns:
    val = st.number_input(f"{col}:", value=0.0, format="%.6f")
    features.append(val)

# Predict button
if st.button("Predict"):
    # Convert input to 2D array
    input_array = np.array([features])

    # Scale input
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ This transaction is predicted as **FRAUD**.")
    else:
        st.success("✅ This transaction is predicted as **NOT fraud**.")
