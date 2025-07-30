import streamlit as st
import pandas as pd
import joblib

st.title("💳 Credit Card Fraud Detection")
st.write("Upload CSV transaction data to check for fraud.")

# Load trained model
model = joblib.load("fraud_model.pkl")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of data:")
    st.dataframe(data.head())

    # Predict
    prediction = model.predict(data)
    data["Fraud Prediction"] = prediction
    st.write("Prediction Results:")
    st.dataframe(data)

