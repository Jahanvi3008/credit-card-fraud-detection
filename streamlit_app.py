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
    raw_data = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(raw_data.head())

    # ----------- ✅ Preprocessing -----------
    data = raw_data.copy()

    # Drop unnecessary columns if they exist
    cols_to_drop = ["transaction_id", "time", "is_fraud"]
    data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)

    # Encode 'transaction_type' as numbers
    if "transaction_type" in data.columns:
        mapping = {"online": 0, "in-store": 1, "mobile": 2}
        data["transaction_type"] = data["transaction_type"].map(mapping).fillna(-1)

    # Fill any missing values just in case
    data = data.fillna(0)

    # Optional: reorder or select specific columns if needed
    # expected_cols = ['amount', 'transaction_type']
    # data = data[expected_cols]

    # ----------- ✅ Prediction -----------
    try:
        prediction = model.predict(data)
        raw_data["Fraud Prediction"] = prediction
        st.success("✅ Prediction completed!")
        st.dataframe(raw_data)
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
