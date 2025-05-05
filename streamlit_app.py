import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')  # Make sure this file exists

# Streamlit app header
st.title("Credit Card Fraud Detection")

# Sidebar for input
st.sidebar.header("Transaction Details")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.1)
time = st.sidebar.number_input("Time (in seconds)", min_value=0.0, step=0.1)

# Additional user inputs
transaction_type = st.sidebar.selectbox("Transaction Type", ["Debit", "Credit"])
location = st.sidebar.text_input("Location", "Unknown")
transaction_category = st.sidebar.text_input("Transaction Category", "Groceries")

# Scale the numeric values (e.g., Amount, Time)
scaled = scaler.transform([[amount, time]])
scaled_amount = scaled[0][0]
scaled_time = scaled[0][1]

# Map categorical inputs to numeric
transaction_type_encoded = 1 if transaction_type == "Debit" else 0

# Prepare the input features
features = [scaled_amount, scaled_time, transaction_type_encoded, location, transaction_category]

# Function to make predictions
def predict_fraud(features):
    return model.predict([features])

# Predict button
if st.sidebar.button("Predict Fraud"):
    prediction = predict_fraud(features)
    if prediction == 1:
        st.write("This transaction is **FRAUDULENT**!")
    else:
        st.write("This transaction is **NOT FRAUDULENT**.")
