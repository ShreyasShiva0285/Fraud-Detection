import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the model
model = joblib.load("fraud_model.pkl")

# Function to make predictions
def predict(model, X):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Safely get class 1 probabilities
    if probabilities.shape[1] == 1:
        prob_class_1 = probabilities[:, 0]  # Only one column exists
    else:
        prob_class_1 = probabilities[:, 1]  # Normal case

    return predictions, prob_class_1

# Title
st.title("Credit Card Fraud Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Assume the target column is 'Class' (0 = non-fraud, 1 = fraud)
    if 'Class' in data.columns:
        X = data.drop('Class', axis=1)
        y = data['Class']
    else:
        X = data
        y = None

    # Make predictions
    predictions, probabilities = predict(model, X)

    # Add results to the dataframe
    data['Prediction'] = predictions
    data['Fraud Probability'] = probabilities

    st.subheader("Prediction Results")
    st.dataframe(data[['Prediction', 'Fraud Probability']].head())

    # Show fraud counts
    fraud_count = np.sum(predictions)
    total = len(predictions)
    st.write(f"Detected {fraud_count} fraudulent transactions out of {total}")

    # If true labels exist, show metrics
    if y is not None:
        st.subheader("Model Performance")
        st.text("Classification Report:")
        st.text(classification_report(y, predictions))
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y, predictions))
        st.text(f"ROC-AUC Score: {roc_auc_score(y, probabilities):.2f}")
else:
    st.info("Upload a CSV file to start fraud detection.")
