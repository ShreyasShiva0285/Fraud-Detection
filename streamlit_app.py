import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection System")
st.markdown("Upload transaction data to detect potential fraud")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function for preprocessing
def preprocess_data(df):
    # Basic preprocessing
    # For real application, more complex feature engineering would be done
    if 'Amount' in df.columns:
        scaler = StandardScaler()
        df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    
    return df

# Function to train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def predict(model, X):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Display raw data
    st.subheader("Raw Data Preview")
    st.write(df.head())
    
    # Data preprocessing
    st.subheader("Data Preprocessing")
    
    # Check if the dataset has the expected structure
    if 'Class' in df.columns or st.checkbox("Specify fraud label column"):
        if 'Class' not in df.columns:
            target_col = st.selectbox("Select the column that indicates fraud (1 for fraud, 0 for legitimate)", df.columns)
            df['Class'] = df[target_col]
    
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Split data for demonstration
        features = [col for col in processed_df.columns if col != 'Class']
        X = processed_df[features]
        y = processed_df['Class']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        st.write(f"Training data shape: {X_train.shape}")
        st.write(f"Testing data shape: {X_test.shape}")
        
        # Train model
        with st.spinner('Training model...'):
            model = train_model(X_train, y_train)
            st.success('Model trained successfully!')
        
        # Make predictions
        predictions, probabilities = predict(model, X_test)
        
        # Create results DataFrame
        results = X_test.copy()
        results['Actual'] = y_test
        results['Predicted'] = predictions
        results['Fraud_Probability'] = probabilities
        
        # Display results
        st.subheader("Detection Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = (predictions == y_test).mean()
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            precision = (predictions & y_test).sum() / predictions.sum() if predictions.sum() > 0 else 0
            st.metric("Precision", f"{precision:.2%}")
        
        with col3:
            recall = (predictions & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
            st.metric("Recall", f"{recall:.2%}")
        
        # Display fraudulent transactions
        st.subheader("Potential Fraudulent Transactions")
        fraud_threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)
        high_prob = results[results['Fraud_Probability'] >= fraud_threshold].sort_values(by='Fraud_Probability', ascending=False)
        
        if not high_prob.empty:
            st.dataframe(high_prob)
            
            # Plot fraud probability distribution
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(results['Fraud_Probability'], bins=50, kde=True, ax=ax)
            plt.axvline(x=fraud_threshold, color='red', linestyle='--')
            plt.title('Distribution of Fraud Probabilities')
            plt.xlabel('Fraud Probability')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importances = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importances.head(10), ax=ax)
            plt.title('Top 10 Important Features')
            st.pyplot(fig)
        else:
            st.info("No transactions above the threshold.")
    else:
        st.error("This dataset does not have the expected structure. Please upload a CSV with a 'Class' column indicating fraud (1) or legitimate (0) transactions.")
else:
    st.info("Please upload a CSV file with transaction data.")

# Add explanatory section
st.sidebar.title("About")
st.sidebar.info(
    "This application uses machine learning to detect potential credit card fraud. "
    "Upload a CSV file with transaction data to get started. "
    "The model will analyze the transactions and highlight those that appear suspicious."
)

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload a CSV file with transaction data
    2. Review the raw data preview
    3. Wait for the model to process and analyze the data
    4. Examine the fraudulent transactions identified
    5. Adjust the probability threshold as needed
    """
)

st.sidebar.title("Model Details")
st.sidebar.markdown(
    """
    This application uses a Random Forest classifier trained to distinguish between legitimate and fraudulent transactions. 
    
    Key features used in fraud detection:
    - Transaction amount
    - Time since previous transaction
    - Merchant category
    - Transaction location
    - And many other anonymized features
    """
)
