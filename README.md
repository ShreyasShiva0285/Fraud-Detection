# Fraud Detection

**Fraud Detection** is a machine learning-powered application designed to detect fraudulent transactions in real-time. The tool analyzes transaction data to assess the likelihood of fraud based on patterns and historical data. It helps financial institutions and businesses in identifying and preventing fraudulent activities.

## Project Overview

The Fraud Detection application uses machine learning algorithms to classify transactions as either fraudulent or legitimate. By analyzing features such as transaction amount, time, location, and other financial indicators, the system provides real-time insights into potential fraud. The application provides actionable alerts, helping organizations take immediate action.

## Key Features

- **Real-Time Fraud Detection**: Analyzes incoming transaction data and classifies it as fraudulent or legitimate.
- **Probability-Based Alerts**: Provides probability scores for fraud likelihood, enabling risk-based decision-making.
- **Interactive Dashboard**: Built with Streamlit, the app presents predictions and insights in an interactive interface.
- **Data Preprocessing**: Includes feature engineering and normalization techniques for better model performance.
- **Model Explainability**: The app provides insights into model decisions using techniques like SHAP (Shapley Additive Explanations).

## Technologies Used

- **Machine Learning**: XGBoost, Random Forest
- **Data Handling**: Pandas, Numpy
- **Model Explainability**: SHAP (Shapley Additive Explanations)
- **Frontend**: Streamlit (for real-time interactive interface)
- **Visualization**: Matplotlib, Seaborn

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ShreyasShiva0285/Fraud-detection.git
