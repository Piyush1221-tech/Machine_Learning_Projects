Credit Card Fraud Detection (ML Model)

This project detects fraudulent credit card transactions using a Machine Learning model trained on an imbalanced dataset.
It uses XGBoost for better accuracy and speed, and SMOTE for handling class imbalance.

Project Structure

fraud_det.ipynb → Jupyter Notebook with data processing and model training
fraud_model_simple.joblib → Saved trained model
data.csv → Sample dataset (optional)
README.md → Project documentation

How It Works

Data Preprocessing – Clean and scale features, handle missing values.

Handle Class Imbalance – Apply SMOTE to balance fraud and non-fraud transactions.

Train Model – Use XGBoost Classifier for fast and accurate training.

Save Model – Store the trained model using joblib for future use.

Predict – Load the model later and make predictions on new transaction data.

Requirements

Install dependencies:
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib

Usage

Train the Model (Optional)
Run fraud_det.ipynb in Jupyter Notebook to train the model and save it.

Use the Saved Model to Predict
Example:
import joblib
import pandas as pd

model = joblib.load('fraud_model_simple.joblib')
new_data = pd.read_csv('data.csv')
predictions = model.predict(new_data)
print("Predictions:", predictions)
probabilities = model.predict_proba(new_data)[:, 1]
print("Fraud Probability:", probabilities)

Dataset

This project is based on the Kaggle Credit Card Fraud Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Features
Handles imbalanced datasets with SMOTE.

Fast and accurate XGBoost model.

Can be integrated into real-time fraud detection systems.

Reusable trained model (.joblib) for quick predictions.

Notes
Ensure the new data has the same columns/features as the training dataset.

The data.csv provided is only a sample — replace it with your own transaction data for predictions.
