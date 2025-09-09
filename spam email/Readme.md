📧 Email Spam Detection 

This project predicts whether an email is spam or ham using machine learning. It uses a dataset of numeric features (word counts or frequencies) and applies the XGBoost Classifier for training and prediction.

📌 Project Overview

The goal of this project is to accurately classify emails as spam (1) or ham (0). It is designed for datasets where features are already numeric, such as word frequencies or counts, making it lightweight and easy to implement.

🔑 Key Steps
Using kaggle spam email dataset.

Dataset Loading: Reads a CSV file containing numeric features and a target column (Prediction).

Feature Selection: Uses all numeric columns as input features and Prediction as the output label.

Data Splitting: Divides the data into training and testing sets for proper evaluation.

Model Training: Trains an XGBoost classifier with parameters optimized for handling imbalanced data.

Model Evaluation: Displays train accuracy, test accuracy, confusion matrix, and classification report.

User Input Prediction: Allows the user to type a custom email message and receive a spam or ham prediction based on the trained model.

📊 Performance Metrics

The project evaluates model performance using:

Train Accuracy – How well the model performs on training data.

Test Accuracy – How well it generalizes to unseen data.

Confusion Matrix – Shows correct and incorrect predictions in detail.

Precision, Recall, F1-Score – Helps understand model quality for spam detection.

🎯 Example Use Case

Load the dataset and train the model.

View accuracy and classification metrics.

Enter a custom email text (e.g., "You won a free prize") and see whether it is predicted as spam or ham.

✅ Highlights

Works with numeric datasets directly (no need for text vectorization).

Handles class imbalance automatically.

Gives probabilities for each class (spam/ham), not just a hard prediction.

Can be easily extended with other machine learning models for comparison.

📌 Notes

The input text must contain words that match the dataset’s feature names for meaningful predictions.

A very high training accuracy compared to test accuracy can indicate overfitting, which can be mitigated by tuning parameters or reducing complexity.

The project is ideal for educational purposes, small to medium datasets, and as a starting point for spam classification tasks.
