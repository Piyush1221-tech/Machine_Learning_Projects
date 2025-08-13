Cancer Prediction Machine Learning Model

Project Overview This project builds a machine learning model to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on clinical measurements. The goal is to assist in early detection of breast cancer using the Breast Cancer Wisconsin Dataset.

Dataset

Source: Breast Cancer Wisconsin (Diagnostic) Dataset

Features: 30 numeric features describing characteristics of cell nuclei from digitized images.

Target: Binary labels — 0 for malignant, 1 for benign.

Project Structure

cancer_prediction.py — Python script for data preprocessing, model training, evaluation, and prediction.

submission.csv — Example output file with predictions on test data.

README.txt — This file, describing the project.

Installation Make sure you have Python 3.x installed along with required libraries: pandas, scikit-learn, matplotlib, seaborn.

Usage

Load and preprocess the data (handle missing values, feature scaling).

Split the dataset into training and testing sets.

Train a Logistic Regression model on the training data.

Evaluate model performance on the test set.

Generate predictions for new/unseen data.

Save predictions to a CSV submission file.

Key Code Snippets

Scale features using StandardScaler.

Train Logistic Regression model.

Predict on test data and evaluate accuracy.

Results

Model achieves around 95% accuracy on the test set.

Confusion matrix and classification report give detailed performance metrics.

Future Improvements

Try other algorithms like Random Forest, SVM, or XGBoost.

Perform hyperparameter tuning.

Deploy as a web app for real-time use.

Add feature importance visualization.
