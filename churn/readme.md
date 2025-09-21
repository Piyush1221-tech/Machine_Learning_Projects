# Customer Churn Prediction ML Project

## Project Overview

This project is designed to predict customer churn for a telecom company using machine learning. The goal is to identify customers who are likely to leave the service, enabling the company to take proactive retention actions.

## Dataset

The dataset includes customer information such as demographic details, services subscribed, account information, and payment methods. Key columns include:

* `customerID`: Unique ID for each customer
* `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* `tenure`, `PhoneService`, `MultipleLines`
* `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* `Contract`, `PaperlessBilling`, `PaymentMethod`
* `MonthlyCharges`, `TotalCharges`, `Churn` (target column)

## Preprocessing Steps

1. **Handle missing values**

   * Convert `TotalCharges` to numeric and fill missing values with median
2. **Encode categorical variables**

   * Convert Yes/No columns to 0/1
   * Label encode multi-category columns
3. **Feature Engineering**

   * `TotalRevenue = tenure * MonthlyCharges`
   * `Contract_Payment` interaction feature
4. **Scaling numeric columns**

   * StandardScaler applied to `tenure`, `MonthlyCharges`, `TotalCharges`, `TotalRevenue`

## Model Training

* **Train/Test Split**: 80% train, 20% test
* **Random Forest Classifier** used as primary model
* Other models considered: Logistic Regression, XGBoost

## Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score

## Usage

1. Clone the repository.
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`.
3. Load the dataset and run the preprocessing and training script.
4. Evaluate the model performance using the provided metrics.

## Key Learnings

* Preprocessing and handling missing values are crucial for ML models.
* Random Forest and boosting models often outperform linear models on churn datasets.


## Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn


---

**Author:** Piyush Tripathi
**Date:** 2025-09-21
