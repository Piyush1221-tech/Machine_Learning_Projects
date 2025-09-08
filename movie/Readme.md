🎬 Movie Success Prediction (Hit vs Flop)

This project predicts whether a movie will be a hit or a flop based on multiple features like budget, revenue, runtime, popularity, genres, and production country.

📌 Project Overview

Every year, thousands of movies are released, but only a few become successful. This project leverages machine learning to predict a movie’s success before release. The model can be used by producers, distributors, and investors to make informed decisions.

📊 Dataset & Features
Features Used

Log Budget – Log-transformed budget to reduce skewness

Runtime – Movie duration in minutes

Popularity – Popularity score from IMDb

Genre Count – Number of genres the movie belongs to

Adjusted Revenue – Revenue normalized by release year’s median

Top Production Countries – One-hot encoded top 5 producing countries

Target Variable

Hit or Flop – Binary classification: 1 = Hit, 0 = Flop

🛠 Data Preprocessing

Removed rows with missing or zero budget/revenue

Dropped movies with missing release dates

Applied log transformation on budget and revenue

Removed outliers using IQR filtering

Created new features: genre_count, adj_revenue, top_country

Applied one-hot encoding for categorical variables

🤖 Model Used

The project uses a Random Forest Classifier as the primary machine learning model because it performs well on classification tasks and handles non-linear data efficiently.

📈 Model Performance
Metric	Score
Training Accuracy	99.34%
Test Accuracy	95.51%

✅ Interpretation: The model shows strong performance with minimal overfitting and generalizes well to unseen data.

🎯 Challenges Faced

Handling missing and zero values in critical columns

Reducing skewness and normalizing data

Preventing overfitting (initial training accuracy was 100%)

Choosing meaningful features to improve prediction accuracy

🔮 Predictions

The trained model can take user input data (like budget, runtime, popularity, etc.) and predict whether a movie will be a hit or a flop.

🚀 Future Improvements

Use text-based features like movie overview and keywords with NLP

Experiment with advanced models (XGBoost, LightGBM, Neural Networks)

Build an interactive web application using Streamlit

Perform deeper feature selection and hyperparameter optimization

📌 Tech Stack

Python for implementation

Pandas, NumPy for data analysis

Matplotlib, Seaborn for visualization

Scikit-learn for machine learning model

Jupyter Notebook for experimentation
