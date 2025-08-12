📊 Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn for a subscription-based service using historical customer data.
By analyzing features such as demographics, usage behavior, and account details, the model identifies customers at risk of leaving so businesses can take proactive retention measures.

🚀 Features

🧹 Data Preprocessing

Dropped irrelevant columns (RowNumber, CustomerId, Surname)

Encoded categorical variables using Label Encoding & One-Hot Encoding

Scaled numerical features with StandardScaler

🌲 Model Training

Implemented Random Forest Classifier for binary classification

📊 Model Evaluation

Measured performance using Accuracy Score, Classification Report, and Confusion Matrix

🔍 Visualization

Used Seaborn Heatmaps to visualize prediction results

📈 Full Dataset Predictions

Generated churn predictions for all customers in the dataset

🛠 Tech Stack
Python

Pandas

NumPy

scikit-learn

Matplotlib

Seaborn

📂 Dataset

The dataset contains customer demographic, account, and behavioral data along with churn labels.

https://www.kaggle.com/datasets/shubh0799/churn-modelling

📊 Model Workflow
Load and explore dataset

Preprocess data (encoding & scaling)

Train-test split for model evaluation

Train Random Forest Classifier

Evaluate model performance

Visualize confusion matrix and feature importance

Generate churn predictions for the entire dataset

📈 Results

Achieved ~86% accuracy with Random Forest Classifier

Identified key factors influencing churn to help in customer retention strategies
