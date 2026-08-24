# Telco Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. The main objective is to analyze customer behavior, identify the factors associated with churn, and build classification models to predict customers who are likely to leave the company.

## Dataset

The dataset contains customer information from a telecommunications company, including demographic details, subscribed services, account information, monthly charges, total charges, and churn status.

The target variable is `Churn`, which indicates whether a customer left the company.

## Project Workflow

The project consists of the following main steps:

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Missing Value Analysis
4. Outlier Analysis
5. Feature Engineering
6. Categorical Variable Encoding
7. Feature Scaling
8. Model Training and Evaluation
9. Hyperparameter Optimization

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure of the dataset and customer behavior.

The analysis included:

- Examination of numerical and categorical variables
- Analysis of variable distributions
- Target variable analysis
- Investigation of categorical variables in relation to churn
- Missing value analysis
- Outlier analysis

## Feature Engineering

New features were created from existing variables to improve the representation of customer behavior and provide additional information for machine learning models.

Categorical variables were encoded and numerical variables were scaled before model training.

## Machine Learning Models

Multiple classification algorithms were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- CatBoost

The models were evaluated using classification performance metrics and selected models were further optimized using hyperparameter tuning.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

## Repository Structure

```text
telco-customer-churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── telco_customer_churn.py
└── README.md
```

## Objective

The main objective of this project is to demonstrate an end-to-end machine learning workflow for a customer churn classification problem, including data analysis, preprocessing, feature engineering, model comparison and model optimization.

## Results

Multiple classification models were evaluated using 10-fold cross-validation.

| Model | Accuracy | ROC-AUC | F1 Score |
|---|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.8473 | 0.5885 |
| KNN | 0.7646 | 0.7810 | 0.5428 |
| Decision Tree | 0.7295 | 0.6617 | 0.5007 |
| Random Forest | 0.7960 | 0.8292 | 0.5636 |
| SVM | 0.7926 | 0.8158 | 0.5172 |
| XGBoost | 0.7840 | 0.8220 | 0.5572 |
| LightGBM | 0.7924 | 0.8376 | 0.5667 |
| CatBoost | 0.8004 | 0.8415 | 0.5780 |

Among the baseline models, Logistic Regression achieved the highest ROC-AUC score of 0.8473 and the highest accuracy of 0.8045.
