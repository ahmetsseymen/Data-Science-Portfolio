# House Price Prediction

## Project Overview

This project focuses on predicting residential house prices using machine learning techniques. The objective is to analyze the factors affecting house prices, perform feature engineering, and build regression models to predict the `SalePrice` of houses.

## Dataset

The project uses the Ames Housing dataset from the Kaggle House Prices competition.

The dataset contains various numerical and categorical features describing residential properties, such as overall quality, living area, construction year, garage information, basement characteristics, and many other property-related attributes.

The target variable is `SalePrice`, which represents the sale price of each house.

## Project Workflow

The project consists of the following main steps:

1. Data Loading and Preparation
2. Exploratory Data Analysis (EDA)
3. Missing Value Analysis
4. Outlier Analysis
5. Data Preprocessing
6. Feature Engineering
7. Categorical Variable Encoding
8. Target Transformation
9. Model Training and Evaluation
10. Hyperparameter Optimization
11. Test Set Prediction

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure and characteristics of the housing dataset.

The analysis included:

- Identification of numerical and categorical variables
- Analysis of variable distributions
- Examination of categorical variables in relation to `SalePrice`
- Outlier analysis
- Missing value analysis
- Investigation of relationships between features and the target variable

## Data Preprocessing

Several preprocessing operations were applied before model training:

- Missing values were analyzed and handled
- Outliers were detected and capped where appropriate
- Categorical variables were encoded
- The `SalePrice` target variable was log-transformed for model training

## Feature Engineering

New features were created from existing property characteristics to provide additional information to the machine learning models.

Feature engineering included combining and transforming information related to areas, property age, renovation history, quality, bathrooms, bedrooms, basement characteristics, and other housing attributes.

## Machine Learning

Regression models were trained and evaluated using RMSE (Root Mean Squared Error).

The project includes model development and hyperparameter optimization using:

- LightGBM
- CatBoost

Cross-validation was used to evaluate model performance and hyperparameter optimization was performed to improve the models.

## Results

The optimized models achieved the following cross-validation results:

| Model | RMSE |
|---|---:|
| LightGBM | 0.12885 |
| CatBoost | 0.12468 |

Among the evaluated optimized models, CatBoost achieved the lower RMSE score.

### Kaggle Result

The final CatBoost model achieved a Kaggle competition submission score of **0.12548**.

## Test Set Prediction

The final model was used to generate predictions for the test dataset. The predicted house prices were transformed back to their original scale and prepared in the required submission format.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- CatBoost

## Repository Structure

```text
house-price-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── house_price_prediction.py
└── README.md
```

## Objective

The main objective of this project is to demonstrate an end-to-end regression machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, model development, hyperparameter optimization, and test set prediction.
