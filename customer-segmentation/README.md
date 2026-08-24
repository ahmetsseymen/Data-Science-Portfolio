# Customer Segmentation

## Project Overview

This project focuses on segmenting FLO customers based on their shopping behavior using unsupervised machine learning techniques.

The objective is to identify meaningful customer groups by analyzing purchasing patterns, customer activity, and channel behavior.

## Dataset

The dataset contains customer transaction information from FLO, including:

- First and last order dates
- Online and offline order counts
- Online and offline purchase values
- Customer category interests
- Shopping channel information

## Project Workflow

The project consists of the following main steps:

1. Data Preparation
2. Feature Engineering
3. Feature Scaling
4. K-Means Clustering
5. Elbow Method
6. Silhouette Analysis
7. Cluster Profiling
8. Hierarchical Clustering

## Feature Engineering

New variables were created to better represent customer behavior:

- `tenure`: Time since the customer's first order
- `recency`: Time since the customer's most recent order
- `category_count`: Number of categories the customer is interested in
- `total_order`: Total number of online and offline orders
- `total_value`: Total online and offline purchase value
- `online_order_ratio`: Proportion of online orders
- `online_value_ratio`: Proportion of online purchase value

## Feature Scaling

Numerical features were standardized using `StandardScaler` before clustering.

Scaling was applied to ensure that variables with different numerical ranges had comparable influence on the clustering algorithms.

## K-Means Clustering

K-Means clustering was applied to identify customer segments.

The number of clusters was investigated using:

- Elbow Method
- Silhouette Score

Different cluster configurations were examined and customer groups were analyzed based on their behavioral characteristics.

## Hierarchical Clustering

Hierarchical clustering was also applied using Ward linkage.

A dendrogram was used to examine the hierarchical structure of the customers, and Agglomerative Clustering was used to create alternative customer segments.

## Cluster Profiling

After assigning customers to clusters, the segments were analyzed by comparing their average behavioral characteristics.

This analysis helps interpret the differences between customer groups and provides a basis for developing segment-specific marketing strategies.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Repository Structure

```text
customer-segmentation/
│
├── data/
│   └── flo_data_20k.csv
│
├── customer_segmentation.py
└── README.md
```

## Objective

The main objective of this project is to demonstrate an unsupervised machine learning workflow for customer segmentation, including feature engineering, scaling, cluster selection, K-Means clustering, hierarchical clustering, and customer segment analysis.
