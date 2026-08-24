"""
Customer Segmentation

This project focuses on segmenting FLO customers based on their
shopping behavior using unsupervised machine learning techniques.

Main steps:
- Data Preparation
- Feature Engineering
- Feature Scaling
- K-Means Clustering
- Elbow and Silhouette Analysis
- Hierarchical Clustering
- Cluster Profiling
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ==================================================
# 1. Data Preparation
# ==================================================

df_ = pd.read_csv("data/flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)

check_df(df,head=5)

date_cols = [col for col in df.columns if "date" in col]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])

df.dtypes

# ==================================================
# 2. Feature Engineering
# ==================================================

analys_date = df["last_order_date"].max()

df["tenure"] = (analys_date  - df["first_order_date"]).dt.days

df["recency"] = (analys_date - df["last_order_date"]).dt.days

df["interested_in_categories_12"].head()
type(df["interested_in_categories_12"].iloc[0])
df["category_count"] = df["interested_in_categories_12"].str.count(",") + 1

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["online_order_ratio"] = df["order_num_total_ever_online"] / df["total_order"]

df["online_value_ratio"] = df["customer_value_total_ever_online"] / df["total_value"]

features = [
    "tenure",
    "recency",
    "total_order",
    "total_value",
    "online_order_ratio",
    "online_value_ratio",
    "category_count"
]


dfC = df[features].copy()

# ==================================================
# 3. Feature Scaling
# ==================================================

scaler = StandardScaler()
dfC[dfC.columns] = scaler.fit_transform(dfC)

# ==================================================
# 4. K-Means Clustering
# ==================================================

# Elbow Method

inertia_values=[]

for k in range(2,11):
    model = KMeans(n_clusters=k, random_state=11)
    model.fit(dfC)
    inertia_values.append(model.inertia_)

plt.plot(range(2, 11), inertia_values)
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Silhouette Analysis

silhouette_scores = []
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=11)
    labels = model.fit_predict(dfC)
    score = silhouette_score(dfC, labels)
    silhouette_scores.append(score)


plt.plot(range(2, 11), silhouette_scores)
plt.xlabel("Küme Sayısı")
plt.ylabel("score")
plt.title("silhouette Method")
plt.show()

# Cluster Modeling and Profiling

model_2 = KMeans(n_clusters=2, random_state=11)
cluster_2 = model_2.fit_predict(dfC)

df["cluster_2"] = cluster_2

model_3 = KMeans(n_clusters=3, random_state=11)
cluster_3 = model_3.fit_predict(dfC)

df["cluster_3"] = cluster_3

print("K = 2")
print(df.groupby("cluster_2")[features].mean())

print("\nK = 3")
print(df.groupby("cluster_3")[features].mean())

# ==================================================
# 5. Hierarchical Clustering
# ==================================================

linkage_matrix = linkage(dfC, method="ward")

dendrogram(linkage_matrix)
plt.show()

modelH_3 = AgglomerativeClustering(n_clusters=3)
clusterH_3 = modelH_3.fit_predict(dfC)

df["clusterH_3"] = clusterH_3


modelH_4 = AgglomerativeClustering(n_clusters=4)
clusterH_4 = modelH_4.fit_predict(dfC)

df["clusterH_4"] = clusterH_4


print("clusterH_3 = 3")
print(df.groupby("clusterH_3")[features].mean())

print("clusterH_4 = 4")
print(df.groupby("clusterH_4")[features].mean())

df["clusterH_4"].value_counts()
