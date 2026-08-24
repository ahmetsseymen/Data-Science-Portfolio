import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
import warnings
warnings.simplefilter(action="ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()

df.head()

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

df.columns = [col.upper() for col in df.columns]

df["CHURN"] = df["CHURN"].map({"Yes": 1, "No": 0})

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kardinal değişken isimlerini verir.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        İncelenmek istenen dataframe.
    cat_th : int, optional (default=10)
        Nümeric fakat kategorik sayılabilecek değişkenler için eşik değer.
    car_th : int, optional (default=20)
        Kategorik fakat kardinal (çok fazla benzersiz değere sahip) değişkenler için eşik değer.

    Returns
    -------
    cat_cols : list
        Kategorik değişken isimleri.
    num_cols : list
        Nümeric değişken isimleri.
    cat_but_car : list
        Kategorik görünümlü kardinal değişken isimleri.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# 1. TotalCharges sütununu sayısala çevirme (Tip Düzeltmesi)
df["TOTALCHARGES"] = pd.to_numeric(df["TOTALCHARGES"], errors="coerce")

# 2. SeniorCitizen sütununu kategorik yapma (Kategori Düzeltmesi)
df["SENIORCITIZEN"] = df["SENIORCITIZEN"].astype("object")

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "CHURN", col)



def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "CHURN", col)


# Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    """Sayısal değişken için IQR yöntemine göre alt ve üst eşik değerleri hesaplar."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """Değişkende aykırı değer var mı yok mu kontrol eder (True/False)."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 6: Eksik gözlem var mı inceleyiniz.

def missing_values_table(dataframe, na_name=False):
    """Eksik verilerin sayısını ve oransal dağılımını tablo olarak verir."""
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
df.head()

#GÖREV 2

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df[df["TENURE"] == 0][["TENURE", "TOTALCHARGES"]]

df["TOTALCHARGES"] = df["TOTALCHARGES"].fillna(0)

# Adım 2: Yeni değişkenler oluşturunuz.

df.head()

df["TENURE"].sort_values(ascending=False).head()

df.loc[(df["TENURE"] <= 12),"NEW_TENSURE_CATEGORY"] = "Newbie"
df.loc[(df["TENURE"] >12) & (df["TENURE"] <= 36) , "NEW_TENSURE_CATEGORY"] = "Regular"
df.loc[(df["TENURE"] >36) , "NEW_TENSURE_CATEGORY"] = "Loyal"

df.loc[(df["TENURE"]>=0) & (df["TENURE"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["TENURE"]>12) & (df["TENURE"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["TENURE"]>24) & (df["TENURE"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["TENURE"]>36) & (df["TENURE"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["TENURE"]>48) & (df["TENURE"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["TENURE"]>60) & (df["TENURE"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"
df.loc[(df["TENURE"]>72) ,"NEW_TENURE_YEAR"] = "+ 6 Year"

df["NEW_AVG_MONTHLY_CHARGE"] = df["TOTALCHARGES"] / (df["TENURE"] + 1e-5)
df["NEW_CHARGE_INCREASE"] = (df["MONTHLYCHARGES"] - df["NEW_AVG_MONTHLY_CHARGE"])
df["NEW_YEARLY_COST"] = df["MONTHLYCHARGES"] * 12
services = [
    "ONLINESECURITY",
    "ONLINEBACKUP",
    "DEVICEPROTECTION",
    "TECHSUPPORT",
    "STREAMINGTV",
    "STREAMINGMOVIES",
]
df["NEW_TOTAL_SERVICES"] = (df[services] == "Yes").sum(axis=1)
df["NEW_HAS_PROTECTION"] = np.where(
    (df["ONLINESECURITY"] == "Yes") & (df["TECHSUPPORT"] == "Yes"), 1, 0
)
df["NEW_STREAMING_USER"] = np.where(
    (df["STREAMINGTV"] == "Yes") | (df["STREAMINGMOVIES"] == "Yes"), 1, 0
)
df.loc[(df["PARTNER"] == "No") & (df["DEPENDENTS"] == "No"), "NEW_FAMILY_STATUS"] = "Single"
df.loc[(df["PARTNER"] == "Yes") & (df["DEPENDENTS"] == "No"), "NEW_FAMILY_STATUS"] = "Couple"
df.loc[(df["PARTNER"] == "Yes") & (df["DEPENDENTS"] == "Yes"), "NEW_FAMILY_STATUS"] = "Family"
df.loc[(df["PARTNER"] == "No") & (df["DEPENDENTS"] == "Yes"), "NEW_FAMILY_STATUS"] = "Single_Parent"
df["NEW_IS_ALONE"] = np.where(
    (df["PARTNER"] == "No") & (df["DEPENDENTS"] == "No"), 1, 0
)
df["NEW_IS_MONTH_TO_MONTH"] = np.where(
    df["CONTRACT"] == "Month-to-month", 1, 0
)
auto_payments = ["Bank transfer (automatic)", "Credit card (automatic)"]
df["NEW_AUTO_PAYMENT"] = np.where(
    df["PAYMENTMETHOD"].isin(auto_payments), 1, 0
)
df["NEW_HIGH_RISK_CUSTOMER"] = np.where(
    (df["CONTRACT"] == "Month-to-month")
    & (df["INTERNETSERVICE"] == "Fiber optic")
    & (df["NEW_TOTAL_SERVICES"] == 0),
    1,
    0,
)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

df.head()
df.drop("CUSTOMERID", axis=1, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def rare_analyser(dataframe, target, cat_cols):
    """Kategorik değişkenlerin alt kategorilerinin frekansını, oranını ve Target ortalamasını sunar."""
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "CHURN", cat_cols)

def rare_encoder(dataframe, rare_perc):
    """Belirli bir oranın (rare_perc) altında kalan nadir sınıfları 'Rare' adı altında birleştirir."""
    temp_df = dataframe.copy()

    rare_columns = [
        col for col in temp_df.columns
        if temp_df[col].dtypes == 'O'
        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)
    ]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

def label_encoder(dataframe, binary_col):
    """2 sınıflı kategorik değişkeni 0 ve 1 olarak dönüştürür."""
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """Kategorik değişkenler için One-Hot Encoding uygular."""
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

# 1. 2'den fazla sınıfa sahip kategorik sütunları seçelim:
ohe_cols = [col for col in cat_cols if 10 >= df[col].nunique() > 2]

# 2. Fonksiyonu çağırıp df'e eşitleyelim:
df = one_hot_encoder(df, ohe_cols, drop_first=True)


#Adım 4: Numerik değişkenler için standartlaştırma yapınız.

df.head()

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])

#rs = RobustScaler()
#df[num_cols] = rs.fit_transform(df[num_cols])

#mms = MinMaxScaler()
#df[num_cols] = mms.fit_transform(df[num_cols])

# -----------Görev 3 : Modelleme --------------------

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

y = df["CHURN"]
X = df.drop(["CHURN"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345,verbose=-1)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=10, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


########### LR ##########
#Accuracy: 0.8045
#Auc: 0.8473
#Recall: 0.5276
#Precision: 0.6667
#F1: 0.5885
########## KNN ##########
#Accuracy: 0.7646
#Auc: 0.781
#Recall: 0.5265
#Precision: 0.5607
#F1: 0.5428
########## CART ##########
#Accuracy: 0.7295
#Auc: 0.6617
#Recall: 0.5115
#Precision: 0.4906
#F1: 0.5007
########## RF ##########
#Accuracy: 0.796
#Auc: 0.8292
#Recall: 0.4971
#Precision: 0.6514
#F1: 0.5636
########## SVM ##########
#Accuracy: 0.7926
#Auc: 0.8158
#Recall: 0.4195
#Precision: 0.6754
#F1: 0.5172
########## XGB ##########
#Accuracy: 0.784
#Auc: 0.822
#Recall: 0.5115
#Precision: 0.6133
#F1: 0.5572
########## LightGBM ##########
#Accuracy: 0.7924
#Auc: 0.8376
#Recall: 0.5115
#Precision: 0.636
#F1: 0.5667
########## CatBoost ##########
#Accuracy: 0.8004
#Auc: 0.8415
#Recall: 0.5153
#Precision: 0.6585
#F1: 0.578
