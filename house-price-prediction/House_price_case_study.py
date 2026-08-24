import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler , MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



# Görev 1: Keşifçi Veri Analizi (EDA)

# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.

# 1. Dosyaları ayrı ayrı oku

df_1 = pd.read_csv("datasets/test.csv")
df_2 = pd.read_csv("datasets/train.csv")

# 2. İki dataframe'i satır bazında birleştir

df_ = pd.concat([df_1, df_2], ignore_index=True)
df =df_.copy()


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


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.


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

# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["MSSUBCLASS"] = df["MSSUBCLASS"].astype("object")

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in ["ID", "SALEPRICE"]]


# Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(dropna=False),
                        "Ratio": 100 * dataframe[col_name].value_counts(dropna=False) / len(dataframe)}))
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


# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "SALEPRICE", col)

saleprice_corr = df[num_cols + ["SALEPRICE"]].corr()["SALEPRICE"].sort_values(ascending=False)

print(saleprice_corr)



# Adım 6: Aykırı gözlem var mı inceleyiniz.


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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


# Adım 7: Eksik gözlem var mı inceleyiniz


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


# Görev 2: Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# 1. İstisna olanlar DIŞINDAKİ tüm kategorik değişkenleri "None" ile doldurma
for col in cat_cols:
    if col not in ["MSZONING", "ELECTRICAL"]:
        df[col] = df[col].fillna("None")

# 2. İstisna olanları MOD (en çok tekrar eden değer) ile doldurma
for col in ["MSZONING", "ELECTRICAL"]:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

zero_cols = ["BSMTFINSF1", "BSMTFINSF2", "BSMTUNFSF", "TOTALBSMTSF", "GARAGEAREA", "MASVNRAREA"]

for col in zero_cols:
    df[col] = df[col].fillna(0)

# 1. Garaj yılı boş olanlara evin inşa yılını atama
df["GARAGEYRBLT"] = df["GARAGEYRBLT"].fillna(df["YEARBUILT"])

df["LOTFRONTAGE"] = df.groupby("NEIGHBORHOOD")["LOTFRONTAGE"].transform(lambda x: x.fillna(x.median()))



def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """Sayısal değişken için IQR yöntemine göre alt ve üst eşik değerleri hesaplar."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """Aykırı değerleri silmek yerine belirlenen eşik değerlerle baskılar (Capping)."""
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "SALEPRICE":
        if check_outlier(df, col):
            replace_with_thresholds(df, col, q1=0.05, q3=0.95)


for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 3: Yeni değişkenler oluşturunuz. (ufak bi değişiklik yapalım slayta göre gitmedim burada)

df.columns

# 1. Bahçe alanı
df["NEW_GARDEN_AREA"] = df["LOTAREA"] - df["1STFLRSF"]

# Garajı müstakil olanları düş
mask = df["GARAGETYPE"] == "Detchd"
df.loc[mask, "NEW_GARDEN_AREA"] = df.loc[mask, "NEW_GARDEN_AREA"] - df.loc[mask, "GARAGEAREA"]
df["NEW_GARDEN_AREA"] = df["NEW_GARDEN_AREA"].clip(lower=0)

# 2. Sundurma ve Bahçe Oranları (+1 sıfıra bölünmeyi engeller)
df["NEW_OPENPORCHSF_NEW_GARDEN_AREA_RATIO"] = df["OPENPORCHSF"] / (df["NEW_GARDEN_AREA"] + 1)
df["NEW_ENCLOSEDPORCH_NEW_GARDEN_AREA_RATIO"] = df["ENCLOSEDPORCH"] / (df["NEW_GARDEN_AREA"] + 1)

df["NEW_ALLPORCH"] = df["ENCLOSEDPORCH"] + df["OPENPORCHSF"]
df["NEW_ALLPORCH_NEW_GARDEN_AREA_RATIO"] = df["NEW_ALLPORCH"] / (df["NEW_GARDEN_AREA"] + 1)

df["NEW_GARDEN_AREA_LOTAREA_RATIO"] = df["NEW_GARDEN_AREA"] / df["LOTAREA"]
df["NEW_POOLAREA_NEW_GARDEN_AREA_RATIO"] = df["POOLAREA"] / (df["NEW_GARDEN_AREA"] + 1)

# 3. Toplam Sosyal Alan ve Oranları (Düzeltilen Kısım)
df["NEW_ALL_SOCIAL_AREA"] = df["POOLAREA"] + df["ENCLOSEDPORCH"] + df["OPENPORCHSF"] + df["WOODDECKSF"]

# İki ayrı isimle iki ayrı oran alıyoruz:
df["NEW_ALL_SOCIAL_AREA_GARDEN_RATIO"] = df["NEW_ALL_SOCIAL_AREA"] / (df["NEW_GARDEN_AREA"] + 1)
df["NEW_ALL_SOCIAL_AREA_LOTAREA_RATIO"] = df["NEW_ALL_SOCIAL_AREA"] / df["LOTAREA"]

# 4. Yaş ve Segmentler
df["NEW_HOUSE_AGE"] = df["YRSOLD"] - df["YEARBUILT"]

df.loc[(df["NEW_HOUSE_AGE"] >= 0) & (df["NEW_HOUSE_AGE"] <= 5), "NEW_HOUSE_SEGMENT"] = "NEW"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10), "NEW_HOUSE_SEGMENT"] = "NORMAL"
df.loc[(df["NEW_HOUSE_AGE"] >= 11), "NEW_HOUSE_SEGMENT"] = "OLD"

# ==========================================
# 5. YÜKSEK / ORTA / DÜŞÜK KALİTELİ EV RİSK MATRİSİ
# ==========================================

# 1. Yüksek Kalite (OVERALLQUAL >= 7)
df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "PERFECT"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "VERY_GOOD"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "GOOD_CONDITION"

df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "VERY_GOOD"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "GOOD_CONDITION"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "MODERATE_RISK"

df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] <= 3), "NEW_HOUSE_RISK_CONTROL"] = "NEED_REPAIR"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] <= 3), "NEW_HOUSE_RISK_CONTROL"] = "NEED_REPAIR"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] >= 7) & (df["OVERALLCOND"] <= 3), "NEW_HOUSE_RISK_CONTROL"] = "HIGH_REPAIR_COST"

# 2. Orta Kalite (OVERALLQUAL 4-6)
df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] >= 4) & (df["OVERALLQUAL"] <= 6) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "GOOD_CONDITION"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] >= 4) & (df["OVERALLQUAL"] <= 6) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "GOOD_CONDITION"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] >= 4) & (df["OVERALLQUAL"] <= 6) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "MODERATE_RISK"

df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] >= 4) & (df["OVERALLQUAL"] <= 6) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "STANDART"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] >= 4) & (df["OVERALLQUAL"] <= 6) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "MODERATE_RISK"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] >= 4) & (df["OVERALLQUAL"] <= 6) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "NEED_REPAIR"

# 3. Düşük Kalite (OVERALLQUAL <= 3)
df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "MODERATE_RISK"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "NEED_REPAIR"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] >= 7), "NEW_HOUSE_RISK_CONTROL"] = "NEED_REPAIR"

df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "NEED_REPAIR"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "HIGH_RISK"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] >= 4) & (df["OVERALLCOND"] <= 6), "NEW_HOUSE_RISK_CONTROL"] = "HIGH_RISK"

df.loc[(df["NEW_HOUSE_AGE"] <= 5) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] <= 3), "NEW_HOUSE_RISK_CONTROL"] = "HIGH_RISK"
df.loc[(df["NEW_HOUSE_AGE"] >= 6) & (df["NEW_HOUSE_AGE"] <= 10) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] <= 3), "NEW_HOUSE_RISK_CONTROL"] = "HIGH_RISK"
df.loc[(df["NEW_HOUSE_AGE"] >= 11) & (df["OVERALLQUAL"] <= 3) & (df["OVERALLCOND"] <= 3), "NEW_HOUSE_RISK_CONTROL"] = "CRITICAL_RISK"

# Kontrol
df["NEW_HOUSE_RISK_CONTROL"].value_counts(dropna=False)

# Bodrum kullanılabilir/işlenmiş alan oranının hesaplanması
df["NEW_BSMT_FIN_RATIO"] = (df["BSMTFINSF1"] + df["BSMTFINSF2"]) / (df["TOTALBSMTSF"] + 1)

# Bodrumu hiç olmayan (TOTALBSMTSF == 0) evlerde oranı doğrudan 0'a sabitliyoruz
df.loc[df["TOTALBSMTSF"] == 0, "NEW_BSMT_FIN_RATIO"] = 0

# Satış anında tadilatın üzerinden geçen yıl sayısı
df["NEW_YEARS_SINCE_REMOD"] = df["YRSOLD"] - df["YEARREMODADD"]

# Tadilat geçmişine göre segmentasyon
df.loc[df["NEW_YEARS_SINCE_REMOD"] <= 1, "NEW_REMOD_SEGMENT"] = "RECENT"
df.loc[(df["NEW_YEARS_SINCE_REMOD"] > 1) & (df["NEW_YEARS_SINCE_REMOD"] <= 5), "NEW_REMOD_SEGMENT"] = "MID_RECENT"
df.loc[(df["NEW_YEARS_SINCE_REMOD"] > 5) & (df["NEW_YEARS_SINCE_REMOD"] <= 10), "NEW_REMOD_SEGMENT"] = "OLD_REMOD"
df.loc[df["NEW_YEARS_SINCE_REMOD"] > 10, "NEW_REMOD_SEGMENT"] = "VERY_OLD_REMOD"

# 11. Zemin üstü yatak odası / Zemin üstü toplam oda oranı
df["NEW_BEDROOM_RATIO"] = df["BEDROOMABVGR"] / (df["TOTRMSABVGRD"] + 1)

# 12. Zemin üstü toplam banyo/tuvalet sayısı (Yarım banyolar 0.5 ile çarpılır)
df["NEW_TOTAL_BATH_ABVGR"] = df["FULLBATH"] + (df["HALFBATH"] * 0.5)

# 13. Zemin üstü banyo / Zemin üstü toplam oda oranı
df["NEW_BATH_RATIO_ABVGR"] = df["NEW_TOTAL_BATH_ABVGR"] / (df["TOTRMSABVGRD"] + 1)

# 15. Mutfak sayısının toplam odaya oranı
df["NEW_KITCHEN_RATIO"] = df["KITCHENABVGR"] / (df["TOTRMSABVGRD"] + 1)

# 16. Şömine sayısının toplam odaya oranı
df["NEW_FIREPLACE_RATIO"] = df["FIREPLACES"] / (df["TOTRMSABVGRD"] + 1)



cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Adım 2: Rare Encoder uygulayınız

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))

        # İndeksleri oluştuğu anda str yapıyoruz ki pd.DataFrame hizalarken tipler karışmasın
        counts = dataframe[col].value_counts()
        counts.index = counts.index.astype(str)

        ratios = dataframe[col].value_counts(normalize=True)
        ratios.index = ratios.index.astype(str)

        target_means = dataframe.groupby(col, observed=False)[target].mean()
        target_means.index = target_means.index.astype(str)

        # Artık sorunsuz birleştirir
        res = pd.concat([counts, ratios, target_means], axis=1)
        res.columns = ["COUNT", "RATIO", "TARGET_MEAN"]

        print(res, end="\n\n\n")

rare_analyser(df, "SALEPRICE", cat_cols)


def rare_encoder(dataframe, rare_perc, cat_cols, protected_labels=["None"]):
    temp_df = dataframe.copy()

    # Sadece belirlenen kategorik sütunlarda işlem yapıyoruz
    rare_columns = [col for col in cat_cols if (temp_df[col].value_counts(normalize=True) < rare_perc).any()]

    for var in rare_columns:
        tmp = temp_df[var].value_counts(normalize=True)

        # Rare oranının altında kalan etiketleri buluyoruz
        rare_labels = tmp[tmp < rare_perc].index.tolist()

        # ⚠️ KRİTİK NOKTA: Korunması gereken etiketleri ("None", "No" vb.) rare_labels listesinden çıkarıyoruz!
        rare_labels = [label for label in rare_labels if str(label) not in protected_labels]

        # Eğer geriye dönüştürülecek etiketi kaldıysa birleştirme yapıyoruz
        if len(rare_labels) > 0:
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01, cat_cols, protected_labels=["None"])


# Adım 4: Encoding işlemlerini gerçekleştiriniz


# 1. Label Encoder Fonksiyonu
def label_encoder(dataframe, binary_col):
    """2 sınıflı kategorik değişkeni 0 ve 1 olarak dönüştürür."""
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col].astype(str))
    return dataframe

# Binary sütunları seçip DÖNGÜYÜ TAMAMLIYORUZ:
binary_cols = [col for col in cat_cols if df[col].nunique(dropna=False) == 2]

for col in binary_cols:
    label_encoder(df, col)


# 2. One-Hot Encoder Fonksiyonu
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """Kategorik değişkenler için One-Hot Encoding uygular."""
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

# 1. 2'den fazla sınıfa sahip tüm kategorik VE kardinal sütunları seçiyoruz:
# (cat_cols + cat_but_car birleşimi sayesinde NEIGHBORHOOD vb. kaçamaz)
ohe_cols = [col for col in (cat_cols + cat_but_car) if df[col].nunique(dropna=False) > 2]

# 2. OHE uyguluyoruz:
df = one_hot_encoder(df, ohe_cols, drop_first=True)

# 3. Kontrol:
df.head()


# Görev 3

# Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)

# ==============================================================================
# 1. TRAIN / TEST AYRIMI VE LOG DÖNÜŞÜMÜ
# ==============================================================================

# SALEPRICE dolu olanlar Train, boş olanlar Test verisidir
train_df = df[df['SALEPRICE'].notna()].copy()
test_df = df[df['SALEPRICE'].isna()].copy()

X = train_df.drop(["SALEPRICE", "ID"], axis=1, errors="ignore")
y = train_df["SALEPRICE"]

# Log Dönüşümü (RMSE performansını ciddi oranda artırır)
y_log = np.log1p(y)

test_id = test_df["ID"]
X_test = test_df.drop(["SALEPRICE", "ID"], axis=1, errors="ignore")


# ==============================================================================
# 2. RANDOM FORESTS REGRESSOR
# ==============================================================================


rf_model = RandomForestRegressor(random_state=17)


# Baz Model RMSE (Log ölçeğinde)
cv_results = cross_val_score(rf_model, X, y_log, cv=5, scoring="neg_mean_squared_error")
print("RF Base Log RMSE:", np.mean(np.sqrt(-cv_results)))

#rf_params = {"max_depth": [5, 8, None],
#             "max_features": [3, 5, 7, "sqrt"],
#             "min_samples_split": [2, 5, 8],
#             "n_estimators": [100, 200, 500]}

#rf_params = {"max_depth": [None],
#             "max_features": ["sqrt"],
#             "min_samples_split": [2,3,4],
#             "n_estimators": [150,175, 200, 225,250]}


#rf_params = {"max_depth": [None],
#             "max_features": ["sqrt"],
#             "min_samples_split": [2],
#             "n_estimators": [230,250,275,300]}


#rf_params = {"max_depth": [None],
#             "max_features": ["sqrt"],
#             "min_samples_split": [2],
#             "n_estimators": [240,250,260,3000]}

rf_params = {"max_depth": [None],
             "max_features": ["sqrt"],
             "min_samples_split": [2],
             "n_estimators": [235,240,245,5000,7000,10000]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=False).fit(X, y_log)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y_log)

cv_results = cross_val_score(rf_final, X, y_log, cv=5, scoring="neg_mean_squared_error")
print("RF Final Log RMSE:", np.mean(np.sqrt(-cv_results)))

# RF Base Log RMSE: 0.1476979348738928
# RF Final Log RMSE: 0.14677900943145916
# RF Final Log RMSE: 0.14666992023402772
# RF Final Log RMSE: 0.1466090696253523


# Değişken Önem Düzeyi (Feature Importance)
def plot_importance(model, features, num=20):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f'Features Importance ({type(model).__name__})')
    plt.tight_layout()
    plt.show()

plot_importance(rf_final, X)


# ==============================================================================
# 3. GBM (Gradient Boosting Regressor)
# ==============================================================================

gbm_model = GradientBoostingRegressor(random_state=17)

# gbm_params = {"learning_rate": [0.01, 0.1],
#               "max_depth": [3, 5, 8],
#               "n_estimators": [200, 500],
#               "subsample": [0.7, 1]}

# gbm_params = {"learning_rate": [0.05, 0.1],
#               "max_depth": [2,3,4],
#               "n_estimators": [100,150,200,250,300],
#               "subsample": [0.8, 1]}

# gbm_params = {"learning_rate": [0.08, 0.1],
#               "max_depth": [2],
#               "n_estimators": [275,300,325 ,1000,5000,10000],
#               "subsample": [0.8, 1]}

# gbm_params = {"learning_rate": [0.08 , 0.09],
#               "max_depth": [2],
#               "n_estimators": [800,1000,2000],
#               "subsample": [0.8, 0.9]}

gbm_params = {"learning_rate": [0.08],
              "max_depth": [2],
              "n_estimators": [750,800,850,10000],
              "subsample": [0.8]}


gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=False).fit(X, y_log)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y_log)

cv_results = cross_val_score(gbm_final, X, y_log, cv=5, scoring="neg_mean_squared_error")
print("GBM Final Log RMSE:", np.mean(np.sqrt(-cv_results)))

# GBM Final Log RMSE: 0.12922912147679275
# GBM Final Log RMSE: 0.12838625889653735
# GBM Final Log RMSE: 0.12784250950719317
# GBM Final Log RMSE: 0.1278362345689136
# GBM Final Log RMSE: 0.1278144130722479

plot_importance(gbm_final, X)


# ==============================================================================
# 4. XGBOOST REGRESSOR
# ==============================================================================

xgboost_model = XGBRegressor(random_state=17)

# xgboost_params = {"learning_rate": [0.01, 0.1],
#                   "max_depth": [5, 8],
#                   "n_estimators": [200, 500],
#                   "colsample_bytree": [0.7, 1]}

# xgboost_params = {"learning_rate": [0.05, 0.1],
#                   "max_depth": [4,5,6,7],
#                   "n_estimators": [400, 500],
#                   "colsample_bytree": [0.6,0.7, 0.8]}

# xgboost_params = {"learning_rate": [0.08, 0.1],
#                   "max_depth": [2,3,4,],
#                   "n_estimators": [350,400, 450,1000],
#                   "colsample_bytree": [0.5,0.6]}

# xgboost_params = {"learning_rate": [0.09, 0.1],
#                   "max_depth": [4],
#                   "n_estimators": [325,350,375,2000],
#                   "colsample_bytree": [0.6]}

xgboost_params = {"learning_rate": [0.1],
                  "max_depth": [4],
                  "n_estimators": [300,325,5000],
                  "colsample_bytree": [0.6]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, y_log)

xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y_log)

cv_results = cross_val_score(xgboost_final, X, y_log, cv=5, scoring="neg_mean_squared_error")
print("XGBoost Final Log RMSE:", np.mean(np.sqrt(-cv_results)))

# XGBoost Final Log RMSE: 0.1327847198019178
# XGBoost Final Log RMSE: 0.12774811971317163
# XGBoost Final Log RMSE: 0.12770895239383145
# XGBoost Final Log RMSE: 0.12759515082515274
# XGBoost Final Log RMSE: 0.12759515082515274

plot_importance(xgboost_final, X)


# ==============================================================================
# 5. LIGHTGBM REGRESSOR
# ==============================================================================

lgbm_model = LGBMRegressor(random_state=17, verbose=-1)

# lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
#                "n_estimators": [300, 500, 1000],
#                "colsample_bytree": [0.5, 0.7, 1]}

# lgbm_params = {"learning_rate": [0.09, 0.1],
#                "n_estimators": [800, 1000,2000],
#                "colsample_bytree": [0.4,0.5, 0.6]}

# lgbm_params = {"learning_rate": [0.09],
#                "n_estimators": [1500,2000,5000],
#                "colsample_bytree": [0.3,0.4]}

lgbm_params = {"learning_rate": [0.09],
               "n_estimators": [5000,7500,10000],
               "colsample_bytree": [0.4]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=False).fit(X, y_log)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y_log)

cv_results = cross_val_score(lgbm_final, X, y_log, cv=5, scoring="neg_mean_squared_error")
print("LightGBM Final Log RMSE:", np.mean(np.sqrt(-cv_results)))

# LightGBM Final Log RMSE: 0.12925102627626006
# LightGBM Final Log RMSE: 0.1288574564549875
# LightGBM Final Log RMSE: 0.12885151284007462
# LightGBM Final Log RMSE: 0.12885105108864475

plot_importance(lgbm_final, X)


# ==============================================================================
# 6. CATBOOST REGRESSOR
# ==============================================================================

catboost_model = CatBoostRegressor(random_state=17, verbose=False)

# catboost_params = {"iterations": [300, 500],
#                    "learning_rate": [0.01, 0.1],
#                    "depth": [3, 6]}

# catboost_params = {"iterations": [400, 500,600],
#                    "learning_rate": [0.05, 0.1],
#                    "depth": [4,5, 6,7]}

# catboost_params = {"iterations": [550,600,1000],
#                    "learning_rate": [0.3,0.05,0.7],
#                    "depth": [2,3,6]}

catboost_params = {"iterations": [1000,2000,5000],
                   "learning_rate": [0.05],
                   "depth": [6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, y_log)

catboost_best_grid.best_params_

catboost_final = catboost_best_grid.best_estimator_

cv_results = cross_val_score(catboost_final, X, y_log, cv=5, scoring="neg_mean_squared_error")
print("CatBoost Final Log RMSE:", np.mean(np.sqrt(-cv_results)))

# CatBoost Final Log RMSE: 0.12697833958709395
# CatBoost Final Log RMSE: 0.1252373644111723
# CatBoost Final Log RMSE: 0.12468316616970285

plot_importance(catboost_final, X)



# SUBMISSION DOSYASI OLUŞTURMA

# 1. CatBoost modelinle test verisi üzerinde tahmin al
test_preds_log = catboost_final.predict(X_test)

# 2. Log1p dönüşümünü tersine çevirip gerçek fiyatlara dön (expm1)
test_preds = np.expm1(test_preds_log)

# 3. Id ve SalePrice kolonlarından oluşan DataFrame'i oluştur
submission_df = pd.DataFrame({
    "Id": test_id.astype(int),
    "SalePrice": test_preds
})

# 4. Dosyayı kaydet
submission_df.to_csv("submission_catboost.csv", index=False)
print("✅ Dosya başarıyla kaydedildi!")

# 5. Dosyanın bilgisayarında tam olarak NEREURDE olduğunu görmek için:
import os
print("Dosyanın Tam Adresi:", os.path.abspath("submission_catboost.csv"))