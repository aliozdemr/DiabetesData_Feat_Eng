import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.simplefilter(action="ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 170)
pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x:"%.3f" % x)

df = pd.read_csv("diabetes_feature_eng/diabetes.csv")

def check_df(dataframe, head=5):
    print("######################SHAPE#######################")
    print(dataframe.shape)
    print("######################TYPES#######################")
    print(dataframe.dtypes)
    print("######################HEAD#######################")
    print(dataframe.head(head))
    print("######################TAIL#######################")
    print(dataframe.tail(head))
    print("######################NA#######################")
    print(dataframe.isnull().sum())
    print("######################QUANTILES#######################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# Glucose, BloodPressure, SkinThickness, Insulin, BMI değerlerinde 0 olarak girilmiş gözlemler var, bu değelerin 0 olması imkansız büyük ihtimal NaN değerleri 0 olarak atamışlar.
# Çıktıya baktığımızda Pregnancies, SkinThickness ve Insulin değişkenlerinde aykırı değer olma ihtimalı yüksek.
def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]

    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat

    num_cols = [col for col in num_cols if col not in num_but_cat ]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, num_but_cat, cat_but_car

cat_cols, num_cols, num_but_cat, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({ col_name : dataframe[col_name].value_counts(),
                         "Ratio": dataframe[col_name].value_counts() / len(dataframe) * 100}))
    print("##########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "Outcome",True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05,0.10 , 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print((dataframe[numerical_col].describe(quantiles)).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, True)

def target_summary_with_nums(dataframe, target, numerical_col):
    print("##################################################")
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))

for col in num_cols:
    target_summary_with_nums(df,"Outcome", col)

df.corr()

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=47).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred,y_test),2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test),2)}")
print(f"f1: {round(f1_score(y_pred,y_test),2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test),2)}")

zero_cols = [col for col in df.columns if df[col].min() == 0 and col not in ["Outcome", "Pregnancies"]]

dff = df.copy()

for i in zero_cols:
    #df[i] = np.where(df[i] == 0, np.nan, df[i])
    dff[i] = dff[i].apply(lambda x: np.nan if x == 0 else x)

def missing_value_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys = ["n_miss, ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_value_table(dff,True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        #temp_df[col + "_NA_Flag"] = np.where(temp_df[col].isnull(),1,0)
        temp_df[col + "_NA_Flag"] = temp_df[col].apply(lambda x: 1 if np.isnan(x) else 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    # na_flagss = [col for col in temp_df.columns if "_NA_" in col]
    for col in na_flags:
        print(pd.DataFrame({"Target_Mean": temp_df.groupby(col)[target].mean(), "Count":temp_df.groupby(col)[target].count()}))
        print("#########################################################")

missing_vs_target(dff,"Outcome", na_columns)

for col in zero_cols:
    dff.loc[dff[col].isnull(), col] = dff[col].median()

dff.isnull().sum()

def outlier_thresholds(dataframe, col_name,q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquartile_range)
    low_limit = quartile1 - (1.5 * interquartile_range)
    return up_limit, low_limit

def check_outlier(dataframe, col_name):
    up_limit, low_limit = outlier_thresholds(dataframe,col_name)

    res = True if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None) else False
    return res

def isoutlier(dataframe):
    for i in dataframe.columns:
        if  check_outlier(dataframe,i):
            return True
        else:
            return False
def replace_with_thresholds(dataframe, col, q1=0.05, q3=0.95):
    up_limit, low_limit = outlier_thresholds(dataframe,col)
    dataframe.loc[dataframe[col] < low_limit, col] = low_limit
    dataframe.loc[dataframe[col] > up_limit, col] = up_limit

for col in dff.columns:
    if check_outlier(dff,col):
        replace_with_thresholds(dff,col)
check_outlier(dff,"Insulin")
isoutlier(dff)

dff.loc[(dff["Age"] >= 21 ) & (dff["Age"] < 50), "New_Age_Cat"] = "mature"
dff.loc[(dff["Age"] >= 50 ), "New_Age_Cat"] = "senior"

dff["New_BMI"] = pd.cut(x = dff["BMI"], bins=[0,18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])
dff["New_Glucose"] = pd.cut(x = dff["Glucose"], bins=[0,140, 200,300], labels=["Normal", "Prediabetes",  "Diabetes"])
dff.loc[((dff["BMI"] < 18.5) & ((dff["Age"] >=21 )& (dff["Age"] <= 50))), "New_Age_BMI_Nom"] = "Underweigtmature"
dff.loc[(dff["BMI"] < 18.5) & (dff["Age"] > 50 ), "New_Age_BMI_Nom"] = "Underweigtsenior"
dff.loc[((dff["BMI"] > 18.5) & (dff["BMI"] < 25)) & (((dff["Age"] >=21) & (dff["Age"] <= 50))), "New_Age_BMI_Nom"] = "Healthymature"
dff.loc[((dff["BMI"] > 18.5) & (dff["BMI"] < 25)) & (dff["Age"] > 50 ), "New_Age_BMI_Nom"] = "Healthysenior"
dff.loc[((dff["BMI"] >= 25) & (dff["BMI"] < 30)) & (((dff["Age"] >=21) & (dff["Age"] <= 50))), "New_Age_BMI_Nom"] = "Overweigtmature"
dff.loc[((dff["BMI"] >= 25) & (dff["BMI"] < 30)) & ((dff["Age"] > 50 )), "New_Age_BMI_Nom"] = "Overweigtsenior"
dff.loc[(dff["BMI"] >= 30) & (((dff["Age"] >=21) & (dff["Age"] <= 50))), "New_Age_BMI_Nom"] = "Obesemature"
dff.loc[(dff["BMI"] >= 30) & (dff["Age"] > 50 ), "New_Age_BMI_Nom"] = "Underweigtsenior"
dff.head()
dff.isnull().sum()
# dff[dff["New_Age_BMI_Nom"].isnull()] Bazı null değerler tespit edilip yukardaki eşitsizliklerde düzeltmeler yapıldı

dff.loc[(dff["Glucose"] < 70 ) & ((dff["Age"] >=21) & (dff["Age"] < 50)), "New_Age_Glucose_Nom"] = "lowmature"
dff.loc[(dff["Glucose"] < 70 ) & ((dff["Age"] >=50)), "New_Age_Glucose_Nom"] = "lowsenior"
dff.loc[((dff["Glucose"] >= 70 ) & (dff["Glucose"] <100)) & ((dff["Age"] >=21) & (dff["Age"] < 50)), "New_Age_Glucose_Nom"] = "normalmature"
dff.loc[((dff["Glucose"] >= 70 ) & (dff["Glucose"] <100)) & ((dff["Age"] >= 50)), "New_Age_Glucose_Nom"] = "normalsenior"
dff.loc[((dff["Glucose"] >= 100 ) & (dff["Glucose"] <125)) & ((dff["Age"] >=21) & (dff["Age"] < 50)), "New_Age_Glucose_Nom"] = "hiddenmature"
dff.loc[((dff["Glucose"] >= 100 ) & (dff["Glucose"] <125)) & ((dff["Age"] >= 50)), "New_Age_Glucose_Nom"] = "hiddensenior"
dff.loc[((dff["Glucose"] >= 125 )) & ((dff["Age"] >=21) & (dff["Age"] < 50)), "New_Age_Glucose_Nom"] = "highmature"
dff.loc[((dff["Glucose"] >= 125 )) & ((dff["Age"] >=50)), "New_Age_Glucose_Nom"] = "highmature"
dff.shape
dff.isnull().sum()

dff["New_Insulin_Score"] = dff["Insulin"].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal" )
dff[["Insulin", "New_Insulin_Score"]].head(20)

dff["New_Glucose*Insulin"] = dff["Glucose"] * dff["Insulin"]
dff["New_Glucose*Pregnancies"] = dff["Glucose"] * dff["Pregnancies"]
dff.head()

cat_cols, num_cols, num_but_cat, cat_but_car = grab_col_names(dff)

binary_cols = [col for col in dff.columns if dff[col].dtypes == "O" and dff[col].nunique() == 2]

labelencoder = LabelEncoder()
for i in binary_cols:
    dff[i] = labelencoder.fit_transform(dff[i])
dff[["New_Age_Cat", "New_Insulin_Score"]]

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]
for i in cat_cols:
    dff = pd.get_dummies(dff, columns=[i], drop_first=True)
dff.shape
dff.head()

scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])
dff["New_Glucose*Insulin"]
dff.head()

y = dff["Outcome"]
X = dff.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=47).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred,y_test),2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test),2)}")
print(f"f1: {round(f1_score(y_pred,y_test),2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test),2)}")

"""
Accuracy: 0.8 --Artmış
Recall: 0.72 --Artmış
Precision: 0.69 --Artmış
f1: 0.7 --Azalmış
Auc: 0.78 --Artmış
"""

"""
Base Model
Accuracy: 0.76
Recall: 0.69
Precision: 0.57
f1: 0.62
Auc: 0.74
"""
X.shape
# rf_model.feature_importances_
def plot_importance(model, features, save = False):
    feature_imp = pd.DataFrame({"Value":model.feature_importances_, "Feature":features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10,20))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values("Value", ascending=False))
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importance.png")

plot_importance(rf_model,X)