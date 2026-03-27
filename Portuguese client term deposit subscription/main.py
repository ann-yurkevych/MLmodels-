import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib.pyplot as plt
from preprocessing import (
    load_data,
    extract_target,
    split_train_test,
    encode_target, 
    numeric_categorical_features,
    drop_columns
)

print("Loading the data")

raw_df = load_data("data/bank-additional-full.csv")

# encode target variable from str to num
raw_df = encode_target(raw_df, 'y') 

# drop features: based on conducted EDA in exporation.ipynb 
raw_df = drop_columns() # add dropped columns

# split dataset into train/test: 80/20
X_train, X_test = split_train_test(raw_df, 'y')

# split train dataset into train/validation: 60/20
X_train, X_validation = split_train_test(X_train, 'y')


# extract target variable from each set: training, validation, test: y_train, y_validation, y_test
X_train, y_train = extract_target(X_train, 'y') 
X_validation, y_validation = extract_target(X_validation, 'y') 
X_test, y_test = extract_target(X_test, 'y') 


# determine numeric, categorical features
numeric_features, categorical_features = numeric_categorical_features(raw_df)

# FunctionTransformer is used if you want to use custom functions in pipeline
# to pass arguments to a function use kw_args = kw_args={'replace_this': 'unknown', 'with_this': 0})

# def replace_value(X, replace_this='unknown', with_this=np.nan):
#     return pd.DataFrame(X).replace(replace_this, with_this)

# FunctionTransformer(replace_value, kw_args={'replace_this': 'unknown', 'with_this': 0})

# ColumnTransformer
#  hyperparameter tuning across the whle pipeline




preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]), numeric_features),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), # custom imputation will be used
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
    
], remainder="drop")

