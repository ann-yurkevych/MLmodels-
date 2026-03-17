import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import os
import matplotlib.pyplot as plt
from preprocessing import (
    load_data,
    extract_target,
    split_train_test,
    encode_target
)

print("Loading the data")

raw_df = load_data("data/bank-additional-full.csv")

# encode target variable from str to num
raw_df = encode_target(raw_df, 'y') 

# split dataset into train/test: 80/20
X_train, X_test = split_train_test(raw_df, 'y')

# split train dataset into train/validation: 60/20
X_train, X_validation = split_train_test(X_train, 'y')

