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
    drop_columns,
    scale_numeric_train, 
    scale_numeric_test
)

print("Loading the data")

raw_df = load_data("data/bank-additional-full.csv")