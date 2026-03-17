import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine 

def load_data(datasets_path: str, file_type: str = "csv", sql_query: str = None, connection: str = None, sep: str = None):
    """
    Reads the following formats: csv, json, xlsx, sql.
    Args:
        datasets_path: Path to the file or database table name.
        file_type: File format - 'csv', 'json', 'xlsx', or 'sql'.
        sql_query (str): SQL query string (required for SQL type).
        connection: Database connection string (required for SQL type).
        sep: Delimiter to use (csv only).
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    if file_type == "csv":
        df = pd.read_csv(datasets_path, sep=sep)
    elif file_type == "json":
        df = pd.read_json(datasets_path)         
    elif file_type == "xlsx":
        df = pd.read_excel(datasets_path)         
    elif file_type == "sql":
        if connection is None:
            raise ValueError("A database connection string 'connection' is required for SQL type.")
        if sql_query is None:
            raise ValueError("A 'sql_query' is required for SQL type.")
        engine = create_engine(connection)
        with engine.connect() as conn:          
            df = pd.read_sql(sql_query, connection=conn)
    else:
        raise ValueError(f"Unsupported file type: '{file_type}'. Choose from: csv, json, xlsx, sql.")
    
    print(df.head())
    return df



def extract_target(df: pd.DataFrame, target: str):
  input_features = df.drop(columns=[target]).copy()
  target = df[target].copy()
  return input_features, target

def drop_columns(df: pd.DataFrame, columns_to_drop: list):
  return df.drop(columns=columns_to_drop, errors="ignore")

# split functions


def impute_missing_values(df: pd.DataFrame):

  df_copy = df.copy()
  numeric_features = df_copy.select_dtypes(include=["number"]).columns.tolist()
  categorical_features = df_copy.select_dtypes(include=["object"]).columns.tolist()

  numeric_imputer = SimpleImputer(strategy="median")
  categorical_imputer = SimpleImputer(strategy="most_frequent")
  if numeric_features:
    df_copy[numeric_features] = numeric_imputer.fit_transform(df_copy[numeric_features])
  if categorical_features:
    df_copy[categorical_features] = categorical_imputer.fit_transform(df_copy[categorical_features])

  return df_copy, numeric_features, categorical_features, numeric_imputer, categorical_imputer

def impute_missing_values_transform(df_test: pd.DataFrame, numeric_features: list, categorical_features: list, numeric_imputer: SimpleImputer, categorical_imputer: SimpleImputer):

  df_test_copy = df_test.copy()

  for column in numeric_features:
    if column not in df_test_copy.columns:
            df_test_copy[column] = np.nan

  for column in categorical_features:
    if column not in df_test_copy.columns:
            df_test_copy[column] = np.nan

  if numeric_features:
        df_test_copy[numeric_features] = numeric_imputer.transform(df_test_copy[numeric_features])

  if categorical_features:
        df_test_copy[categorical_features] = categorical_imputer.transform(df_test_copy[categorical_features])

  return df_test_copy


def scale_numeric_train(X_train_encoded: pd.DataFrame):
  scaler = StandardScaler()
  X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)

  return X_train_scaled, scaler

def scale_numeric_test(X_test_encoded: pd.DataFrame, scaler: StandardScaler):
  X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)

  return X_test_scaled



   
