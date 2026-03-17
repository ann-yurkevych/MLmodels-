import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold
) 
from EDA import (
    decriptive_stats
)

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

# encode target variable before splitting
def encode_target(df: pd.DataFrame, target: str):
    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target])
    return df
    

def split_train_test(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
  """
    Splits a dataset into training and test sets
    using stratified sampling based on the target variable.

    Args:
        df (pd.DataFrame): Full dataset containing features and target.
        target (str): Target column name used for stratification.

    Returns:
        X_train (pd.DataFrame): Training subset.
        X_test (pd.DataFrame): Test subset.
    """
  X_train, X_test = train_test_split(
        df,
        test_size=test_size,  # 80% train, 20% validation
        stratify=df[target],
        random_state=random_state
    )

  return X_train, X_test

def extract_target(df: pd.DataFrame, target: str):
  input_features = df.drop(columns=[target]).copy()
  target = df[target].copy()
  return input_features, target

def drop_columns(df: pd.DataFrame, columns_to_drop: list):
  return df.drop(columns=columns_to_drop, errors="ignore")





   
