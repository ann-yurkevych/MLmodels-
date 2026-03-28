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

def numeric_categorical_features(df: pd.DataFrame, target: str = 'y'):
    # returns a list of categorical + numeric features in two different lists
    df_features = df.drop(columns=[target])
    return df_features.select_dtypes(include=["number"]).columns.tolist(), df_features.select_dtypes(include=["object"]).columns.tolist()

def unknown_values_replace(df: pd.DataFrame, categories_to_keep: list, imputer=None):
    """
    categories_to_keep will be "unknown categories" which based on EDA were decided to be kept.
    """

    df = df.copy()
    cols_to_impute = [col for col in df.select_dtypes(include='object').columns
                      if col not in categories_to_keep]
    
    df[cols_to_impute] = df[cols_to_impute].replace('unknown', np.nan)
    
    if imputer is None:
        # Training set — fit and transform
        imputer = SimpleImputer(strategy='most_frequent')
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
        return df, imputer
    else:
        # Validation/test set — transform only
        df[cols_to_impute] = imputer.transform(df[cols_to_impute])
        return df, imputer
    
def binarize_feature(X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame, feature: str):

    X_train = X_train.copy()
    X_validation = X_validation.copy()
    X_test = X_test.copy()
    
    X_train[feature] = (X_train[feature] >= 1).astype(int)
    X_validation[feature] = (X_validation[feature] >= 1).astype(int)
    X_test[feature] = (X_test[feature] >= 1).astype(int)
    
    return X_train, X_validation, X_test


# functions to handle class imbalance: SMOTENC will be used in the pipeline, but indices of  categorical features are needed to get
def get_categorical_indices(X, numeric_columns):
    columns_list = X.columns.to_list()
    
    categorical_indices = [
        idx for idx, col in enumerate(columns_list)
        if col not in numeric_columns
    ]
    
    return categorical_indices


