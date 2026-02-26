import numpy as np
import pandas as pd
import opendatasets as od
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold
)
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text, plot_tree
from sklearn.metrics import roc_curve, auc
from typing import Any, Dict, List


def split_dataset(df: pd.DataFrame, target: str):
    """
    Splits a dataset into training and validation sets 
    using stratified sampling based on the target variable.

    Args:
        df (pd.DataFrame): Full dataset containing features and target.
        target (str): Target column name used for stratification.

    Returns:
        X_train (pd.DataFrame): Training subset.
        X_validation (pd.DataFrame): Validation subset.
    """
    
    X_train, X_validation = train_test_split(
        df,
        test_size=0.20,  # 80% train, 20% validation
        stratify=df[target],
        random_state=42
    )

    return X_train, X_validation

def extract_target(df: pd.DataFrame, target: str):
  """
    Separates input features and target variable from a dataset.

    Args:
    df : pd.DataFrame
        The full raw containing both feature columns and the target column.
    target : str
        Name of the target column to be extracted from the dataset.

    Return:
    input_features : A copy of the dataset without the target column (feature matrix X).
    target : A copy of the target column (label vector y).
    """
  input_features = df.drop(columns=[target]).copy()
  target = df[target].copy()
  return input_features, target

def drop_na_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    return df.dropna(subset=columns)

def impute_missing_values(data: Dict[str, Any], numeric_cols: list) -> None:
    """
    Impute missing numerical values using the mean strategy.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    imputer = SimpleImputer(strategy='mean').fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])

def drop_columns_splits(X_train: pd.DataFrame, X_validation: pd.DataFrame, columns_to_drop: list):
    """
    Drop specified columns from train, validation, and test splits.
    """

    X_train = X_train.drop(columns=columns_to_drop, errors="ignore")
    X_validation = X_validation.drop(columns=columns_to_drop, errors="ignore")

    return X_train, X_validation

def encode_splits_categories(X_train: pd.DataFrame, X_validation: pd.DataFrame):
    """
    Applies One-Hot Encoding to categorical features across train, validation,
    and test splits using a single encoder fitted on the training data only.

    Args:
    X_train : Training feature matrix used to fit the encoder.
    X_validation : Validation feature matrix to be transformed using the fitted encoder.
    

    Return:
    X_train : Transformed training set with numerical and encoded categorical features.
    X_validation : Transformed validation set with numerical and encoded categorical features.
    encoder : Fitted OneHotEncoder instance (can be reused for inference or saved).
    categorical_features : List of categorical column names that were encoded.
    """

    numerical_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # fit on X_train only
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_features])

    # transform
    def _transform(inputs: pd.DataFrame) -> pd.DataFrame:
        encoded = encoder.transform(inputs[categorical_features])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=inputs.index
        )
        return pd.concat([inputs[numerical_features], encoded_df], axis=1)

    X_train = _transform(X_train)
    X_validation = _transform(X_validation)

    return X_train, X_validation, encoder, categorical_features

def scale_numerical_cols(X_train: pd.DataFrame, X_validation: pd.DataFrame, scaler_numeric: bool = True):
    """
    Scales numerical columns using MinMaxScaler fitted on X_train only.
    Args:
        X_train (pd.DataFrame): Training features (used to fit scaler).
        X_validation (pd.DataFrame): Validation features to transform.
        scaler_numeric (bool): Whether to apply scaling.

    Return:
        Tuple:
            - X_train (pd.DataFrame)
            - X_validation (pd.DataFrame)
            - List of numerical columns (list)
            - Fitted MinMaxScaler object (or None if scaling disabled)
    """

    numerical_features = X_train.select_dtypes(include=["number"]).columns.tolist()

    if not scaler_numeric:
        return X_train, X_validation, numerical_features, None

    scaler = MinMaxScaler()
    scaler.fit(X_train[numerical_features])

    X_train[numerical_features] = scaler.transform(X_train[numerical_features])
    X_validation[numerical_features] = scaler.transform(X_validation[numerical_features])

    return X_train, X_validation, numerical_features, scaler



def compute_auroc_and_build_roc(model, inputs, targets, name=''):
    y_pred_proba = model.predict_proba(inputs)[:, 1]

    fpr, tpr, _ = roc_curve(targets, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f'AUROC for {name}: {roc_auc:.4f}')

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()

    return roc_auc



def preprocess_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.
    Args:
        raw_df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets for train and validation sets. 
       7:  X_train, y_train, X_validation, y_validation, input_features, scaler, encoder.
    """
    # divide dataset into train/validation
    X_train, X_validation = split_dataset(df, "Exited")
    

    # extract target variable to prevent data leakage
    X_train, y_train = extract_target(X_train, "Exited")
    X_validation, y_validation = extract_target(X_validation, "Exited")

    # drop categorical columns which don't serve any meaningful value
    drop_cols = ["id", "CustomerId", "Surname"]
    X_train, X_validation= drop_columns_splits(X_train, X_validation, drop_cols)

    # encode categorical features
    X_train, X_validation, encoder, categorical_features = encode_splits_categories(X_train, X_validation)

    # scale numerical features
    X_train, X_validation, numerical_features, scaler = scale_numerical_cols(X_train, X_validation)
    
    # get all the input features for the base model
    input_features = X_train.columns.to_list()

    return X_train, y_train, X_validation, y_validation, scaler, encoder, input_features


def preprocess_test_data(df_test: pd.DataFrame, scaler, encoder, input_features: List[str]) -> pd.DataFrame:
    """
    Preprocess test data using fitted encoder/scaler from training.

    Args:
        raw_df (pd.DataFrame): The test raw dataframe.

    Returns: X_test
    """
  
    X_test = df_test.copy()

    drop_cols = ["id", "CustomerId", "Surname"]
    X_test, _ = drop_columns_splits(X_test, X_test, drop_cols) 

    categorical_features = list(encoder.feature_names_in_) 

    for c in categorical_features:
        if c not in X_test.columns:
            X_test[c] = ""

    numerical_features = X_test.select_dtypes(include=["number"]).columns.tolist()

    encoded = encoder.transform(X_test[categorical_features])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_features),
        index=X_test.index
    )

    X_test = pd.concat([X_test[numerical_features], encoded_df], axis=1)
    numeric_cols = X_test.select_dtypes(include=["number"]).columns.tolist()
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    X_test = X_test.reindex(columns=input_features, fill_value=0)

    return X_test
