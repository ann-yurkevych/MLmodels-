# functions for preprocessing test + training data
# rememember about function which would call all other functions: preprocess_data(), preprocess_test_data()
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

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

def cross_validation_split(X_train: pd.DataFrame, y_train: pd.Series, cv_type: str = "kfold", n_splits: int = 5, random_state: int = 42):
    """
    X_train: training data
    y_train: training target
    cv_type: 'kfold' or 'stratified'
    n_splits: number of folds
    random_state: reproducibility
    shuffle: whether to shuffle data before splitting
    """

    if cv_type.lower() == "kfold":
        cv = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    elif cv_type.lower() == "stratified":
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    else:
        raise ValueError("cv_type must be 'kfold' or 'stratified'")

    return cv.split(X_train, y_train)

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

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    return X, y

def drop_columns(X: pd.DataFrame, columns_to_drop):
  """
    Drop specified columns from train, validation, or test splits.
    """
  return X.drop(columns=columns_to_drop, errors="ignore")

def encode_target_train(y_train: pd.Series):
  """
  Use LabelEncoder() to encode target variable
  """
  target_encoder = LabelEncoder()
  y_train_encoded = target_encoder.fit_transform(y_train)

  return y_train_encoded, target_encoder

def target_transofrm_test(y_test: pd.Series, target_encoder: LabelEncoder):
  """
  Target transform for test set.
  """
  return target_encoder.transform(y_test)

def detect_missing_values_cols(df: pd.DataFrame, threshold: float = 0.6):
    """
    Returns:
    - columns containing missing values
    - columns where missing percentage exceeds threshold
    """
    cols_with_na = []
    cols_to_delete = []

    for column in df.columns:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            cols_with_na.append(column)

            missing_percentage = missing_count / len(df)

            if missing_percentage > threshold:
                cols_to_delete.append(column)

    return cols_with_na, cols_to_delete

def impute_missing_values(X_train: pd.DataFrame):

  X_train_copy = X_train.copy()
  numeric_features = X_train_copy.select_dtypes(include=["number"]).columns.tolist()
  categorical_features = X_train_copy.select_dtypes(include=["object"]).columns.tolist()

  numeric_imputer = SimpleImputer(strategy="median")
  categorical_imputer = SimpleImputer(strategy="most_frequent")

  if numeric_features:
    X_train_copy[numeric_features] = numeric_imputer.fit_transform(X_train_copy[numeric_features])
  if categorical_features:
    X_train_copy[categorical_features] = categorical_imputer.fit_transform(X_train_copy[categorical_features])

  return X_train_copy, numeric_features, categorical_features, numeric_imputer, categorical_imputer

def impute_missing_values_transform(X_test: pd.DataFrame, numeric_features: list, categorical_features: list, numeric_imputer: SimpleImputer, categorical_imputer: SimpleImputer):

  X_test_copy = X_test.copy()

  for column in numeric_features:
    if column not in X_test_copy.columns:
            X_test_copy[column] = np.nan

  for column in categorical_features:
    if column not in X_test_copy.columns:
            X_test_copy[column] = np.nan

  if numeric_features:
        X_test_copy[numeric_features] = numeric_imputer.transform(X_test_copy[numeric_features])

  if categorical_features:
        X_test_copy[categorical_features] = categorical_imputer.transform(X_test_copy[categorical_features])

  return X_test_copy

def encode_categories_train(X_train: pd.DataFrame, numeric_features: list, categorical_features: list, encoder_type: str = "OneHotEncoder"):
    """
    Supports different types of encoding for categorical variables:
    encoder_type = "OneHotEncoder" or "OrdinalEncoder"
    """
    X_train_copy = X_train.copy()

    if not categorical_features:
        X_train_encoded = X_train_copy[numeric_features].copy()
        feature_columns = X_train_encoded.columns.tolist()
        return X_train_encoded, None, feature_columns

    if encoder_type == "OneHotEncoder":

        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        encoder.fit(X_train_copy[categorical_features])

        encoded = encoder.transform(X_train_copy[categorical_features])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=X_train_copy.index
        )

        X_train_encoded = pd.concat(
            [X_train_copy[numeric_features], encoded_df],
            axis=1
        )

    elif encoder_type == "OrdinalEncoder":

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        encoder.fit(X_train_copy[categorical_features])

        encoded = encoder.transform(X_train_copy[categorical_features])

        encoded_df = pd.DataFrame(
            encoded,
            columns=categorical_features,
            index=X_train_copy.index
        )

        X_train_encoded = pd.concat(
            [X_train_copy[numeric_features], encoded_df],
            axis=1
        )

    else:
        raise ValueError('encoder_type must be "OneHotEncoder" or "OrdinalEncoder"')

    feature_columns = X_train_encoded.columns.tolist()

    return X_train_encoded, encoder, feature_columns

def encode_categories_test(X_test: pd.DataFrame, numeric_features: list, categorical_features: list, encoder, feature_columns: list):

    X_test_copy = X_test.copy()

    if categorical_features:

        encoded = encoder.transform(X_test_copy[categorical_features])

        # OneHotEncoder option
        if isinstance(encoder, OneHotEncoder):

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(categorical_features),
                index=X_test_copy.index
            )
        # Ordinal Encoder option

        elif isinstance(encoder, OrdinalEncoder):

            encoded_df = pd.DataFrame(
                encoded,
                columns=categorical_features,
                index=X_test_copy.index
            )

        else:
            raise ValueError("Unsupported encoder type")

        X_test_encoded = pd.concat(
            [X_test_copy[numeric_features], encoded_df],
            axis=1
        )

    else:
        X_test_encoded = X_test_copy[numeric_features].copy()

    for col in feature_columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0.0

    X_test_encoded = X_test_encoded[feature_columns]

    return X_test_encoded

def scale_numeric_train(X_train_encoded: pd.DataFrame):
  scaler = MinMaxScaler()
  X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)

  return X_train_scaled, scaler

def scale_numeric_test(X_test_encoded: pd.DataFrame):
  X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)

  return X_test_scaled

