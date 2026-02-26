# functions for preprocessing test + training data
# rememember about function which would call all other functions: preprocess_data(), preprocess_test_data()
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

def split_train_test(df: pd.DataFrame, target: str):
    """
    Splits a dataset into training and test sets 
    using stratified sampling based on the target variable.

    Args:
        df (pd.DataFrame): Full dataset containing features and target.
        target (str): Target column name used for stratification.

    Returns:
        X_train (pd.DataFrame): Training subset.
        X_validation (pd.DataFrame): Validation subset.
    """
    
    X_train, X_test= train_test_split(
        df,
        test_size=0.20,  # 80% train, 20% validation
        stratify=df[target],
        random_state=42
    )

    return X_train, X_test