import mlflow
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

from preprocessing import (
    load_data,
    extract_target,
    split_train_test,
    encode_target, 
    numeric_categorical_features,
    drop_columns,
    unknown_values_replace,
    binarize_feature,
    get_categorical_indices
)

from classifiers import (
    base_decision_tree_model, 
    base_logistic_regression,
    base_KNeighborsClassifier,
    base_xgboost,
    base_lightgbm,
    base_catboost
)


"""PIPELINE
1. Load the data. 
2. Encode the target variable from string to numeric value. 
3. Drop features (based on prior EDA). 
4. Create new features if needed for the model.
5. Divide dataset into training, test, validation using stratified train_test_split. 
6. Extract target variable from each training, test, validation sets.

STEPS 7-14 WILL BE DONE IN THE PIPELINE, using ColumnTransformer, FunctionTransformer

7. Detect and impute missing values: no numeric missing values were detected, "unknown" categories will be replaced with SimpleImputer "most_frequent"
8. Encode categorical variables. 
9. Binarize previous feature: based on EDA I found out that subscription rate is high if previous > 1.
10. Scale numeric variables (fit + transform only on training, transform on validation + test sets)
11. Apply SMOTE techniques to handle class imbalance (only on the training data). 
12. For each model train the model with hyperparameters.
13. Pick the best hyperparameters (hyperparamters for each model with lowest validation error are the best for the model).
14. Train the best model using hyperparameters.

15. Evaluate on test set. 
16. Analysis of feature importance table.
17. Analysis of impact of features on predictions using SHAP library.

"""

print("Loading the data")

raw_df = load_data("data/bank-additional-full.csv")

# encode target variable from str to num
raw_df = encode_target(raw_df, 'y') 

# drop features: based on conducted EDA in exporation.ipynb 
raw_df = drop_columns(['duration', 'emp.var.rate', 'nr.employed']) 

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

# indices for category columns
columns_category_indices = get_categorical_indices(X_train, numeric_features)

# PIPELINE STARTS # 

# impute missing values
X_train, imputer = unknown_values_replace(X_train, ['default', 'education'])
X_validation = unknown_values_replace(X_validation, ['default', 'education'], imputer=imputer) # imputer, so no fit() on test/validation
X_test = unknown_values_replace(X_test, ['default', 'education'], imputer=imputer)

# impute missing values in numeric cols: there are no missing numeric values in this dataset, but for future reference


# binarize the 'previous' features (decision is made based on EDA)
X_train, X_validation, X_test = binarize_feature(X_train, X_validation, X_test, 'previous')

# encode categorical variables with OneHotEncoder

# scale numeric cols with StandardScaler()

# SMOTENC resampling
smotenc = SMOTENC(categorical_features=columns_category_indices, random_state=42)
X_train_smotenc, y_train_smotenc = smotenc.fit_resample(X_train, y_train)


# models training : multiple models with hyperparameters tuning in both ways: Hyperopt and Randomized Search CV





"""
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]), numeric_features),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
    
])

"""