import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



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

from configurations import params_search
from hyperparameters_tuning import (
    randomized_search_cv,
    tune_xgb,
    tune_lightgbm,
    tune_catboost
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report
)
from evaluation import evaluate_model, build_excel_report
from utils import shap_summary, shap_single_prediction, analyze_errors, feature_importance_tree_based
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
raw_df = drop_columns(raw_df, ['duration', 'emp.var.rate', 'nr.employed']) 

# split dataset into train/test: 80/20
X_train, X_test = split_train_test(raw_df, 'y')

# split train dataset into train/validation: 60/20
X_train, X_validation = split_train_test(X_train, 'y')

# binarize previous variable
X_train, X_validation, X_test = binarize_feature(X_train, X_validation, X_test, 'previous')

# extract target variable from each set: training, validation, test: y_train, y_validation, y_test
X_train, y_train = extract_target(X_train, 'y') 
X_validation, y_validation = extract_target(X_validation, 'y') 
X_test, y_test = extract_target(X_test, 'y') 

# replace unknown categories
X_train, imputer = unknown_values_replace(X_train, ['default', 'education'])
X_validation, _ = unknown_values_replace(X_validation, ['default', 'education'], imputer=imputer)
X_test, _ = unknown_values_replace(X_test,       ['default', 'education'], imputer=imputer)

# determine numeric, categorical features
numeric_features, categorical_features = numeric_categorical_features(X_train)

# indices for category columns
columns_category_indices = get_categorical_indices(X_train, numeric_features)

# PIPELINE STARTS # 

models = {
    'DecisionTree':         base_decision_tree_model(),
    'KNN':                  base_KNeighborsClassifier(),
    'LogisticRegression':   base_logistic_regression(),
    'XGBoost':              base_xgboost(),
    'CatBoostClassifier':   base_catboost(),
    'LGBMClassifier':       base_lightgbm(),
}


def build_pipeline(model, numeric_features, categorical_features):
    """
    Builds a full imblearn Pipeline:
    preprocessor → SMOTENC → model
    
    resampling is used only for training set
    """

    numeric_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

 
    categorical_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

   
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


    return Pipeline(steps=[ # imbalanced pipeline, not scikit
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)          
    ])

best_models_randomized = {}

for name, model in models.items():
    print(f"\nTuning of {name}")
    pipeline = build_pipeline(model, numeric_features, categorical_features)
    
    best_estimator, best_params = randomized_search_cv(
        pipeline,
        name, 
        X_train, y_train
    )
    best_models_randomized[name] = {
        'estimator': best_estimator,
        'params': best_params,
        'method': 'RandomizedSearchCV',
    }
    print(f"{name} best params: {best_params}")

# fitting preprocessing on training, transforming on test/validation
_prep_pipeline = build_pipeline(base_xgboost(), numeric_features, categorical_features)
_prep_pipeline.named_steps['preprocessor'].fit(X_train, y_train)
preprocessor_fit = _prep_pipeline.named_steps['preprocessor']
 
X_train_proc = preprocessor_fit.transform((X_train.reset_index(drop=True)))
X_val_proc   = preprocessor_fit.transform((X_validation.reset_index(drop=True)))
X_test_proc  = preprocessor_fit.transform((X_test.reset_index(drop=True)))

# SMOTE on training only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train)
 
boosting_tuners = {
    'XGBoost':            tune_xgb,
    'LGBMClassifier':     tune_lightgbm,
    'CatBoostClassifier': tune_catboost,
}
 
best_models_hyperopt = {}

for name, tuner in boosting_tuners.items():
    print(f"\n  Tuning {name} with Hyperopt...")
    best_estimator, best_params = tuner(
        X_train_resampled, y_train_resampled,
        X_val_proc,        y_validation
    )
    best_models_hyperopt[name] = {
        'estimator': best_estimator,
        'params':    best_params,
        'method':    'Hyperopt',
    }
    print(f"  Best params: {best_params}")

results = []
for name, result in best_models_randomized.items():
    r = evaluate_model(
        result['estimator'], X_validation, y_validation,
        name, 'RandomizedSearchCV', preprocessed=False
    )
    results.append(r)
    print(f"  {name:<22} RandomizedSearchCV  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}")
 
for name, result in best_models_hyperopt.items():
    r = evaluate_model(
        result['estimator'], X_val_proc, y_validation,
        name, 'Hyperopt', preprocessed=True
    )
    results.append(r)
    print(f"  {name:<22} Hyperopt  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}")


best_result = max(results, key=lambda r: r['roc_auc'])
print(f"\n  ★ Best model: {best_result['model_name']} ({best_result['method']})")
print(f"    Validation AUC: {best_result['roc_auc']:.4f}")
 
if best_result['preprocessed']:
    y_test_pred      = best_result['estimator'].predict(X_test_proc)
    y_test_pred_prob = best_result['estimator'].predict_proba(X_test_proc)[:, 1]
else:
    y_test_pred      = best_result['estimator'].predict(X_test)
    y_test_pred_prob = best_result['estimator'].predict_proba(X_test)[:, 1]
 
test_f1  = f1_score(y_test, y_test_pred, average='weighted')
test_auc = roc_auc_score(y_test, y_test_pred_prob)
 
print(f"\n  FINAL TEST RESULTS")
print(f"  F1 (weighted): {test_f1:.4f}")
print(f"  ROC AUC:       {test_auc:.4f}")
print(f"\n{classification_report(y_test, y_test_pred)}")

build_excel_report(
    results               = results,
    best_result           = best_result,
    best_models_randomized= best_models_randomized,
    best_models_hyperopt  = best_models_hyperopt,
    y_test                = y_test,
    y_test_pred           = y_test_pred,
    y_test_pred_prob      = y_test_pred_prob,
    X_val_proc            = X_val_proc,
    X_validation          = X_validation,
    y_validation          = y_validation,
    X_test_proc           = X_test_proc,
    X_test                = X_test,
    output_path           = 'model_results.xlsx',
)

# extract raw model from pipeline
best_catboost_pipeline = best_models_randomized['CatBoostClassifier']['estimator']
catboost_model = best_catboost_pipeline.named_steps['model']

# get feature names after preprocessing
ohe_feature_names = (
    best_catboost_pipeline
    .named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(categorical_features)
    .tolist()
)
all_feature_names = numeric_features + ohe_feature_names

X_train_proc_df = pd.DataFrame(X_train_proc, columns=all_feature_names)
X_test_proc_df  = pd.DataFrame(X_test_proc, columns=all_feature_names)

# 7. Feature importance
print("\n── Feature Importance ──")
feature_importance_tree_based(catboost_model, X_train_proc_df)

# SHAP — feature impact on predictions
explainer, shap_values = shap_summary(catboost_model, X_train_proc_df)
shap_single_prediction(catboost_model, X_train_proc_df, X_test_proc_df, row_index=0)

# Error analysis
print("\n── Error Analysis ──")
error_df = analyze_errors(catboost_model, X_test_proc_df, y_test)
print(error_df.describe())
