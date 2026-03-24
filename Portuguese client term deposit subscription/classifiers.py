
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# HYPERPARAMETER TUNING: Randomized Search, Hyperopt: Bayesian Optimization: hyperparameters are stored in configurations.py
# tuning is stored in hyperparameter_tuning file

def base_decision_tree_model():
  return tree.DecisionTreeClassifier(random_state=42, class_weight='balanced')

def base_logistic_regression():
  return LogisticRegression(random_state=42, class_weight='balanced')

def base_KNeighborsClassifier():
  return KNeighborsClassifier(n_neighbors=3)

# Boosting algorithms: XGBoost, LightGBM, CatBoost
def base_xgboost():
    return xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

def base_lightgbm():
    return LGBMClassifier(random_state=42, verbose=-1)

def base_catboost():
    return CatBoostClassifier(random_state=42, verbose=0)

# I used voting classifier for two purposes: compare boosting algorithms and compare all models

def stacking_classifier_all_models():
    base_learners = [ # estimators in the attribute from scikit 
        ('decision_tree', base_decision_tree_model()),
        ('knn', base_KNeighborsClassifier()),
        ('xgboost', base_xgboost()),
        ('lightgbm', base_lightgbm()),
        ('catboost', base_catboost()),
    ]

    learner = base_logistic_regression() # state in the README why you chose LogisticRegression as meta learner: low risk of overfitting

    return StackingClassifier(estimators=base_learners, final_estimator=learner, cv=5, passthrough=False, stack_method='predict_proba')

def voting_classifier_all_models(voting: str="soft"):
    estimators = [
        ('decision_tree', base_decision_tree_model()),
        ('logistic_regression', base_logistic_regression()),
        ('knn', base_KNeighborsClassifier()),
        ('xgboost', base_xgboost()),
        ('lightgbm', base_xgboost()),
        ('catboost', base_catboost()),
    ]
    return VotingClassifier(estimators=estimators, voting=voting)


def compare_boosting_algorithms(X_train: pd.DataFrame, y_train, cv=5):
    """
    Runs cross-validation on XGBoost, LightGBM, CatBoost
    and prints a comparison.
    """
    boosting_models = {
        'XGBoost':   base_xgboost(),
        'LightGBM':  base_xgboost(),
        'CatBoost':  base_catboost(),
    }
    
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    
    print("=" * 45)
    print(f"{'Model':<12} {'Mean F1':>10} {'Std':>10}")
    print("=" * 45)
    
    for name, model in boosting_models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=kfold,
            scoring='f1_weighted'
        )
        results[name] = scores
        print(f"{name:<12} {scores.mean():>10.4f} {scores.std():>10.4f}")
    
    print("=" * 45)
    return results

def compare_all(X_train, y_train, X_test, y_test):
    models = {
        # Base model
        'Decision Tree':        base_decision_tree_model(),
        'Logistic Regression':  base_logistic_regression(),
        'KNN':                  base_KNeighborsClassifier(),

        # Boosting algorithms
        'XGBoost':              base_xgboost(),
        'LightGBM':             base_lightgbm(),
        'CatBoost':             base_catboost(),

        # Ensemble models
        'Voting (soft)':        voting_classifier_all_models(voting='soft'),
        'Voting (hard)':        voting_classifier_all_models(voting='hard'),
        'Stacking':             stacking_classifier_all_models(),
    }
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\n{'Model':<25} {'CV F1 Mean':>12} {'CV F1 Std':>10} {'Test F1':>10}")
    print("=" * 62)
    
    for name, model in models.items():
        # Cross-val on train
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_weighted')
        
        # Fit and evaluate on test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{name:<25} {cv_scores.mean():>12.4f} {cv_scores.std():>10.4f} {test_f1:>10.4f}")

