
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from configurations import params_search
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# RANDOMIZED SEARCH tuning
def randomized_search_cv(model, model_name, X_train, y_train, n_iter=20, cv=5):
    clf = RandomizedSearchCV(estimator=model, param_distributions=params_search[model_name], n_iter=n_iter, scoring='f1_weighted', cv=cv, random_state=0, n_jobs=-1, refit=True)
    clf.fit(X_train, y_train)
    return clf.best_estimator_, clf.best_params_

# HYPEROPT: Bayesian Optimisation, used objective function inside of every boosting algorithm tuning
def tune_xgb(X_train, y_train, X_val, y_val, max_evals=20):
    def objective(params):
        clf = xgb.XGBClassifier(
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            max_depth=int(params['max_depth']),
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            enable_categorical=True,
            use_label_encoder=False,
            missing=np.nan,
            device='cuda',
            early_stopping_rounds=10
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = clf.predict(X_val)
        return {'loss': -f1_score(y_val, pred, average='weighted'), 'status': STATUS_OK}

    space = {
        'n_estimators':     hp.quniform('n_estimators', 50, 500, 25),
        'learning_rate':    hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth':        hp.quniform('max_depth', 3, 15, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample':        hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma':            hp.uniform('gamma', 0, 0.5),
        'reg_alpha':        hp.uniform('reg_alpha', 0, 1),
        'reg_lambda':       hp.uniform('reg_lambda', 0, 1)
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=Trials())
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])

    final = xgb.XGBClassifier(**best, enable_categorical=True,
                               use_label_encoder=False, missing=np.nan, device='cuda')
    final.fit(X_train, y_train)
    return final, best


def tune_lightgbm(X_train, y_train, X_val, y_val, max_evals=20):
    def objective(params):
        clf = LGBMClassifier(
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            max_depth=int(params['max_depth']),
            num_leaves=int(params['num_leaves']),
            min_child_samples=int(params['min_child_samples']),
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
        pred = clf.predict(X_val)
        return {'loss': -f1_score(y_val, pred, average='weighted'), 'status': STATUS_OK}

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'num_leaves': hp.quniform('num_leaves', 20, 150, 1),   # LightGBM-specific
        'min_child_samples': hp.quniform('min_child_samples', 5, 50, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1)
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=Trials())
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])
    best['num_leaves'] = int(best['num_leaves'])
    best['min_child_samples'] = int(best['min_child_samples'])

    final = LGBMClassifier(**best, random_state=42, verbose=-1)
    final.fit(X_train, y_train)
    return final, best


def tune_catboost(X_train, y_train, X_val, y_val, max_evals=20):
    def objective(params):
        clf = CatBoostClassifier(
            iterations=int(params['iterations']),
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            l2_leaf_reg=params['l2_leaf_reg'],
            bagging_temperature=params['bagging_temperature'],  # CatBoost-specific
            random_strength=params['random_strength'],          # CatBoost-specific
            random_state=42,
            verbose=0,
            early_stopping_rounds=10
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        pred = clf.predict(X_val)
        return {'loss': -f1_score(y_val, pred, average='weighted'), 'status': STATUS_OK}

    space = {
        'iterations': hp.quniform('iterations', 50, 500, 25),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'depth': hp.quniform('depth', 3, 10, 1),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        'random_strength': hp.uniform('random_strength', 0, 1),
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=max_evals, trials=Trials())
    best['iterations'] = int(best['iterations'])
    best['depth'] = int(best['depth'])

    final = CatBoostClassifier(**best, random_state=42, verbose=0)
    final.fit(X_train, y_train)
    return final, best