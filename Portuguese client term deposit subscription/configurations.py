
# 6 models: Decision Tree, K-Nearest Neigbors Classifier, Logistic Regression, XGBoost, CatBoost, LGBMBoost
# 2 tuning hyper parameters methods: Randomized Search CV, Hyperopt Bayesian Optimization
# as result: 12 tuned models - find the best model: model type, hyperparameters, method
params_search = {
    "DecisionTree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__criterion": ["gini", "entropy"]
    },
    "KNN": {
        "model__n_neighbors": range(1, 20),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "LogisticRegression": {
        "model__penalty": ["l1", "l2", "elasticnet"],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__l1_ratio": [0, 0.5, 1],
    },
    
    "XGBoost": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    },

    "CatBoostClassifier": {
        "model__iterations": [100, 200, 500],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.1],
    },

    "LGBMClassifier": {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [-1, 3, 5, 7],
        "model__learning_rate": [0.01, 0.1],
        "model__num_leaves": [31, 50, 100],
    },

}