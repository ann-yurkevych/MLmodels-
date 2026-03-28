
# 6 models: Decision Tree, K-Nearest Neigbors Classifier, Logistic Regression, XGBoost, CatBoost, LGBMBoost
# 2 tuning hyper parameters methods: Randomized Search CV, Hyperopt Bayesian Optimization

params_grid_search = {
    "DecisionTree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "KNN": {
        "model__n_neighbors": range(1, 20),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "LogisticRegression": {
        "model__n_neighbors": range(1, 20),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    },

    "CatBoostClassifier": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    },

    "LGBMClassifier": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    },
}