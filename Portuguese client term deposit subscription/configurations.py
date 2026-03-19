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
    
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    },
}