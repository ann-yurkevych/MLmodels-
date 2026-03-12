# classifier constructors
from preprocessing import *
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def base_Logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    best_log_reg = optimal_parameters_grid_search(LogisticRegression(), X_train, y_train)
    return LogisticRegression(class_weight='balanced').fit(X_train, y_train)

def base_KNearest_Neighbors_classifier(X_train: pd.DataFrame, y_train: pd.Series, params: dict = {"n_neighbors": range(1, 31)}):
    # use cross-validation to find the optimal number of neighbours
    best_knn = optimal_parameters_grid_search(KNeighborsClassifier(), X_train, y_train, params)
    return best_knn

def base_Decision_Tree_classifier(X_train: pd.DataFrame, y_train: pd.Series, params: dict = {"n_neighbors": range(1, 31)}):
    best_Decision_Tree = optimal_parameters_grid_search(DecisionTreeClassifier(), X_train, y_train, params)
    return best_Decision_Tree

def optimal_parameters_grid_search(model, X_train: pd.DataFrame, y_train: pd.Series, parameters: dict, cv: int = 5, scoring: str = "f1_macro"):
    """
    optimal paramters for model attributes (KNN - neigbours) and regularization parameters(Logistic Regression - regularization parameter: C)
    KNN: optimal number of neigbours
    Logistic Regression, SVM: regularization parameters
    Decision Tree: max_depth, min_samples_leaf, criterion
    XGBoost: learning_rate, n_estimators, max_depth
    """

    grid_search_CV = GridSearchCV(model, parameters, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search_CV.fit(X_train, y_train)
    return grid_search_CV.best_estimator_

def RandomizedSearchCV(model, X_train: pd.DataFrame, y_train: pd.Series, parameters: dict, cv: int = 5, scoring: str = "f1_macro"):
    randomized_search_cv = RandomizedSearchCV(model, parameters, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    randomized_search_cv.fit(X_train, y_train)
    return randomized_search_cv.best_estimator_

