# work on reducing variance, bias-variance trade-off

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report
)

def evaluate_model(estimator, X, y_true, model_name, method, preprocessed=False):
    """
    Returns a dict of metrics for one model.
    preprocessed=True  → X is already preprocessed (Hyperopt models)
    preprocessed=False → X goes through the full pipeline (RandomizedSearch models)
    """
    y_pred      = estimator.predict(X)
    y_pred_prob = estimator.predict_proba(X)[:, 1]
 
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
 
    return {
        'model_name':  model_name,
        'method':      method,
        'precision':   precision_score(y_true, y_pred,      average='weighted'),
        'recall':      recall_score(y_true, y_pred,         average='weighted'),
        'f1':          f1_score(y_true, y_pred,             average='weighted'),
        'roc_auc':     roc_auc_score(y_true, y_pred_prob),
        'fpr':         fpr,
        'tpr':         tpr,
        'estimator':   estimator,
        'preprocessed': preprocessed,
    }