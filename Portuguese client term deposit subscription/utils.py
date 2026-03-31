import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
# this file will be used for different methods of feature importance decision + SHAP library analysis

# DETERMINE THE FEATURE IMPORTANCE


# tree-based models
def feature_importance_tree_based(model, X):

    feature_importance_df = (
        pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        .sort_values(by='Importance', ascending=False)
    )
 
    feature_importance_df.set_index('Feature').plot.bar(
        figsize=(10, 5),
        title=f'Feature importances — impurity-based ({type(model).__name__})',
        legend=False
    )
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    plt.savefig('feature_importance.png', bbox_inches='tight', dpi=150)
    plt.close()
 
    return feature_importance_df

# feature importance for any models

def permutation_feature_importance(model, X, y):
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
 
    feature_importance_df = (
        pd.DataFrame({'Feature': X.columns, 'Importance': perm.importances_mean})
        .sort_values(by='Importance', ascending=False)
    )
 
    feature_importance_df.set_index('Feature').plot.bar(
        figsize=(10, 5),
        title=f'Feature importances — permutation-based ({type(model).__name__})',
        legend=False
    )
    plt.ylabel('Mean accuracy drop')
    plt.tight_layout()
    plt.show()
    plt.savefig('feature_importance_permutation.png', bbox_inches='tight', dpi=150)
    plt.close()
    return feature_importance_df

# THE IMPACT OF FEATURES ON PREDICTIONS
def shap_summary(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
 
    # For binary classification, shap_values is a list [class_0, class_1]
    values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
 
    print("── SHAP summary (bar) ──")
    shap.summary_plot(values_class1, X_train, plot_type="bar")
    plt.savefig('shap_bar.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("── SHAP summary (beeswarm) ──")
    shap.summary_plot(values_class1, X_train)
    plt.savefig('shap_beeswarm.png', bbox_inches='tight', dpi=150)
    plt.close()
    return explainer, shap_values


def shap_single_prediction(model, X_train, X_test, row_index=0):
    """
    Explains the model's prediction for one specific record using a SHAP force plot.
    Useful for understanding why the model classified a particular sample the way it did.
 
    Args:
        model:      fitted tree-based classifier
        X_train:    training DataFrame (needed to fit the explainer)
        X_test:     test DataFrame containing the record to explain
        row_index:  index of the row in X_test to explain (default: 0)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
 
    # For binary classification, use class 1 values and expected value
    values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value
    )
 
    shap.waterfall_plot(
        shap.Explanation(
            values=values_class1[row_index],
            base_values=expected_value,
            data=X_test.iloc[row_index],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.savefig(f'shap_force_row{row_index}.png', bbox_inches='tight', dpi=150)
    plt.close()

# ANALYSE THE ERRORS MODEL MADE
def analyze_errors(model, X_test, y_test):
    """
    Identifies and returns all test records where the model's prediction
    does not match the true label. Useful for spotting systematic patterns
    in model mistakes (e.g. a certain class or feature range always misclassified).
 
    Args:
        model:  fitted classifier with a predict() method
        X_test: test feature DataFrame
        y_test: true labels (Series or array)
 
    Returns:
        DataFrame with all misclassified records, their true label, and predicted label.
    """
    y_pred = model.predict(X_test)
 
    error_df = X_test.copy()
    error_df['true_label'] = y_test.values
    error_df['predicted_label'] = y_pred
    error_df = error_df[error_df['true_label'] != error_df['predicted_label']]
 
    print(f"Total errors: {len(error_df)} / {len(X_test)} ({100 * len(error_df) / len(X_test):.1f}%)")
    print(error_df[['true_label', 'predicted_label']].value_counts())
 
    return error_df
