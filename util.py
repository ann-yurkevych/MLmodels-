from bank_chunk_data_processing import *

def max_depth_error(md, X_train, y_train, X_validation, y_validation):
    model = DecisionTreeClassifier(max_depth=md, random_state=42)
    model.fit(X_train, y_train)

    train_error = 1 - model.score(X_train, y_train)
    val_error   = 1 - model.score(X_validation, y_validation)

    return {
        'Max Depth': md,
        'Training Error': train_error,
        'Validation Error': val_error
    }


