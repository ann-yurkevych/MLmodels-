"""IMBALANCED LOGISTIC REGRESSION 

PIPELNE
1. Load the data.
2. Split into training/test. 80/20 - train_test_split, stratified
3.
4. Impute missing values.
5. Encode categorical variables. + scale numerical features.
6. Apply resampling to train-folds only. : SMOTE, SMOTE Tomek.
4. Training: Use Stratified K-fold splitting for training and validation sets.
5. Evaluation
6. Testing.
7. Save test results in a csv file -> output from classification metrics?
7. What I mean is: save confusion matrices, AUC ROC plots - Per-class AUC scores, classification reports result, Cohen's Kappa score, cross-validation scores

"""

def main():
    print("Hello")

if __name__ == "__main__":
    main()