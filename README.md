# Portuguese client term deposit subscription modeling

### Predicting client behaviour on subscription rate for a term deposit. 
Exploratory data analysis and followed by it decisions were made in `exploratory.ipynb` file. 
## PROJECT STRUCTURE
Portuguese client term deposit subscription/
│
├── data/
│   └── bank-additional-full.csv       # Raw dataset
│
├── main.py                            # Main pipeline: data preprocessing, tuning, evaluation, Excel export
├── preprocessing.py                   # Data loading, splitting, encoding, feature engineering
├── classifiers.py                     # Base model definitions
├── configurations.py                  # Hyperparameter search spaces
├── hyperparameters_tuning.py          # RandomizedSearchCV + Hyperopt Bayesian tuning
├── evaluation.py                      # Model evaluation metrics
├── EDA.py                             # Functions for EDA
├── exploration.ipynb                  # EDA notebook
├── statistical_tests.py               # Statistical significance tests: TO BE UPDATED 
├── utils.py                           # SHAP interpretation, feature importance
│
├── model_results.xlsx                 # Output: metrics, ROC curves, classification reports
├── requirements.txt                   
└── README.md

## INSTALLATION
Clone the repository
```bash
git clone <your-repo-url>
cd "Portuguese client term deposit subscription"
```
Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```

## HOW TO RUN
```bash
python main.py
```
Results are saved to `model_results.xlsx`.


## Machine Learning models used for predicting
| Model | Tuning Method |
|---|---|
| Decision Tree | RandomizedSearchCV |
| K-Nearest Neighbors | RandomizedSearchCV |
| Logistic Regression | RandomizedSearchCV |
| XGBoost | RandomizedSearchCV + Hyperopt |
| LightGBM | RandomizedSearchCV + Hyperopt |
| CatBoost | RandomizedSearchCV + Hyperopt |

 I chose these 6 models to cover a wide range of approaches. I started with 3 simple models — Decision Tree, Logistic Regression, and KNN — as baselines to have a reference point. If a complex model can't beat a Decision Tree, something is wrong with it. I picked the 3 most popular gradient boosting libraries — XGBoost, LightGBM, and CatBoost. All three are known to perform best on tabular data, but they work differently under the hood, so I wanted to compare them directly on the same dataset and see which one wins. 

## Performance metrics

## Feature importance
![Feature importance bar plots](Portuguese client term deposit subscription/images/feature_importance.png)
## SHAP analysis
![Alt text](path/to/your/image.png)
## How to improve the model?
