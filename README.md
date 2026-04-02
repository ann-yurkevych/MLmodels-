# Portuguese client term deposit subscription modeling

### Predicting client behaviour on subscription rate for a term deposit. 
Exploratory data analysis and followed by it decisions were made in `exploratory.ipynb` file. 


All evaluation results with AUC ROC plot, recall, precision, F1-score and model comparisons can be found in `model_results.xlsx`
## PROJECT STRUCTURE
```
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
```
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
![Feature importance bar plots](Portuguese_client_term_deposit/images/validation_set/AUC_ROC.png)

All boosting models are practically identical. `XGBoost, LightGBM, and CatBoost` — regardless of whether they were tuned with RandomizedSearchCV or Hyperopt — produced curves that overlap almost perfectly, all sitting between 0.80 and 0.802. 

In the bottom-left region (low false positive rate), all boosting models rise closely together. They all agree on the most obvious subscribers. They are making the same decisions on individual clients. 

KNN has a different shaped curve. It is a sign that the model is not producing reliable probability estimates, just hard boundaries. KNN is the lowest performed model.

DecisionTree has a strange flat spot around 0.15-0.20 false positive rate. I noticed that the curve stops climbing for a moment before continuing upward. What this means s is that at certain thresholds, the model temporarily stops getting better at finding subscribers. I found this to be a classic sign that the tree memorized some very specific patterns from the training data that don't hold up on new clients.

What really struck me was that a very simple Logistic Regression at 0.796 almost perfectly matched the best complex boosting model at 0.802: their curves sit almost on top of each other.
This tells me that adding more complexity to the model won't help.
![Models evaluation metrics](Portuguese%20client%20term%20deposit%20subscription/images/test%20set/evaluation_results.png)

If I were to compare three boosting models tuned with both methods, RandomizedSearchCV consistently matched Hyperopt. 

LGBMClassifier with RandomizedSearchCV got `0.8018` AUC vs `0.8011` with Hyperopt, CatBoost got `0.8014` vs `0.7821`. This tells me that the search space I defined for Hyperopt wasn't well-calibrated — Bayesian optimization only wins when the search space is meaningful, and here RandomizedSearchCV found equally good regions with less complexity.

KNN is the real outlier and it reveals something about the data. With only 0.667 AUC, KNN struggled significantly more than every other model. KNN relies on distance between points being meaningful, this poor performance suggests the feature space after one-hot encoding is very sparse, making distance metrics unreliable (more features increases dimensionality which actually hurts the model and doesn't improve it). 


All models ended up having almost similar hyperparameters for boosting. XGBoost, LGBM, and CatBoost all ended up with n_estimators=500, max_depth=7, learning_rate=0.01 regardless of tuning method. This tells me that more tuning won't help. (However, changing input features and experimenting with creating new features based on current can actually help). 

`LogisticRegression` achieved 0.7956 AUC: a linear model with essentially one hyperparameter (C=100), when the best boosting models reached ~0.80. (It could mean that complex models don't capture linear relationships).

![LGBMBoost Classifier classification report](Portuguese%20client%20term%20deposit%20subscription/images/test%20set/test-set-LGBMBoost-classification-report.png)

The weighted F1 of 0.891 looks great but is misleading. That number is heavily influenced by how well the model handles non-subscribers (class 0), which make up the vast majority of the data. If I only look at this number, the model seems excellent, but it hides a much bigger problem underneath.

The model misses every second potential subscriber. Recall of 0.49 for class 1 means that out of 928 real subscribers in the test set, the model only caught about half of them. The other half were incorrectly labeled as "won't subscribe." For a bank running a marketing campaign, those missed clients are direct lost revenue.

When the model does predict a subscription, it's right about half the time. Precision of 0.53 for class 1 means that if the bank called everyone the model flagged as a likely subscriber.

The model performing slightly better on the test set than validation is a good sign. The fact that it slightly improved, the model genuinely learned real patterns rather than memorizing the training data. It generalizes well to clients it has never seen before.
The best performance (based on ROC AUC plot) I received from LGBMBoost Classifier. 
## Feature importance
![Feature importance bar plots](Portuguese%20client%20term%20deposit%20subscription/images/SHAP%20analysis/feature_importance.png)

The bar chart shows how much each feature "matters" to the model when making decisions.

`campaign` (number of times the client was called) has a bar almost **4x taller** than everything else — the model splits on it constantly. `day_of_week` and `job_admin` come next, but at a much lower level. `euribor3m` (the interest rate) sits somewhere in the middle despite being economically very meaningful.

Thus the bar plot shows only partially true: favors features with many categories (one-hot encoded `job`, `education`, `day_of_week`) because each category gets its own bar and collectively they look more important than they are. Meanwhile,`euribor3m` gets one bar and looks weaker than it actually is.
Features near zero (`poutcome_nonexistent`, `default_yes`) I consider as noise and are candidates for removal in future model iterations.

## SHAP analysis
![Mean Absolute Impact](Portuguese%20client%20term%20deposit%20subscription/images/SHAP%20analysis/shap_bar.png)

I found out that two features determined (in general settings) what model learned: `euribor3m` and `campaign`. 

The third most important feature (`contact_cellular`) has less than a third of their impact. 

I found out that the model has essentially learned that term deposit subscription comes down to two completely opposite forces — one that cannot be controlled, and one that can.

I noticed that `euribor3m` represents the interest rate that nobody at the bank can change. When rates are favorable, clients are naturally more receptive, regardless of anything the bank does. Finding this feature as the most important in SHAP bar plot tells that market timing matters more than any client characteristic — a client's job, education, marital status, or housing situation barely moves the needle compared to whether the economy is in a low-rate period.

What surprised me most was `campaign` was the second important feature. I expected client profile features to dominate, but instead I found that call frequency something fully within the bank's control is the second strongest predictor. This is actually a warning sign, not a success story. It suggests the bank has been compensating for poor client targeting by calling more, and the model has simply learned that being called many times is itself a strong signal that the client won't subscribe.

I also can see that everything below these two `contact type`, `month`, `day of week`, `job`, edu`cation reflects patterns rather than client insight. This tells me the current feature set has a ceiling. Better client profiling data financial history, savings behavior, previous product holdings could potentially unlock predictive signals than what's currently available in this dataset.


![Direction of Impact](Portuguese%20client%20term%20deposit%20subscription/images/SHAP%20analysis/shap_beeswarm.png)
`euribor3m`: High values (red) push strongly negative (left), meaning high interest rates reduce subscription probability. Low values (blue) push positive — clients subscribe more when rates are low. This is economically intuitive.

`campaign`: High values (red, many contacts) strongly push negative — being contacted many times actually hurts conversion, likely indicating client fatigue.

`contact_cellular`: Being contacted by cellular (red=1) pushes positive, telephone (blue=0) pushes negative — cellular contact is more effective.

`month_may`: Being in May (red=1) pushes slightly negative — May campaigns underperform other months.
default_no and default_unknown: Clients with no credit default push slightly positive, unknown default status pushes negative.

![Single Prediction: row 0](Portuguese%20client%20term%20deposit%20subscription/images/SHAP%20analysis/shap_force_row0.png)
For this specific client, the model was very confident they would not subscribe to a term deposit.

The starting point (average prediction across all clients) was `0.776`, but the final score dropped to `−1.855` — far into "will not subscribe" territory.

The biggest reasons the model said NO:

`campaign (−0.5)` — this client was contacted multiple times during the campaign, which actually signals they are unlikely to subscribe (too many calls = client fatigue)
`default_unknown` and `default_no (−0.37 each)` — the client's credit default status pulled the prediction down
`day_of_week_fri (−0.23)` — the contact was not made on a Friday, which the model associates with lower conversion
47 other features combined (−0.75) — the rest of the client's profile also collectively pushed toward "no"

The only thing working in favor of subscription:

`euribor3m (+0.09)` — the interest rate at the time was relatively low, which generally makes term deposits more attractive to clients

Overall, almost everything about this client pointed toward not subscribing, and the model correctly captured that.
## How to improve the model?
From SHAP analysis, I can see that `euribor3m` is a very important feature. 

I can generate other features based on this one: how it changes over time, create categories from it(low, medium, high).

I also could delete noise features such as: `poutcome_nonexistent`, `default_yes`, `job_housemaid`, `job_retired`, `month_dec`.

SHAP analysis shows that high campaign values push strongly negative. I could convert campaign into categorical feature, so the model learns behaviour better: campaign > 3 is worse than campaign <= 2 because clients get annoyed. I could binarize it in a way I did with `previous` feature. 

the model is good at pushing non-subscribers far negative, but struggles to confidently push subscribers positive. I could do some work around this, possibly by trying different resampling methods, experimenting with it using classification report results. 

