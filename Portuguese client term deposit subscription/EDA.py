import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Conduct exploratory data analysis 

# SHAPE AND DATA TYPES
def decriptive_stats(df: pd.DataFrame):
    data_frame_shape = df.shape
    df_types = df.dtypes
    df_info = df.info()
    df_describe = df.describe()
    print(data_frame_shape, df_types, df_info, df_describe)

# MISSING VALUES STATISTICS: % of missing values per each column, missing pattern type, isnull() per column
def missing_values_statistics():
    print()

# Skewness, kurtosis, percentiles

# DISTRIBUTIONS, OUTLIERS

# histograms, boxplots, violin plots, Z-score, IQR plot
# conduct normality tests: Shapiro-Wilk, D'Agostino-Pearson, plot Q-Q plots

# test for specific distribution: conduct Anderson-Darling

# MULTICOLLINEARITY DETECTION: VIF

# CATEGORICAL ANALYSIS

# CORRELATIONS
# heatmaps, Pearsons correlation, Cramér's V(only for categorical variables)
"""
Cramér's V - variable association
Chi-Square test
Cramér's V = 
No association with target → likely useless feature
High association with target + low association with other features → ideal feature
High association with another feature → redundant, pick one
"""




# TARGET VARIABLE ANALYSIS

# distribution of target variable: histograms to spot class imbalance
# imbalance ration definition
# feature vs target: scatter plot, box plot per class


"""
* remember to apply SMOTE only on training set, not validation or test
In this section make conclusion about if you need to use any technoiques to handle class imbalance:
1. 
2. SMOTE: SMOTE(only on numeric), SMOTENC(for numeric + categorical), SMOTEN(categorical), SMOTE-Tomek(numeric, but )
3. class_weight = "balanced"

Also, based on TARGET VARIABLE ANALYSIS, you decide which performance metrics to use
if your target is heavily imbalanced (95% class 0, 5% class 1), accuracy is misleading. You'd use F1, ROC-AUC, or precision-recall instead.

"""

# INTERACTION TERMS check
"""
interaction term: the effect of feature A on the target depends on the value of feature B.
if you spot an interaction term -> for linear models you need to create manually an interaction term
"""





"""
1. Normality tests: is the data normally distributed? 
2. Variance Equality Tests: do my groups have equal variance? - it's only for categorical variables
Levene's test — robust to non-normality
Bartlett's test — assumes normality
You need it when you want to compare categorical variables with some continuous variable: 

3. 
"""


# make conclusions/assumptions which features influence target variable y

"""ADDITIONAL CONCLUSIONS TO MAKE
which features to encode and how, which to scale, which to drop, 
which to transform, and which models are reasonable given what you found.
"""