import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SHAPE AND DATA TYPES
def type_shape_stats(df: pd.DataFrame):
    print("dataframe shape:", df.shape)
    print("dataframe types:", df.dtypes)

def descriptive_stats(df: pd.DataFrame):
    print(df.describe())


# MISSING VALUES STATISTICS: 

def missing_values_summary(df: pd.DataFrame):
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    
    summary = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct': missing_pct
    })
    
    summary = summary[summary['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    return summary

# missing pattern type: MCAR, MAR, and MNAR


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
def class_distribution_plot(df: pd.DataFrame, target: str = 'y'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of counts
    counts = df[target].value_counts()
    axes[0].bar(counts.index.astype(str), counts.values, color=['green', 'coral'], edgecolor='white', width=0.5)
    axes[0].set_title('Class Distribution (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        # set the total number of counts per bar
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
    
    # Bar plot of proportions
    proportions = df[target].value_counts(normalize=True)
    axes[1].bar(proportions.index.astype(str), proportions.values, color=['green', 'coral'], edgecolor='white', width=0.5)
    axes[1].set_title('Class Distribution (Proportions)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Proportion')
    for i, v in enumerate(proportions.values):
        # set the proportion per bar(class)
        axes[1].text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')
    
    plt.suptitle(f'Target Variable Class Distribution: "{target}"', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# feature vs target: scatter plot, box plot per class - how each feature relates to target variable

# cross-tabulation relation from category to target variable
"""
The bigger the deviation from baseline, the more useful that category is

"""


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