import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# SHAPE AND DATA TYPES
def type_shape_stats(df: pd.DataFrame):
    print("dataframe shape:", df.shape)
    print("dataframe types:", df.dtypes)

def descriptive_stats(df: pd.DataFrame):
    print(df.describe().T)


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

# Correlation between missing values at the same time in different features
# 1. Does 'unknown' occurs at the same time in multiple columns? Is there any connection between unknown in one column and another?
"""
If the same rows tend to have "unknown" in multiple columns simultaneously,
that pattern itself is informative — it might mean a specific subgroup of people refused to answer, which could correlate with the target.
"""
def unknown_cooccurrence_matrix(df):
    """
    For each pair of categorical columns, count how many rows
    have 'unknown' in BOTH columns simultaneously.
    Returns a heatmap showing the co-occurrence counts (and %).
    """
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    unknown_mask = (df[cat_cols] == 'unknown').astype(int)

    # Co-occurrence matrix: dot product gives count of rows
    # where both columns are 'unknown' at the same time
    cooc = unknown_mask.T @ unknown_mask  # shape: (n_cols, n_cols)

    # Convert to % of total rows for easier interpretation
    cooc_pct = (cooc / len(df)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cooc, annot=True, fmt='d', cmap='YlOrRd',
                ax=axes[0], linewidths=0.5, square=True,
                cbar_kws={'label': 'rows'})
    axes[0].set_title('Unknown co-occurrence (row counts)')
    plt.xticks(rotation=45, ha='right')
    sns.heatmap(cooc_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=axes[1], linewidths=0.5, square=True,
                cbar_kws={'label': '% of rows'})
    
    axes[1].set_title('Unknown co-occurrence (% of dataset)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return cooc, cooc_pct

# education, housing, loan: check behavior with target variable 
def plot_unknown_vs_known(df: pd.DataFrame, target_col: str, cols: list):
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 5))
    
    target = df[target_col]
    overall = target.mean() * 100
    
    for ax, col in zip(axes, cols):
        is_unknown = df[col] == 'unknown'
        rate_unknown = target[is_unknown].mean() * 100
        rate_known   = target[~is_unknown].mean() * 100
        
        sns.barplot(x=['unknown', 'known'], y=[rate_unknown, rate_known],
                    palette=['#e74c3c', '#3498db'], ax=ax)
        
        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.set_ylabel('Subscription rate (%)' if ax == axes[0] else '')
        ax.set_ylim(0, 25)
        
        for bar, rate in zip(ax.patches, [rate_unknown, rate_known]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        ax.axhline(y=overall, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(1.1, overall + 0.3, f'avg {overall:.1f}%', fontsize=9, color='gray')
    
    sns.despine()
    plt.suptitle('Subscription rate: unknown vs known', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Skewness, kurtosis
def skewness_kurtosis(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Skewness: measures asymmetry of distribution
        - 0          : perfectly symmetric
        - > 1 or < -1: highly skewed -> consider log transform
        - 0.5 to 1   : moderately skewed
 
    Kurtosis: measures tail heaviness
        - 3 (excess=0): normal distribution
        - > 3          : heavy tails, more outliers
        - < 3          : light tails
    """
    results = []
    for col in features:
        results.append({
            'feature': col,
            'skewness': round(df[col].skew(), 3),
            'kurtosis': round(df[col].kurtosis(), 3),
            'skew_flag': 'High skew' if abs(df[col].skew()) > 1 else ''
        })
 
    return pd.DataFrame(results).sort_values('skewness', key=abs, ascending=False)

# percentiles
def percentiles_summary(df: pd.DataFrame, features: list):
    """Returns 1, 5, 25, 50, 75, 95, 99 percentiles per feature."""
    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    return df[features].quantile(percentiles).T.rename(
        columns={p: f'p{int(p*100)}' for p in percentiles}
    )
 
# DISTRIBUTIONS
def plot_histograms(df: pd.DataFrame, features: list, bins: int = 30):
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        axes[i].hist(df[col].dropna(), bins=bins, edgecolor='white', color='steelblue', alpha=0.7)
        axes[i].set_title(f'{col}\nskew={df[col].skew():.2f}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# OUTLIERS
# histograms, boxplots, violin plots, Z-score, IQR plot
# conduct normality tests: Shapiro-Wilk, D'Agostino-Pearson, plot Q-Q plots

# test for specific distribution: conduct Anderson-Darling

# CATEGORICAL ANALYSIS

# CORRELATIONS
# heatmaps Cramér's V(only for categorical variables)
"""
Cramér's V - variable association
Chi-Square test
Cramér's V = 
No association with target → likely useless feature
High association with target + low association with other features → ideal feature
High association with another feature → redundant, pick one
"""


# MULTICOLLINEARITY DETECTION: VIF
# build two correlation matrices: correlation feature vs. feature, correlation feature vs. target
# if feature is highly correlated with another feature -> bad 
# if feature is 
# If age and experience are 95% correlated, they're essentially saying the same thing. This is called multicollinearity 

# correlation between features: if 0.7-0.9 -> check for multicollinearity
def features_correlation(df: pd.DataFrame, corr_features: list):
  # prints out the correlation between features 
  # features with corelation 0.7-0.9 are considered high correlation

  corr_matrix = df[corr_features].corr()
  plt.figure(figsize=(12, 10))
  sns.heatmap(corr_matrix,
            annot=True,  # show correlation values
            fmt='.1f',   # Format numbers to 1 decimal place
            cmap='YlGnBu',
            center=0,    # Center at 0
            square=True, # cells square-shaped
            linewidths=1,  
            cbar_kws={'label': 'Correlation coefficient'})
  plt.xticks(rotation=45, ha='right')
  plt.title('Correlation between features', fontsize=16, pad=20)
  plt.tight_layout()
  plt.show()

def high_corr_features(df: pd.DataFrame, corr_features: list, threshold: float = 0.7):

    corr_matrix = df[corr_features].corr()
    # avoid duplicate pairs : (A, B) == (B, A) 
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        (col, row, upper_triangle.loc[row, col])
        for col in upper_triangle.columns
        for row in upper_triangle.index
        if abs(upper_triangle.loc[row, col]) >= threshold
    ]
    
    if not high_corr_pairs:
        print(f"No feature pairs with correlation >= {threshold}")
        return
    
    print(f"Highly correlated feature pairs\n")
    for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1} <-> {feat2}: {corr_val:.2f}")


# print out pairs which are highly correlated
def high_corr_features(df: pd.DataFrame, corr_features: list, threshold: float = 0.7):

    corr_matrix = df[corr_features].corr()
    
    # avoid duplicate pairs : (A, B) == (B, A) 
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        (col, row, upper_triangle.loc[row, col])
        for col in upper_triangle.columns
        for row in upper_triangle.index
        if abs(upper_triangle.loc[row, col]) >= threshold
    ]
    
    if not high_corr_pairs:
        print(f"No feature pairs with correlation >= {threshold}")
        return
    
    print(f"Highly correlated feature pairs (|corr| >= {threshold}):\n")
    for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1} <-> {feat2}: {corr_val:.2f}")

def check_multicollinearity(df: pd.DataFrame, features: list):
    # variance inflamation factor is used to measure multicollinearity
    X = df[features].dropna()
    
    vif = pd.DataFrame({
        'feature': features,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(features))]
    })
    
    vif['severity'] = pd.cut(vif['VIF'], 
                              bins=[0, 5, 10, float('inf')], 
                              labels=['Acceptable', 'High', 'Severe'])
    
    return vif.sort_values('VIF', ascending=False).reset_index(drop=True)
    


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

