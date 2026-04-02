from statsmodels.stats.outliers_influence import variance_inflation_factor

# vif to check multicollinearity
# statistical tests 1) to check for normal distribution
# statistical test 2: check for specific distribution of your data



"""
1. Normality tests: is the data normally distributed? 
2. Variance Equality Tests: do my groups have equal variance? - it's only for categorical variables
Levene's test — robust to non-normality
Bartlett's test — assumes normality
You need it when you want to compare categorical variables with some continuous variable: 

3. 
"""

# INTERACTION TERMS check
"""
interaction term: the effect of feature A on the target depends on the value of feature B.
if you spot an interaction term -> for linear models you need to create manually an interaction term
"""




