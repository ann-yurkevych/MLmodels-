# classifier constructors
from preprocessing import *

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier 
from sklearn.tree import DecisionTreeClassifier

def base_Logistic_regression()