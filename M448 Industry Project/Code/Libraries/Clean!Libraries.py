#############
# Libraries #
#############

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler # imbalanced dataset
import itertools # memory-efficient tools for iterators & loops
import numpy as np # efficient num computing & multidim arrays
import pandas as pd # data manipulation, analysis, cleaning
import scipy # scientific, math, engineering compute
from sklearn.datasets import fetch_openml # machine learning datasets

# replaces ggplot2
import plotnine as plot
import seaborn as sea 
import matplotlib.pyplot as plt

# stats models
import statsmodels.api as sm 
import statsmodels.formula.api as smf # specify stat models using R-style formula strings a& pandas dfs

# Scikit-learning Models & tools

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, #LDA
                                           QuadraticDiscriminantAnalysis #QDA
                                           )
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SequentialFeatureSelector, RFE)
from sklearn.linear_model import (ElasticNet, #glmnet 
                                  LinearRegression, LogisticRegression)
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.model_selection import (cross_val_score, train_test_split)




