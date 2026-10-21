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

################
# Data Explore #
################

Rnames = pd.read_csv(r"D:\Coding\R Storage\M448\Project Code\Resume Names.csv")
print("Dimensions of Rnames are", Rnames.shape) # dimensions

# column names
colnames = Rnames.columns.to_list()
print("The columns of Rnames are", colnames)

# summary stats
Rnames.info()
Rnames.describe()

#############
# Data Prep #
#############

y = Rnames["call"]
x = Rnames.drop("call", axis = 1)
print(f"Class Distribution: {Counter(y)}")

# downsample
rus = RandomUnderSampler(sampling_strategy = "majority", random_state = 42)
Dno, Dyes = rus.fit_resample(x, y)
print(f"After downsampling: {Counter(Dyes)}")

newNames = pd.concat([Dno, Dyes], axis = 1)

# summary stats
print("Dimensions of new balanced dataset:", newNames.shape)
newNames.info()
newNames.describe()

names = newNames.drop(columns = ["rownames", "city"]) # Removing columns
train, test = train_test_split(names, test_size = 0.2, random_state = 42) 