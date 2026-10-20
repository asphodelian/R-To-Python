#############
# Libraries #
#############
import pandas as pd
# models
from sklearn.linear_model import ElasticNet #glmnet
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #QDA
# best subset selection alts #
# greedy best approx
from sklearn.feature_selection import SequentialFeatureSelector 
from sklearn.linear_model import LogisticRegression
# recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# custom true best subset select
import itertools
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
#replaces ggplot2
import plotnine as plot
import seaborn as sea 
import matplotlib.pyplot as plt
# replaces MASS
import statsmodels as stats 
# replaces stats
import scipy 

#############
# Data Prep #
#############

Rnames = pd.read_csv(r"D:\Coding\R Storage\M448\Project Code\Resume Names.csv")
print("Dimensions of Rnames are", Rnames.shape) # dimensions

# column names
colnames = Rnames.columns.to_list()
print("The columns of Rnames are", colnames)

# summary stats
print(Rnames.head())
Rnames.info()
Rnames.describe()

# Downsampling
sumCall = Rnames["call"].value_counts()
color = ["firebrick", "darkgreen"]
plt.pie(sumCall, labels = sumCall.index, colors= color, autopct='%1.1f%%')
plt.title("Callback Distribution")
plt.show()