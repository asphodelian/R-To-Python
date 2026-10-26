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

##########
# Models #
##########

resume = newNames.drop(columns = ["rownames", "name", "city",
                                 "quality", "volunteer", "military", 
                                 "email", "computer", "special", 
                                 "minimum", "requirements", "reqcomm", 
                                 "reqeduc", "reqcomp", "reqorg", "jobs",
                                 "gender", "ethnicity", "school", "college",
                                 "equal", "wanted", "reqexp"]) # Removing columns
trainR, testR = train_test_split(resume, test_size = 0.2, random_state = 42) 

featuresR = [col for col in trainR.columns if col != 'call']
formula_strR = 'call ~ ' + ' + '.join(featuresR)
glmFit1 = smf.glm(formula = formula_strR, data = trainR, family = sm.families.Binomial()).fit()

paper = resume.drop(columns = "industry") # Removing column
trainP, testP = train_test_split(paper, test_size = 0.2, random_state = 42) 
featuresP = [col for col in trainP.columns if col != 'call']
formula_strP = 'call ~ ' + ' + '.join(featuresP)
glmFit2 = smf.glm(formula = formula_strP, data = trainP, family = sm.families.Binomial()).fit()

print(glmFit1.summary())
print(glmFit2.summary())

###################################
# Quadratic Discriminant Analysis #
###################################

model = QuadraticDiscriminantAnalysis()

y1 = resume["call"]
x1Raw = resume.drop("call", axis = 1)
x1 = pd.get_dummies(x1Raw, drop_first = True)
model.fit(x1,y1)
print("QDA Model successfully fit!")
