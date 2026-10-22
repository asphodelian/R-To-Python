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

names = newNames.drop(columns = ["rownames", "name", "city"]) # Removing columns
train, test = train_test_split(names, test_size = 0.2, random_state = 42) 
#print(names.head())

##############
# Base Model #
##############

features = [col for col in train.columns if col != 'call']
formula_str = 'call ~ ' + ' + '.join(features)
glmFit = smf.glm(formula=formula_str, data=train, family=sm.families.Binomial()).fit()
#print(glmFit.summary())

"""
Everything is significant, which is strange...
Doing an odds ratio to see what coefficients matters
"""

#odds_ratios = np.exp(glmFit.params)
"""
modelMetrics = pd.DataFrame({
    "Coeff (Log-Odds)": glmFit.params,
    "Odds Ratio": odds_ratios,
    "P-value": glmFit.pvalues
})
"""
#print(modelMetrics)

"""
The design matrix is rank-deficient, so there's complete separation/perfect multi-collinearity.
Some odds ratios are infinity in the print also...
Removal: quality, volunteer, military, 
         email, computer, special, minimum, 
         requirements, reqcomm, reqeduc, reqcomp, 
         reqorg, jobs
Iffy = industry (same problem I had in R)
"""

resume = names.drop(columns = ["quality", "volunteer", "military", 
                               "email", "computer", "special", 
                               "minimum", "requirements", "reqcomm", 
                               "reqeduc", "reqcomp", "reqorg", "jobs"]) # Removing columns
train, test = train_test_split(resume, test_size = 0.2, random_state = 42) 
features = [col for col in train.columns if col != 'call']
formula_str = 'call ~ ' + ' + '.join(features)
glmFit1 = smf.glm(formula = formula_str, data = train, family = sm.families.Binomial()).fit()
print(glmFit1.summary())

"""
To Drop:
- gender
- ethnicity
- school
- college 
- equal
- wanted
- reqexp
- industry
"""

paper = resume.drop(columns = ["gender", "ethnicity", "school", "college",
                               "equal", "wanted", "reqexp", "industry"]) # Removing columns
train, test = train_test_split(paper, test_size = 0.2, random_state = 42) 
features = [col for col in train.columns if col != 'call']
formula_str = 'call ~ ' + ' + '.join(features)
glmFit2 = smf.glm(formula = formula_str, data = train, family = sm.families.Binomial()).fit()
print(glmFit2.summary())