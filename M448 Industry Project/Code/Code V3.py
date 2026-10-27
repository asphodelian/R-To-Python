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
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold, train_test_split)

################
# Data Explore #
################

Rnames = pd.read_csv(r"D:\Coding\R Storage\M448\Project Code\Resume Names.csv")
print("Dimensions of Rnames are", Rnames.shape, "\n") # dimensions

# column names
colnames = Rnames.columns.to_list()
print("The columns of Rnames are", colnames, "\n")

# summary stats
Rnames.info()
Rnames.describe()

#############
# Data Prep #
#############

y = Rnames["call"]
x = Rnames.drop("call", axis = 1)
print(f"\nClass Distribution: {Counter(y)} \n")

# downsample
rus = RandomUnderSampler(sampling_strategy = "majority", random_state = 42)
Dno, Dyes = rus.fit_resample(x, y)
print(f"After downsampling: {Counter(Dyes)} \n")

newNames = pd.concat([Dno, Dyes], axis = 1)

# summary stats
print("Dimensions of new balanced dataset:", newNames.shape, "\n")
newNames.info()
newNames.describe()
print("\n")

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
resume.info()
resume.describe()
trainR, testR = train_test_split(resume, test_size = 0.2, random_state = 42) 

featuresR = [col for col in trainR.columns if col != 'call']
formula_strR = 'call ~ ' + ' + '.join(featuresR)
glmFit1 = smf.glm(formula = formula_strR, data = trainR, family = sm.families.Binomial()).fit()

paper = resume.drop(columns = "industry") # Removing column
paper.info()
paper.describe()
trainP, testP = train_test_split(paper, test_size = 0.2, random_state = 42) 
featuresP = [col for col in trainP.columns if col != 'call']
formula_strP = 'call ~ ' + ' + '.join(featuresP)
glmFit2 = smf.glm(formula = formula_strP, data = trainP, family = sm.families.Binomial()).fit()

print(glmFit1.summary(), "\n")
print(glmFit2.summary(), "\n")

###################################
# Quadratic Discriminant Analysis #
###################################

model = QuadraticDiscriminantAnalysis()

y1 = resume["call"]
x1Raw = resume.drop("call", axis = 1)
x1 = pd.get_dummies(x1Raw, drop_first = True)
model.fit(x1,y1)

y2 = paper["call"]
x2Raw = paper.drop("call", axis = 1)
x2 = pd.get_dummies(x2Raw, drop_first = True)
model.fit(x2,y2)

# model evaluation: k-fold cross validation

cv1 = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 1)
cv2 = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 1)
cv3 = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
cv4 = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 5, random_state = 1)

score11 = cross_val_score(model, x1,y1, scoring = 'accuracy', cv = cv1, n_jobs = -1)
print("With k=5 and n=3, the model with the industry variable performed a mean accuracy of", np.mean(score11)*100, "%\n")
score12 = cross_val_score(model, x2,y2, scoring = 'accuracy', cv = cv1, n_jobs = -1)
print("With k=5 and n=3, the model with no industry variable performed a mean accuracy of", np.mean(score12)*100, "%\n")

score21 = cross_val_score(model, x1,y1, scoring = 'accuracy', cv = cv2, n_jobs = -1)
print("With k=5 and n=5, the model with the industry variable performed a mean accuracy of", np.mean(score21)*100, "%\n")
score22 = cross_val_score(model, x2,y2, scoring = 'accuracy', cv = cv2, n_jobs = -1)
print("With k=5 and n=5, the model with no industry variable performed a mean accuracy of", np.mean(score22)*100, "%\n")

score31 = cross_val_score(model, x1,y1, scoring = 'accuracy', cv = cv3, n_jobs = -1)
print("With k=10 and n=3, the model with the industry variable performed a mean accuracy of", np.mean(score31)*100, "%\n")
score32 = cross_val_score(model, x2,y2, scoring = 'accuracy', cv = cv3, n_jobs = -1)
print("With k=10 and n=3, the model with no industry variable performed a mean accuracy of", np.mean(score32)*100, "%\n")

score41 = cross_val_score(model, x1,y1, scoring = 'accuracy', cv = cv4, n_jobs = -1)
print("With k=10 and n=5, the model with the industry variable performed a mean accuracy of", np.mean(score41)*100, "%\n")
score42 = cross_val_score(model, x2,y2, scoring = 'accuracy', cv = cv4, n_jobs = -1)
print("With k=10 and n=5, the model with no industry variable performed a mean accuracy of", np.mean(score42)*100, "%\n")
