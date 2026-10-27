#############
# Libraries #
#############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

########
# Data #
########
Rnames = pd.read_csv(r"D:\Coding\R Storage\M448\Project Code\Resume Names.csv")
print("Dimensions of Rnames are", Rnames.shape, "\n") # dimensions

# column names
colnames = Rnames.columns.to_list()
print("The columns of Rnames are", colnames, "\n")

# summary stats
Rnames.info()
Rnames.describe()