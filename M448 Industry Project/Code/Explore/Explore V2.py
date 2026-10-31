#############
# Libraries #
#############

from imblearn.under_sampling import RandomUnderSampler # imbalanced dataset
import itertools # memory-efficient tools for iterators & loops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
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

#################
# Visualization #
#################

# Pair Plots #
plot_cols = ["experience", "jobs", "gender",
             "ethnicity", "quality", "industry",
             "call"]
df_plot = Rnames[plot_cols].copy()

le = LabelEncoder()
categorical_cols = ["gender", "ethnicity", "quality", "industry"]

for col in categorical_cols:
    df_plot[col] = le.fit_transform(df_plot[col].astype(str))

sns.set_theme(style="ticks")
ax = sns.pairplot(
    df_plot,
    hue="call",
    palette="Set1",
    diag_kind="kde",
    plot_kws={"alpha": 0.6, "s": 30},  # Adds transparency to handle overlapping points
)

plt.suptitle("Pair Plot of Key Resume Features & Callbacks", y=1.02, fontsize=14)
sns.move_legend(
    ax, "lower center", bbox_to_anchor=(0.5, 1), title=None, frameon=False
)
plt.tight_layout()
plt.show()

# Historgrams #
plot_cols = ["experience", "jobs", "gender", "ethnicity"]
plt.figure(figsize=(12, 10))

for i, feature in enumerate(plot_cols):
    plt.subplot(2, 2, i + 1)
    
    sns.histplot(data = Rnames, x = feature, hue = 'call', 
                 kde = (Rnames[feature].dtype != 'O'),
                 multiple = "dodge", shrink = 0.8)
    plt.title(f'{feature.capitalize()} Distribution by Callback')
    plt.xlabel(feature.capitalize())

plt.tight_layout()
plt.show()

# Correlation Heatmap #
subset = pd.get_dummies(df_plot, drop_first = True)
correlation_matrix = subset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            linewidths=0.5, fmt = ".2f", vmin= -1, vmax = 1)
plt.title("Correlation Heatmap of Key Resume Features & Callbacks")
plt.show()