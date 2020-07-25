import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

full_df = pd.read_csv('ed2017_data.csv',low_memory=False)
full_df = full_df.fillna(value=-1)
print(full_df.head())

## PREPROCESSING ##

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# take a subset of the dataframe - could potentially drop AGER as well (which is age for people under 1 years)
#subset_df = full_df.iloc[:,list(range(36)) + list(range(41,46)) + list(range(60,81))].drop(columns=["BLANK1"]) # this one large
subset_df = full_df.iloc[:,:36].drop(columns=["BLANK1"])

# for now remove negative values -9 and -7 from WAITTIME
subset_df = subset_df[subset_df["WAITTIME"] != -9]
subset_df = subset_df[subset_df["WAITTIME"] != -7]

# set all negative values in the dataframe to the median of the attribute - this is probably not a great way of 
# processing the data, so may need to change later on.
subset_df[subset_df<0] = subset_df[subset_df<0].apply(lambda x: x*subset_df[subset_df>0][x.name].median()/x)

# Discretise numeric attributes into bins (helps to reduce noise in data)
mod_df = subset_df.apply(lambda x: (x/100).astype(int)*100+30 if x.name in ["ARRTIME"] else x) # '+30' as using minutes
mod_df = subset_df.apply(lambda x: (x/5).astype(int)*5 if x.name in ["WAITTIME"] else x)
cols = ["AGE","TEMPF","PULSE","BPSYS","BPDIAS"]
mod_df = mod_df.apply(lambda x: (x/3).astype(int)*3+1.5 if x.name in cols else x)

# set all negative values in dataframe to mode of the class??
# mod_df.groupby("WAITTIME").mean()

# change data so that all independent variables are binary (0 and 1)
final_df = pd.get_dummies(mod_df, columns=list(mod_df.drop(columns="WAITTIME")))

# Splitting the data into test and training sets
X = final_df.iloc[:,1:].values
y = final_df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

# Using SMOTE to fix imbalanced data set - create synthetic samples of minority class 
#smote = SMOTE('minority',random_state=2)
#X_res, y_res = smote.fit_sample(X_train, y_train)

# we have an imbalanced dataset, need to look at confusion matrices later
plt.hist(final_df["WAITTIME"],bins=100)
plt.xlabel('Waiting Time (minutes)')
plt.ylabel('Frequency')
plt.title('Waiting Time Frequencies From NHAMCS Data')
plt.show()

unique, counts = np.unique(y_test, return_counts = True)
print("Accuracy for majority class model: " + str(counts[0]/sum(counts)))