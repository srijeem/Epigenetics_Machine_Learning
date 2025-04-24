
### 2nd_Feature_Selection_Wrapper_Method
### Recursive Feature Elimination Cross-Validation (RFECV)


### This script contains the only features of ICC>= 0.6  
### Performed Wrapper (RFECV) on thOSE features to reduce it to 10k
### For using those 10k for our Machine Learning Algorithms I/P features (Look into the ML algo scripts for that)

#!/usr/bin/env python
# coding: utf-8

## All imports here

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier



## Read the dataframe with only features of ICC >= 0.6
o_i = pd.read_table("path_to_input_data/input_file_1.txt")
o_i


###Read the test data
testtargets = pd.read_table("path_to_input_data/test_data.txt")
testtargets


## Split the 90% training data into 70-30 training and validation set
x_train = o_i.iloc[:, 3:]  
y_train = o_i['R2']  

# Select common features between training and test datasets
common_features = x_train.columns.intersection(testtargets.columns)
x_train = x_train[common_features]
x_test = testtargets[common_feature

# Check the shapes of the datasets
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}')

# Splitting the data into 70% training and 30% validation 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.70, random_state=42)
print(f'x_train shape after splitting: {x_train.shape}')
print(f'y_train shape after splitting: {y_train.shape}')
print(f'x_valid shape: {x_valid.shape}')
print(f'y_valid shape: {y_valid.shape}')


## Performing Wrapper Method for performing further feature selection
## The aim to perform this method was to reduce huge number of features to 10k featuress

# Define the estimator for RFE (Decision Tree in this case)
estimator = DecisionTreeClassifier()

# Start the timer
start_time = time.time()

# Perform RFECV
rfe = RFECV(estimator, step=100, min_features_to_select=10000, cv=3, scoring='accuracy', n_jobs=-1)
rfe.fit(x_train, y_train)

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print("Elapsed Time:", elapsed_time, "seconds")

# Get the selected features
selected_features_mask = rfe.support_
selected_features_indices = [i for i, selected in enumerate(selected_features_mask) if selected]
selected_features = x_train.columns[selected_features_indices]

# Ensure exactly 100 features are selected
selected_features = selected_features[:10000]

# Convert selected features to a DataFrame
selected_features_df = pd.DataFrame(selected_features, columns=['Selected Features'])

# Save the selected features to a file
selected_features_df.to_csv("path_to_output_data/selected_features_rfe_10k.csv", index=False)

#######################################################################################################

END of preparing the data for using them as I/P for our ML models

