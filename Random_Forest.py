#ML analysis:

### This script is performing Random Forest
### with the use of 10k features extracted from the RFECV as an input


#!/usr/bin/env python
# coding: utf-8


## All imports here
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold


### Read the test data
testtargets = pd.read_table("path_to_input_data/test_data.txt")
testtargets

## Read the 10k selected features which have been extracted from Wrapper

selected = pd.read_csv("path_to_output_data/selected_features_rfe_10k.csv")
selected


selected_features_list = selected['Selected Features'].tolist()


# Transform the datasets to keep only the selected features
x_train_selected = x_train[selected_features_list]
x_valid_selected = x_valid[selected_features_list]
x_test_selected = x_test[selected_features_list]


# Check the shapes of the transformed datasets
print(f'x_train_selected shape: {x_train_selected.shape}')
print(f'x_valid_selected shape: {x_valid_selected.shape}')
print(f'x_test_selected shape: {x_test_selected.shape}')

## Performed Random Forest

# Define the Random Forest classifier
random_forest = RandomForestClassifier(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train_selected, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the Random Forest classifier with the best hyperparameters
best_random_forest = RandomForestClassifier(random_state=42, **best_params)
best_random_forest.fit(x_train_selected, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_random_forest, x_train_selected, y_train, cv=10)
print("Cross-Validation Mean Accuracy:", cv_scores.mean())

# Evaluate the model on the validation set
y_valid_pred = best_random_forest.predict(x_valid_selected)
validation_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation Accuracy:", validation_accuracy)

# Evaluate the model on the test set
y_test = testtargets['R2']
y_test_pred = best_random_forest.predict(x_test_selected)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

## Did Hyperparameter Tuning and tested it on the
## 10% test set (Not providing with the hyperparameters which gave us the proper accuracy)

## Performed K-fold cross validation


# Define the cross-validator object (e.g., KFold with 10 folds)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store AUC for each fold
aucs = []

# Initialize figure for ROC plot
plt.figure(figsize=(8, 6))

# Perform cross-validation for each fold
for i, (train_idx, test_idx) in enumerate(cv.split(x_train_selected, y_train)):
    x_train_fold, x_test_fold = x_train_selected.iloc[train_idx], x_train_selected.iloc[test_idx]
    y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
    # Train the model
    best_random_forest.fit(x_train_fold, y_train_fold)
    
    # Predict probabilities for positive class
    y_proba = best_random_forest.predict_proba(x_test_fold)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_fold, y_proba)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    # Plot ROC curve for the fold
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (AUC = %0.2f)' % (i+1, roc_auc))

# Plot ROC curve for the validation set
plt.plot(fpr, tpr, auc_score, color='b', lw=2)

# Plot random guess line
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Random Guess')

# Set plot attributes
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest with Cross-Validation')
plt.legend(loc="lower right")

# Show plot
plt.show()


#################################################################################################

## END OF Random Forest