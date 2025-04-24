ML analysis:

### This script is performing SVM
### with the use of 10k features extracted from the RFECV as an input


#!/usr/bin/env python
# coding: utf-8


## All imports here
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
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


## Performing SVM 

# Define the SVM classifier
svm_classifier = SVC(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train_selected, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM classifier with the best hyperparameters
best_svm = SVC(random_state=42, **best_params)
best_svm.fit(x_train_selected, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_svm, x_train_selected, y_train, cv=10)
print("Cross-Validation Mean Accuracy:", cv_scores.mean())

# Evaluate the model on the validation set
y_valid_pred = best_svm.predict(x_valid_selected)
validation_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation Accuracy:", validation_accuracy)

# Evaluate the model on the test set
y_test = testtargets['R2']
test_accuracy = accuracy_score(y_test, best_svm.predict(x_test_selected))
print("Test Accuracy:", test_accuracy)



## This code of SVM is after performing Hyperparameter Tuning and 
## also tested the model on the 10% test data along with the ROC for the test set

# Define the SVM classifier
svm_classifier = SVC(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'C': [10],
    'kernel': ['rbf'],
    'gamma': ['auto']
}
# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train_selected, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM classifier with the best hyperparameters
best_svm = SVC(random_state=42, probability=True, **best_params)
best_svm.fit(x_train_selected, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_svm, x_train_selected, y_train, cv=10)
print("Cross-Validation Mean Accuracy:", cv_scores.mean())

# Evaluate the model on the validation set
y_valid_pred = best_svm.predict(x_valid_selected)
validation_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation Accuracy:", validation_accuracy)

# Evaluate the model on the test set
y_test = testtargets['R2']
y_test_pred = best_svm.predict(x_test_selected)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Predict probabilities on the test set
y_test_pred_proba = best_svm.predict_proba(x_test_selected)[:, 1]

# Calculate false positive rate, true positive rate, and threshold values for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_test_pred_proba)

# Plot ROC curve
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


## Performed K-fold cross validation for SVM

# Define the cross-validator object (e.g., KFold with 10 folds)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store AUC for each fold
aucs = []

# Initialize figure for ROC plot
plt.figure(figsize=(8, 6))

# Perform cross-validation for each fold
for i, (train_idx, test_idx) in enumerate(cv.split(x_train, y_train)):
    x_train_fold, x_test_fold = x_train.iloc[train_idx], x_train.iloc[test_idx]
    y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
    # Train the model
    best_svm.fit(x_train_fold, y_train_fold)
    
    # Get decision scores on the test fold
    decision_scores_fold = best_svm.decision_function(x_test_fold)
    
    # Convert decision scores to probability estimates using sigmoid function
    probabilities_fold = 1 / (1 + np.exp(-decision_scores_fold))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_fold, probabilities_fold)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    # Plot ROC curve for the fold
    plt.plot(fpr, tpr, lw=1,alpha=0.3, label='Fold %d (AUC = %0.2f)' % (i+1, roc_auc))

# Get decision scores on the validation set using SVM classifier
decision_scores_svm = best_svm.decision_function(x_valid)

# Convert decision scores to probability estimates using sigmoid function
probabilities_svm = 1 / (1 + np.exp(-decision_scores_svm))

# Calculate false positive rate, true positive rate, and threshold values for ROC curve
fpr_svm, tpr_svm, _ = roc_curve(y_valid, probabilities_svm)

# Calculate AUC score
auc_score_svm = roc_auc_score(y_valid, probabilities_svm)

# Plot ROC curve for the validation set
plt.plot(fpr_svm, tpr_svm, auc_score_svm, color='b', lw=2)

# Plot random guess line
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Random Guess')

# Set plot attributes
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM with Cross-Validation')
plt.legend(loc="lower right")
plt.show()


#################################################################################################

## END OF SVM
