# ML analysis:

### This script is performing XGBoost 
### Have the 10k features extracted from the RFECV as an input
### Performed highest feature extraction through XGBoost and used them to train the XGBoost model
#### Caculated Youden's Index to find out the balance between sensitivity and specificity


#!/usr/bin/env python
# coding: utf-8


## All imports here
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

## Performed XGBoost

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_selected, y_train)

# Extract feature importances
feature_importances = pd.DataFrame({
    'Feature': x_train_selected.columns,
    'Importance': xgb_model.feature_importances_
})

# Ensure 'Random_Probe' exists
if 'Random_Probe' in feature_importances['Feature'].values:
    # Get the importance of 'Random_Probe'
    random_probe_importance = feature_importances.loc[
        feature_importances['Feature'] == 'Random_Probe', 'Importance'
    ].values[0]

    # Sort features by importance
    feature_importances_sorted = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Highlight the probes relative to Random_Probe
    feature_importances_sorted['Category'] = feature_importances_sorted['Importance'].apply(
        lambda x: 'Higher' if x > random_probe_importance else
                  'Equal' if x == random_probe_importance else
                  'Lower'
    )

    # Extract the features with higher importance than Random_Probe
    higher_importance_features = feature_importances_sorted[
        feature_importances_sorted['Category'] == 'Higher'
    ].copy()

    # Retrieve beta values for these features from the original dataset
    beta_values = x_train_selected[higher_importance_features['Feature']].mean().reset_index()
    beta_values.columns = ['Feature', 'Beta_Value']

    # Merge beta values with feature importance
    higher_importance_features = higher_importance_features.merge(beta_values, on='Feature', how='left')

    # Save to Excel
    output_filename = "feature_importances.xlsx"
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        higher_importance_features[['Feature', 'Importance', 'Beta_Value']].to_excel(writer, sheet_name='Higher_Importance', index=False)
        feature_importances_sorted.to_excel(writer, sheet_name='All_Features', index=False)

    print(f"Feature importances and beta values saved to {output_filename}")

    # Select features with importance greater than Random_Probe
    selected_features = feature_importances[
        feature_importances['Importance'] > random_probe_importance
    ]['Feature']

    # Count the number of probes with higher importance than Random_Probe
    higher_importance_count = len(selected_features)
    print(f"Number of probes with higher importance than Random_Probe: {higher_importance_count}")

    # Subset the datasets to keep only selected features
    x_train_filtered = x_train_selected[selected_features]
    x_valid_filtered = x_valid_selected[selected_features]
    x_test_filtered = x_test_selected[selected_features]
    

    # Load the saved Excel sheets
    df_higher_importance = pd.read_excel(output_filename, sheet_name="Higher_Importance")
    # df_all_features = pd.read_excel(output_filename, sheet_name="All_Features")

    # Display first few rows of the 'Higher_Importance' sheet
    print(df_higher_importance.head())

else:
    raise ValueError("'Random_Probe' is not present in the feature importances.")

### XGBoost with parallelization


# Define the XGBoost model with parallelization
xgb_classifier = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    
    n_jobs=-1  # Use all available CPU cores for parallelization
)

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [3] ,  # Depth of trees
    'learning_rate': [0.01],  # Learning rate
    'n_estimators': [50],  # Number of trees
    'subsample': [0.6],  # Subsampling ratio
    'colsample_bytree': [0.3],  # Column sampling ratio
    'gamma': [0.0],
     'min_child_weight': [1],
    'reg_alpha': [0.5],
    'reg_lambda': [1.5],
    'scale_pos_weight': [1]
}


# Perform grid search cross-validation to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1  # Parallelize GridSearchCV computations
)
grid_search.fit(x_train_filtered, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the XGBoost classifier with the best hyperparameters 
best_xgb = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric= 'logloss',  # 
    n_jobs=-1,  # Parallelize XGBoost tree building
    **best_params
)
best_xgb.fit(x_train_filtered, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_xgb, x_train_filtered, y_train, cv=10, n_jobs=-1)  # Parallelize CV

# Evaluate the model on the validation set
y_valid_pred = best_xgb.predict(x_valid_filtered)
validation_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation Accuracy:", validation_accuracy)

# Evaluate the model on the test set
y_test = testtargets['R2']
y_test_pred = best_xgb.predict(x_test_filtered)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Get predicted probabilities for the test set (for ROC calculation)
y_test_pred_proba_class_1 = best_xgb.predict_proba(x_test_filtered)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba_class_1)

# Compute ROC-AUC score
roc_auc = roc_auc_score(y_test, y_test_pred_proba_class_1)
print(f"ROC-AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

## Performing K-fold cross validation

#x_train_filtered, y_train, and x_valid_filtered are already defined

# Initialize the XGBoost model
xgb_classifier = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1  # Use all available CPU cores for parallelization
)

# Define KFold cross-validation (K =10 folds)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize figure for ROC plot
plt.figure(figsize=(8, 6))

# Perform K-fold cross-validation
for i, (train_idx, test_idx) in enumerate(cv.split(x_train_filtered, y_train)):
    # Split the data into training and testing sets for this fold
    x_train_fold, x_test_fold = x_train_filtered.iloc[train_idx], x_train_filtered.iloc[test_idx]
    y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
    # Train the XGBoost model on the current fold
    xgb_classifier.fit(x_train_fold, y_train_fold)
    
    # Get predictions for training data and testing data
    train_pred = xgb_classifier.predict(x_train_fold)
    test_pred = xgb_classifier.predict(x_test_fold)
    
    # Calculate training accuracy for this fold
    train_accuracy = accuracy_score(y_train_fold, train_pred)
    
    # Calculate validation accuracy for this fold
    valid_accuracy = accuracy_score(y_test_fold, test_pred)
    
    # Get predicted probabilities for ROC curve (test data)
    y_test_pred_proba = xgb_classifier.predict_proba(x_test_fold)[:, 1]
    
    # Calculate ROC curve and AUC for this fold
    fpr, tpr, _ = roc_curve(y_test_fold, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve for the fold
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    # Print the results for each fold
    print(f"Fold {i+1}:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {valid_accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.2f}")
    print("-" * 40)

# Evaluate the final model on the validation set 
xgb_classifier.fit(x_train_filtered, y_train)
y_valid_pred_proba = xgb_classifier.predict_proba(x_valid_filtered)[:, 1]

# Calculate ROC-AUC for the validation set
fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_pred_proba)
roc_auc_valid = roc_auc_score(y_valid, y_valid_pred_proba)

# Plot ROC curve for the validation set
plt.plot(fpr_valid, tpr_valid, color='b', lw=2, label=f'Validation Set (AUC = {roc_auc_valid:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random classifier")

# Set plot attributes
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost with Cross-Validation')
plt.legend(loc='lower right')
plt.show()


## Confusion Metrics

# Compute the confusion matrix for the test set with the updated threshold
confusion_matrix_result = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=[0, 1])
cm_display.plot()
plt.title('Confusion Matrix for Test Set with Default Threshold')
plt.show()



## Calculating Youden's Index to find a balance between sensitivity and specificity 


# Calculate Youden's Index for each threshold
youden_index = tpr - fpr

# Find the threshold that maximizes Youden's Index
optimal_threshold = thresholds[youden_index.argmax()]
print(f"Optimal Threshold based on Youden's Index: {optimal_threshold}")

# Now, predict using the optimal threshold
y_test_pred_optimal = (y_test_pred_proba_class_1 >= optimal_threshold).astype(int)


# Confusion Metrics with the optimum threshold obtained from Youden's Index

# Use the new threshold to classify predictions
threshold = 0.2488  # New threshold
y_test_pred = (y_test_pred_proba_class_1 >= threshold).astype(int)

# Compute the confusion matrix for the test set with the updated threshold
confusion_matrix_result = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=[0, 1])
cm_display.plot()
plt.title('Confusion Matrix for Test Set with Threshold = {:.4f}'.format(threshold))
plt.show()

#################################################################################################

## END OF XGBOOST
