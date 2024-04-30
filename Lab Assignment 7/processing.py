from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np  


data = pd.read_csv('Data/parkinsson_data.csv')

# Load your dataset
data = pd.read_csv('Data/parkinsson_data.csv')

# Exclude non-numeric columns and 'status' column
X = data.drop(['status', 'name'], axis=1)  
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
mlp = MLPRegressor()
svm = SVR()
decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor()
catboost = CatBoostRegressor()
adaboost = AdaBoostRegressor()
xgboost = XGBRegressor()
linear_regression = LinearRegression()

# Define the hyperparameter search space (You may need to adjust this based on the models)
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (200,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

param_grid_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
}

# Perform hyperparameter tuning using RandomizedSearchCV
mlp_cv = RandomizedSearchCV(mlp, param_grid_mlp, cv=5)
svm_cv = RandomizedSearchCV(svm, param_grid_svm, cv=5)

# Fit the models
mlp_cv.fit(X_train_scaled, y_train)
svm_cv.fit(X_train_scaled, y_train)

# Get the best hyperparameters
mlp_best_params = mlp_cv.best_params_
svm_best_params = svm_cv.best_params_

# Print the best hyperparameters
print("MLP Best Parameters:", mlp_best_params)
print("SVM Best Parameters:", svm_best_params)

# Perform cross-validation for MLP
mlp_scores = cross_val_score(mlp_cv.best_estimator_, X_train_scaled, y_train, cv=5)
mlp_mean_score = np.mean(mlp_scores)

# Perform cross-validation for SVM
svm_scores = cross_val_score(svm_cv.best_estimator_, X_train_scaled, y_train, cv=5)
svm_mean_score = np.mean(svm_scores)

# Define the other regressors
decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor()
catboost = CatBoostRegressor()
adaboost = AdaBoostRegressor()
xgboost = XGBRegressor()
linear_regression = LinearRegression()

# Perform cross-validation for other regressors
decision_tree_scores = cross_val_score(decision_tree, X_train_scaled, y_train, cv=5)
random_forest_scores = cross_val_score(random_forest, X_train_scaled, y_train, cv=5)
catboost_scores = cross_val_score(catboost, X_train_scaled, y_train, cv=5)
adaboost_scores = cross_val_score(adaboost, X_train_scaled, y_train, cv=5)
xgboost_scores = cross_val_score(xgboost, X_train_scaled, y_train, cv=5)
linear_regression_scores = cross_val_score(linear_regression, X_train_scaled, y_train, cv=5)

decision_tree_mean_score = np.mean(decision_tree_scores)
random_forest_mean_score = np.mean(random_forest_scores)
catboost_mean_score = np.mean(catboost_scores)
adaboost_mean_score = np.mean(adaboost_scores)
xgboost_mean_score = np.mean(xgboost_scores)
linear_regression_mean_score = np.mean(linear_regression_scores)

# Create a table of results
results = pd.DataFrame({
    'Regressor': ['MLP', 'SVM', 'Decision Tree', 'Random Forest', 'CatBoost', 'AdaBoost', 'XGBoost', 'Linear Regression'],
    'Mean Score': [mlp_mean_score, svm_mean_score, decision_tree_mean_score, random_forest_mean_score, catboost_mean_score, adaboost_mean_score, xgboost_mean_score, linear_regression_mean_score]
})

print(results)
