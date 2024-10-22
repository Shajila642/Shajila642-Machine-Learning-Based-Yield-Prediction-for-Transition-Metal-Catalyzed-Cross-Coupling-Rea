# Importing the numpy library for efficient numerical computations
import numpy as np
# Importing the Random Forest model from sklearn
from sklearn.ensemble import RandomForestRegressor 
# Importing tools for splitting data and performing hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV
# Importing the copy library for making deep copies of objects
import copy
# Importing Bayesian Search for hyperparameter tuning from skopt 
from skopt import BayesSearchCV  

# Function to perform Random Forest regression with default hyperparameters
def random_forest(X, y, test_size=0.2, n_iterations=1):
    """
    A function to train a RandomForestRegressor on the input dataset 'X' and target variable 'y'.
    The function splits the data into train/test sets, trains the model, and returns predicted vs. true values.
    
    Parameters:
    - X: The feature matrix
    - y: The target variable
    - test_size: The proportion of the dataset to include in the test split
    - n_iterations: The number of times to repeat the process with different random seeds

    Returns:
    - expt_values: Array of true values of the target variable (y) for the test set
    - model_values: Array of predicted values by the model for the test set
    """
    
    model_values = []  # List to store the predicted values
    expt_values = []   # List to store the true values (actual output)
    
    # Iterate over n_iterations to get different train/test splits
    for i in range(n_iterations):
        # Split the dataset into training and testing sets
        X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        # Initialize a RandomForestRegressor model with default 100 estimators (trees)
        predictor = RandomForestRegressor(n_estimators=100)
        # Create a deep copy of the model to ensure the original remains unchanged
        pred = copy.deepcopy(predictor)
        # Train the model on the training data
        pred.fit(X_training, y_training)
        # Use the trained model to make predictions on the test set
        y_pred = pred.predict(X_test)
        
        # Append predicted values and true values to their respective lists
        model_values.extend(y_pred)
        expt_values.extend(y_test)
        
    # Return the true and predicted values as numpy arrays
    return np.array(expt_values), np.array(model_values)

# Function to perform hyperparameter tuning for RandomForest using Grid Search
def random_forest_h_tuning_grid(X, y, test_size=0.2, n_iterations=1):
    """
    A function that performs hyperparameter tuning for RandomForestRegressor using GridSearchCV.
    It returns the true values of y and model predictions after selecting the best hyperparameters.

    Parameters:
    - X: The feature matrix
    - y: The target variable
    - test_size: The proportion of the dataset to include in the test split
    - n_iterations: The number of times to repeat the process with different random seeds

    Returns:
    - true_values: Array of true values of the target variable (y) for the test set
    - model_values: Array of predicted values by the best-tuned model for the test set
    """
    
    true_values = []  # List to store true values
    model_values = []  # List to store predicted values

    # Loop to handle multiple iterations (different random seeds)
    for i in range(n_iterations):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        # Initialize a base RandomForestRegressor model
        rf = RandomForestRegressor()

        # Define the hyperparameter grid to search over for tuning
        param_grid = {
            'n_estimators': [10, 50, 100, 150],  # Number of trees in the forest
            'max_depth': [None, 10, 20],         # Maximum depth of the trees
            'min_samples_split': [2, 5],         # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4]        # Minimum number of samples required to be at a leaf node
        }

        # Perform Grid Search with cross-validation (5-fold) using the R2 score as the metric
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   scoring='r2', cv=5, n_jobs=-1)
        # Fit the grid search model to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model based on the search
        best_rf = grid_search.best_estimator_

        # Make predictions on the test set using the best model
        y_true = y_test
        y_pred = best_rf.predict(X_test)

        # Append true and predicted values to their respective lists
        true_values.extend(y_true)
        model_values.extend(y_pred)

    # Return the true and predicted values as numpy arrays
    return np.array(true_values), np.array(model_values)

# Function to perform hyperparameter tuning using Bayesian Search
def random_forest_h_tuning_bayes_strat(X, y, test_size=0.2, n_iterations=1):
    """
    A function that performs Bayesian hyperparameter tuning for a Random Forest model.
    It minimizes mean absolute error (MAE) and returns true values and model predictions.

    Parameters:
    - X: The feature matrix
    - y: The target variable
    - test_size: The proportion of the dataset to include in the test split
    - n_iterations: The number of times to repeat the process with different random seeds

    Returns:
    - true_values: Array of true values of the target variable (y) for the test set
    - model_values: Array of predicted values by the best-tuned model for the test set
    """
    
    true_values = []  # List to store true values
    model_values = []  # List to store predicted values

    # Loop to handle multiple iterations (different random seeds)
    for i in range(n_iterations):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
    
        # Initialize a base RandomForestRegressor model
        rf = RandomForestRegressor()
    
        # Define the search space for Bayesian hyperparameter tuning
        search_space = {
            'n_estimators': (10, 1000),  # Range for the number of trees
            'max_depth': (1, 32),        # Range for the maximum depth of the trees
            'min_samples_split': (2, 20),  # Range for the minimum number of samples required to split a node
            'min_samples_leaf': (1, 20)   # Range for the minimum number of samples required to be at a leaf node
        }
    
        # Perform Bayesian Search with 30 iterations, 5-fold cross-validation, and R2 score as the objective
        bayes_search = BayesSearchCV(
            rf, search_space, n_iter=30, cv=5, scoring='r2', n_jobs=-1
        )
        # Fit the Bayesian search model to the training data
        bayes_search.fit(X_train, y_train)
    
        # Get the best hyperparameters from the search
        best_params = bayes_search.best_params_
    
        # Train the best Random Forest model with the optimal hyperparameters on the training set
        best_rf = RandomForestRegressor(**best_params)
        best_rf.fit(X_train, y_train)
    
        # Make predictions on the test set
        y_true = y_test
        y_pred = best_rf.predict(X_test)
    
        # Append true and predicted values to their respective lists
        true_values.extend(y_true)
        model_values.extend(y_pred)

    # Return the true and predicted values as numpy arrays
    return np.array(true_values), np.array(model_values)
