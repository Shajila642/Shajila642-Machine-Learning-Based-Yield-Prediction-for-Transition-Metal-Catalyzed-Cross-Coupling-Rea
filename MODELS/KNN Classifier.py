import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


def preprocess_yield(y):
    """
    Preprocess the target variable `y` into four distinct yield categories.
    
    Args:
    - y (array-like): The input array of yield values.
    
    Returns:
    - classes (np.array): A NumPy array of yield categories based on the following ranges:
        - "POOR YIELD": if yield < 25
        - "BELOW AVERAGE YIELD": if 25 <= yield < 50
        - "AVERAGE YIELD": if 50 <= yield < 75
        - "GOOD YIELD": if yield >= 75
    """
    classes = []  # List to store the categories for each yield
    for yi in y:
        if yi < 25:
            classes.append("POOR YIELD")
        elif yi < 50:
            classes.append("BELOW AVERAGE YIELD")
        elif yi < 75:
            classes.append("AVERAGE YIELD")
        else:
            classes.append("GOOD YIELD")
    
    return np.array(classes)  # Return categories as a NumPy array


def knn_classification_HPT(X, y, test_size=0.2, n_iterations=10):
    """
    Perform K-Nearest Neighbors (KNN) classification with Hyperparameter Tuning (HPT) using GridSearchCV.
    
    Args:
    - X (array-like): Feature matrix (input data).
    - y (array-like): Target variable (reaction yields).
    - test_size (float): Fraction of data to use for testing (default: 0.2).
    - n_iterations (int): Number of random splits for train-test (default: 10).
    
    Returns:
    - true_values (np.array): True class labels from the test set.
    - model_values (np.array): Predicted class labels from the model.
    """
    
    # Initialize lists to store true values and model predictions across iterations
    true_values = []
    model_values = []
    
    # Preprocess the target variable into categories (e.g., "GOOD YIELD", "POOR YIELD")
    y = preprocess_yield(y)

    # Perform multiple iterations to get different train-test splits
    for i in range(n_iterations):
        # Split the data into training and testing sets randomly (changing random_state each iteration)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i)

        # Initialize K-Nearest Neighbors (KNN) classifier
        knn = KNeighborsClassifier()

        # Define the hyperparameter search space for GridSearchCV
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider in KNN
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Search algorithm
            'weights': ['uniform', 'distance'],  # Weighting scheme: equal or distance-based
            'p': [1, 2]  # Distance metric: Manhattan (1) or Euclidean (2)
        }

        # Perform grid search with cross-validation to find the best hyperparameters
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)  # Fit the model using the training data

        # Retrieve the best hyperparameters from the grid search
        best_params = grid_search.best_params_

        # Train the final KNN classifier using the best hyperparameters
        best_knn = KNeighborsClassifier(**best_params)
        best_knn.fit(X_train, y_train)  # Train on the full training set

        # Make predictions on the test set
        y_pred = best_knn.predict(X_test)

        # Store the true and predicted values for later analysis
        true_values.extend(y_test)
        model_values.extend(y_pred)

    return np.array(true_values), np.array(model_values)  # Return the true and predicted labels


def knn_classification(X, y, test_size=0.2, n_iterations=5):
    """
    Perform K-Nearest Neighbors (KNN) classification without hyperparameter tuning.
    
    Args:
    - X (array-like): Feature matrix (input data).
    - y (array-like): Target variable (reaction yields).
    - test_size (float): Fraction of data to use for testing (default: 0.2).
    - n_iterations (int): Number of random splits for train-test (default: 5).
    
    Returns:
    - true_values (np.array): True class labels from the test set.
    - model_values (np.array): Predicted class labels from the model.
    """
    
    # Initialize lists to store true values and model predictions across iterations
    true_values = []
    model_values = []
    
    # Preprocess the target variable into categories (e.g., "GOOD YIELD", "POOR YIELD")
    y = preprocess_yield(y)

    # Perform multiple iterations to get different train-test splits
    for i in range(n_iterations):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i)

        # Initialize K-Nearest Neighbors (KNN) classifier with specific hyperparameters
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)  # Set n_neighbors=5, Euclidean distance (p=2)

        # Train the KNN classifier on the training data
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)

        # Store the true and predicted values for later analysis
        true_values.extend(y_test)
        model_values.extend(y_pred)

    return np.array(true_values), np.array(model_values)  # Return the true and predicted labels

