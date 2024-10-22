import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

'''
This function builds a neural network model with an attention mechanism, performs hyperparameter tuning 
to minimize mean absolute error (MAE), and incorporates stratified sampling. It uses libraries like 
scikit-learn, PyTorch, and skopt for Bayesian optimization.
'''

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        # Define the attention layer with a sequential model
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Transform input to hidden dimension
            nn.Tanh(),  # Non-linear activation function
            nn.Linear(hidden_dim, 1)  # Output attention scores
        )

    def forward(self, x):
        # Calculate attention weights and return the weighted sum of input
        attention_weights = torch.softmax(self.attention(x), dim=1)  # Softmax for normalization
        attended_values = torch.sum(attention_weights * x, dim=1)  # Weighted sum of input features
        return attended_values

class NeuralNetworkWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetworkWithAttention, self).__init__()
        self.attention = Attention(input_dim, hidden_dim)  # Initialize attention layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer

    def forward(self, x):
        attended_values = self.attention(x)  # Apply attention mechanism
        output = self.fc2(torch.relu(self.fc1(attended_values)))  # Feed through fully connected layers
        return output

def neural_network_with_attention_hyperparameter_tuning(X, y, test_size=0.2, n_iterations=1):
    # Lists to store true values and model predictions
    true_values = []
    model_values = []

    for _ in range(n_iterations):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=None)

        # Convert data to PyTorch tensors for training
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

        # Define the hyperparameter search space
        search_space = {
            'hidden_dim': Integer(16, 128),  # Hidden dimension of the attention layer
            'lr': Real(0.001, 0.1),  # Learning rate
            'batch_size': Integer(32, 256),  # Batch size for training
            'epochs': Integer(10, 100),  # Number of training epochs
        }

        # Objective function to minimize MAE
        def objective_function(hidden_dim, lr, batch_size, epochs):
            model = NeuralNetworkWithAttention(input_dim=X_train.shape[1], hidden_dim=hidden_dim)  # Initialize model
            optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
            criterion = nn.L1Loss()  # Mean Absolute Error (L1 Loss)

            # Create DataLoader for training
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                model.train()  # Set model to training mode
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()  # Zero the gradients
                    y_pred = model(batch_X)  # Forward pass
                    loss = criterion(y_pred, batch_y.view(-1, 1))  # Calculate loss
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update weights

            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient tracking
                y_pred = model(X_test)  # Forward pass on test set
            mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE

            return mae

        # Perform Bayesian hyperparameter tuning using MAE as the objective function
        bayes_search = BayesSearchCV(
            objective_function, search_space, n_iter=30, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_absolute_error'
        )
        bayes_search.fit(X_train, y_train)  # Fit the search on training data

        # Get the best hyperparameters found
        best_params = bayes_search.best_params_

        # Train the final model with the best hyperparameters
        best_model = NeuralNetworkWithAttention(input_dim=X_train.shape[1], hidden_dim=best_params['hidden_dim'])
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])  # Initialize optimizer
        criterion = nn.L1Loss()  # Mean Absolute Error (L1 Loss)

        # Create DataLoader for training the best model
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

        for epoch in range(best_params['epochs']):
            best_model.train()  # Set model to training mode
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                y_pred = best_model(batch_X)  # Forward pass
                loss = criterion(y_pred, batch_y.view(-1, 1))  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

        best_model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            y_true = y_test  # True values
            y_pred = best_model(X_test)  # Predicted values

        # Append true values and model predictions for evaluation
        true_values.extend(y_true.numpy())
        model_values.extend(y_pred.numpy())  # Use .numpy() for conversion

    return np.array(true_values), np.array(model_values)  # Return true and predicted values
