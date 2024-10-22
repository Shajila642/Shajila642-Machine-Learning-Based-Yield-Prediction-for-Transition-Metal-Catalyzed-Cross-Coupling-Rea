import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.base import BaseEstimator, RegressorMixin

# Define the Attention mechanism class
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        # Define the attention layers: first a linear layer to project the input, then a Tanh activation, 
        # followed by another linear layer to produce attention scores
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Compute attention weights and apply them to the input
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_values = torch.sum(attention_weights * x, dim=1)
        return attended_values

# Define the neural network class with an attention mechanism
class NeuralNetworkWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetworkWithAttention, self).__init__()
        self.attention = Attention(input_dim, hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Ensure input_dim matches n_features
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer for regression

    def forward(self, x):
        # Pass the input through the attention layer and then through the feedforward layers
        attended_values = self.attention(x)
        print("Shape of attended_values:", attended_values.shape)  # Debugging output for attended values shape
        output = self.fc2(torch.relu(self.fc1(attended_values)))
        return output

# Define a wrapper class for the neural network that implements scikit-learn's estimator interface
class NeuralNetworkWithAttentionEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_dim=64, lr=0.001, batch_size=32, epochs=100):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None  # To hold the trained model

    def fit(self, X, y):
        # Create and train the neural network
        self.model = NeuralNetworkWithAttention(input_dim=X.shape[1], hidden_dim=self.hidden_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Adam optimizer
        criterion = nn.L1Loss()  # Use Mean Absolute Error (L1 Loss)

        # Create a PyTorch DataLoader for training data
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Train the model over multiple epochs
        for epoch in range(self.epochs):
            self.model.train()  # Set the model to training mode
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # Zero gradients for the optimizer
                y_pred = self.model(batch_X)  # Forward pass
                loss = criterion(y_pred, batch_y.view(-1, 1))  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update model parameters

    def predict(self, X):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y_pred = self.model(X)  # Make predictions
        return y_pred.numpy()  # Return predictions as numpy array
    
def neural_network_with_attention_hyperparameter_tuning(X, y, test_size=0.2, n_iterations=1):
    true_values = []  # Store true values for evaluation
    model_values = []  # Store model predictions for evaluation

    # Perform multiple iterations of training and testing
    for _ in range(n_iterations):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=_)

        # Convert data to PyTorch tensors
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

        # Define the hyperparameter search space
        search_space = {
            'hidden_dim': Integer(16, 128),  # Search for hidden dimensions between 16 and 128
            'lr': Real(0.001, 0.1),  # Search for learning rate between 0.001 and 0.1
            'batch_size': Integer(32, 256),  # Search for batch size between 32 and 256
            'epochs': Integer(10, 100),  # Search for number of epochs between 10 and 100
        }

        # Define the objective function to minimize MAE
        def objective_function(hidden_dim, lr, batch_size, epochs):
            estimator = NeuralNetworkWithAttentionEstimator(hidden_dim=hidden_dim, lr=lr, batch_size=batch_size, epochs=epochs)
            estimator.fit(X_train, y_train)  # Fit the model
            y_pred = estimator.predict(X_test)  # Make predictions on test set
            mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
            return mae

        # Perform Bayesian optimization for hyperparameter tuning
        bayes_search = BayesSearchCV(
            NeuralNetworkWithAttentionEstimator(),
            search_space,
            n_iter=30,  # Number of iterations for Bayesian search
            cv=5,  # 5-fold cross-validation
            n_jobs=-1,  # Use all available cores
            verbose=1,  # Print detailed output
            scoring='neg_mean_absolute_error'  # Use negative MAE for scoring
        )
        bayes_search.fit(X_train, y_train)  # Fit the Bayesian search

        # Get the best hyperparameters
        best_params = bayes_search.best_params_

        # Train the final model with the best hyperparameters
        best_model = NeuralNetworkWithAttention(input_dim=X_train.shape[1], hidden_dim=best_params['hidden_dim'])
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
        criterion = nn.L1Loss()  # Mean Absolute Error (L1 Loss)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

        # Train the model using the best hyperparameters
        for epoch in range(best_params['epochs']):
            best_model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = best_model(batch_X)
                loss = criterion(y_pred, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()

        # Evaluate the model on the test set
        best_model.eval()
        with torch.no_grad():
            y_true = y_test
            y_pred = best_model(X_test)

        true_values.extend(y_true.numpy())  # Store true values
        model_values.extend(y_pred.numpy())  # Store model predictions

    return np.array(true_values), np.array(model_values)  # Return arrays of true values and model predictions
