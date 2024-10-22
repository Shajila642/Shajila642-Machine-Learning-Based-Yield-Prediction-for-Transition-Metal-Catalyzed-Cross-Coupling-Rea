import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Define a simple neural network model class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the neural network architecture.

        Parameters:
            input_size (int): Number of input features for the neural network.
        """
        super(NeuralNetwork, self).__init__()  # Call to the parent class's constructor
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 32)           # Second hidden layer with 32 neurons
        self.fc3 = nn.Linear(32, 1)             # Output layer with 1 neuron for regression

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (Tensor): Input tensor containing features.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = self.fc3(x)               # No activation function in the output layer for regression
        return x

def neural_network(X, y, test_size, n_iterations, epochl, lr):
    """
    Train a neural network regressor using PyTorch and return the true and predicted values.
    
    Parameters:
        X (np.array): Features of the dataset, of shape (n_samples, n_features).
        y (np.array): Labels of the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        n_iterations (int): Number of iterations for training with different random states.
        epochl (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.

    Returns:
        Tuple: True values and predicted values from the model.
    """
    expt_values = []  # List to store the true labels
    model_values = [] # List to store model predictions

    for i in range(n_iterations):  # Iterate through the number of specified iterations
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.Tensor(X_train)  # Training features
        y_train = torch.Tensor(y_train)   # Training labels
        X_test = torch.Tensor(X_test)     # Test features
        y_test = torch.Tensor(y_test)     # Test labels

        # Create and train the neural network
        input_size = X_train.shape[1]  # Number of features in the training set
        model = NeuralNetwork(input_size)  # Initialize the neural network model
        optimizer = optim.Adam(model.parameters(), lr)  # Adam optimizer with the specified learning rate
        criterion = nn.MSELoss()  # Loss function (Mean Squared Error)

        # Lists to track training and validation losses (if validation is implemented)
        train_losses = []
        val_losses = []
        
        for epoch in range(epochl):  # Loop over the number of epochs
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(X_train)  # Forward pass: compute predicted outputs
            loss = criterion(outputs, y_train.unsqueeze(1))  # Calculate the loss
            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update the model parameters
            
            # Uncomment the following lines if validation loss tracking is needed
            '''
            # Calculate validation loss
            with torch.no_grad():  # Disable gradient calculation for validation
                val_predictions = model(X_test)
                val_loss = criterion(val_predictions, y_test.view(-1, 1))

            train_losses.append(loss.item())  # Store training loss
            val_losses.append(val_loss.item())  # Store validation loss
            '''

        # Make predictions using the trained model on the test set
        y_pred = model(X_test)

        # Convert predictions and true labels back to numpy arrays for further analysis
        y_test = y_test.detach().numpy()  # Detach tensor and convert to numpy
        y_pred = y_pred.detach().numpy()  # Detach tensor and convert to numpy
        
        expt_values.extend(y_test)  # Extend the list of true values
        model_values.extend(y_pred)  # Extend the list of model predictions
        
    # Convert the predicted values to strings (optional)
    new_model_val = []
    for x in model_values:
        x = str(x[0])  # Convert to string (this might not be necessary depending on the use case)
        new_model_val.append(x)

    return expt_values, new_model_val  # Return true and predicted values
