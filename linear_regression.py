import numpy as np


class LinearRegression:
    """
    Linear Regression implemented from scratch using Gradient Descent.

    This class allows:
    - fitting a linear model to data
    - predicting new values
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the model parameters.

        Parameters:
        - learning_rate: step size for gradient descent
        - n_iterations: number of iterations for training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Model parameters (to be learned during training)
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.

        Steps (high-level):
        1. Initialize weight and bias
        2. Loop over the number of iterations
        3. Compute predictions
        4. Compute error
        5. Update weight and bias
        """
        # Number of training samples
        n_samples = X.shape[0]

        # Initialize parameters
        self.weight = 0.0
        self.bias = 0.0

        # Gradient Descent loop
        for _ in range(self.n_iterations):
            # Compute predictions
            # Compute error
            # Update parameters
            pass


    def predict(self, X):
        """
        Predict output values for given input data X.

        Returns:
        - Predicted values
        """
        pass  # to be implemented
