# perceptron_gradient_descent.py
import numpy as np

class PerceptronGradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.counter = 0  # Rename 'epochs' to 'counter'
        self.weights = None
        self.bias = None  # Update the attribute name from bias to offset
        self.max_iterations = max_iterations  # Rename 'epochs' to 'max_iterations'

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for self.counter in range(self.max_iterations):  # Rename 'epochs' to 'counter'
            # Calculate the dot product of features and weights
            linear_output = np.dot(X, self.weights) + self.bias

            # Apply the step function to get predictions
            predictions = np.where(linear_output >= 0, 1, 0)

            # Calculate the gradient of the loss function
            gradient_weights = np.dot(X.T, y - predictions)
            gradient_bias = np.sum(y - predictions)

            # Update weights and bias
            self.weights += self.learning_rate * gradient_weights
            self.bias += self.learning_rate * gradient_bias

            # Check for convergence
            if np.all(predictions == y):
                # Calculate the slope and intercept of the line
                slope = -self.weights[0] / self.weights[1]
                intercept = -self.bias / self.weights[1]

                print(f"Converged in {self.counter + 1} iterations! Line equation: y = {slope} * x + {intercept}")
                break

    def predict(self, X):
        # Calculate the dot product of features and weights
        linear_output = np.dot(X, self.weights) + self.bias

        # Apply the step function to get predictions
        predictions = np.where(linear_output >= 0, 1, 0)
        return predictions
