# perceptron.py
import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, weights=None, offset=None, y_classes=None):
        self.weights = weights
        self.offset = offset
        self.y_classes = y_classes
        self.counter = 0  # Initialize the counter

    def fit(self, X, y, seed=None):
        gen = np.random.default_rng(seed=seed)

        while True:
            # Generate random weights within range
            random_weights = gen.uniform(low=-1000, high=1000, size=X.shape[1])

            # Get a random point within the boundaries
            random_point = gen.uniform(low=X.min(axis=0), high=X.max(axis=0))

            # Get the offset of the line to be on the random point
            offset = -np.sum(random_point * random_weights)

            # Calculate the dot product of features and weights
            dot_product = np.sum(X * random_weights, axis=1) + offset

            # Determine the sign of the dot product for each data point
            signed_values = np.where(dot_product >= 0, 1, -1)

            # Convert signed values to predicted classes (0 or 1)
            preds = (signed_values + 1) / 2

            # Check if the hyperplane separates the data
            if np.all(preds == y) and np.all(dot_product != 0):
                break

            self.counter += 1

        # Calculate the slope and intercept of the line
        slope = -random_weights[0] / random_weights[1]
        intercept = -offset / random_weights[1]

        print(f"Separation achieved in {self.counter} guesses! Line equation: y = {slope} * x + {intercept}")

        self.weights = random_weights
        self.offset = offset
        self.y_classes = np.unique(y)

    # Make predictions based on the learned hyperplane
    def predict(self, X):
        # Calculate the dot product of features and weights
        dot_product = np.sum(X * self.weights, axis=1) + self.offset

        # Determine the sign of the dot product for each data point
        signed_values = np.where(dot_product >= 0, 1, -1)

        # Convert signed values to predicted classes (0 or 1)
        preds = (signed_values + 1) / 2
        return preds
