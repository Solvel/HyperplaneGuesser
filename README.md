# Perceptron Project

### This repository contains Python code for implementing and exploring the behavior of a perceptron hyperplane guesser. The project is designed for binary classification tasks where the goal is to find a decision boundary (hyperplane) that separates data points into two classes.

## Contents

###    perceptron.py: Implementation of a basic perceptron.
###    perceptronGradientDescent.py: Implementation of a perceptron using gradient descent.
###    main.py: Main script to run the perceptron hyperplane guesser, with options to set the type of perceptron and the number of runs.
###    plotting.py: Module for plotting graphs using matplotlib.

## Usage

Install the required libraries:

    pip install numpy pandas matplotlib

Run the main script:

    python main.py

You can specify the type of perceptron and the number of runs as command-line arguments.
### perceptron types:
    --perceptron_type 0  (random)
    --perceptron_type 1  (gradient descent)

Example

    python main.py --runs 10 --perceptron_type 1

This will run the perceptron-based hyperplane guesser for 10 runs using the gradient descent method.
