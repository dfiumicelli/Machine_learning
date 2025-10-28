# Solution of Ex 1
import numpy as np
import pandas as pd

# vectorized sigmoid function
def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

# define useful check function
def check_input(X,Y):
    if isinstance(X, pd.DataFrame):
        X = np.ndarray(X.as_matrix())
        Y = np.ndarray(Y.as_matrix())
    elif isinstance(X, np.ndarray):
        X = np.ndarray(X)
        Y = np.ndarray(Y)
    elif not isinstance(X, np.matrix):
        print("Undefined input X type in function logistic_cost. Abort")
        exit()
    return X, Y

# Logistic Regression Cost Function
def logistic_cost(W, X, Y):
    #l = sum(Y * np.log(sigmoid(X @ W)) + (1 - Y) * np.log(1 - sigmoid(X @ W)))
    h = sigmoid(np.dot(X, W))
    cost = -np.dot(Y.T, np.log(h)) - np.dot((1-Y).T, np.log(1-h))

    return cost

# logistic regression cost gradient
def vectorized_gradient(W, X, Y):
    W = W.reshape(W.shape[0], 1)
    h = sigmoid(X @ W)
    grad = (X.T @ (h - Y)).T
    return grad

def gradient(W, X, Y):
    W = np.matrix(W)
    X = np.matrix(X)
    Y = np.matrix(Y)
    n_parameters = X.shape[1]
    grad = np.zeros(n_parameters)
    if W.shape[0] == n_parameters:
        error = sigmoid(X @ W) - Y
    else:
        error = sigmoid(X @ W.T) - Y
    for j in range(n_parameters):
        term = np.multiply(error, X[:, j])
        grad[j] = np.sum(term)
    return grad

def predict(W, X, Y):
    N = X.shape[0]
    Y_hat = sigmoid(np.dot(X, W)) >= 0.5
    accuracy = sum(Y_hat == Y)/N
    return accuracy, Y_hat