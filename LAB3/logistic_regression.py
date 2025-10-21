# Solution of Ex 1
import numpy as np
import pandas as pd

# vectorized sigmoid function
def sigmoid(z):
    #TODO
    return

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
    #TODO
    return

# logistic regression cost gradient
def vectorized_gradient(W, X, Y):
    #TODO
    return

def gradient(theta, X, y):
    #TODO

    return

def predict(W, X, Y):
    #TODO
    return