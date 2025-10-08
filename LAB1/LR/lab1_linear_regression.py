import numpy as np
from time import time

'''
LAB 1
Design a generic linear regression solution using the knwledge acquired so far.
If you are unsure on the solution to this lab, start with a simple working solution (e.g. using for loops). 
Then implement the vectorized solution. Compare the two using time() to measure the processing time. 
'''
#-----------------------------------------------------
# Hypothesis function
#-----------------------------------------------------

def hyp(X, w):
    '''

    :param X: Design matrix
    :param w: Linear regression weights
    :return: the value of the hypothesis function for each row of X
    '''

    # your code here

    return np.dot(X, w)  # matrix multiplication


#-----------------------------------------------------
# Cost function
#-----------------------------------------------------
def cost(t, X, w):
    '''

    :param t: target values
    :param X: Design matrix
    :param w: Linear regression weights
    :return: The cost function for the given input data
    '''
    # your code here

    return

#-----------------------------------------------------
# Linear regression solver - gradient descent
#-----------------------------------------------------
def linear_regression_fit_GD(t, X, alpha, epsilon=0.001):
    '''

    :param t:
    :param X:
    :param epsilon:
    :return:
    '''

    # your code here

    return

# -----------------------------------------------------
# Linear regression solver - LMS solution
# -----------------------------------------------------
def linear_regression_fit_NE(X, t):
    '''
    :param t: target values
    :param X: Design matrix
    :return: Linear regression weights
    '''
    X_t = np.transpose(X)
    m = np.linalg.inv(np.dot(X_t, X))
    aux = np.dot(m, X_t)
    return np.dot(aux, t)

# -----------------------------------------------------
# Standardization
# -----------------------------------------------------
def standardize(X, mean=None, std=None):
    if mean is not None and std is not None:
        return (X - mean) / std #axis = 0 for column-wise operation
    else:
        mean=np.mean(X, axis=0)
        std=np.std(X, axis=0)
        return (X - mean) / std, mean, std #axis = 0 for column-wise operation

# -----------------------------------------------------
# Min-Max Scaling
# -----------------------------------------------------
def min_max_scale(X):
    '''
    :param X: Design matrix
    :return: min-max scaled X
    '''
    return (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) #axis = 0 for column-wise operation


# -----------------------------------------------------
# Mean Normalization
# -----------------------------------------------------
def mean_normalize(X):
    '''
    :param X: Design matrix
    :return: mean normalized  X
    '''
    # TODO


# -----------------------------------------------------
# Residual Sum of Squares
# -----------------------------------------------------
def RSS(t_true, t_hat):
    return np.sum((t_true - t_hat) ** 2)


# -----------------------------------------------------
# Residual Standard Error
# -----------------------------------------------------
def RSE(t_true, t_hat, X):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :param X: Design matrix
    :return: RSE
    '''
    # TODO


# -----------------------------------------------------
# Mean Absolute Error
# -----------------------------------------------------
def MAE(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: MAE
    '''
    return np.mean(np.abs(t_true - t_hat))


# -----------------------------------------------------
# Mean Squared Error
# -----------------------------------------------------
def MSE(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: MSE
    '''
    return np.mean((t_true - t_hat) ** 2)


# -----------------------------------------------------
# Root Mean Square Error
# -----------------------------------------------------
def RMSE(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: RMSE
    '''
    return np.sqrt(MSE(t_true, t_hat))


# -----------------------------------------------------
# R2 - Statistics
# -----------------------------------------------------
def R2(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: R2
    '''
    ss_res = np.sum((t_true - t_hat) ** 2)
    ss_tot = np.sum((t_true - np.mean(t_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# -----------------------------------------------------
# Adjusted R2
# -----------------------------------------------------
def R2_adj(t_true, t_hat, X):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :param X: Design matrix
    :return: adjusted R2
    '''
    # TODO



