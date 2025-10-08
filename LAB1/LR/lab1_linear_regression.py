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

    return


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
    # TODO


# -----------------------------------------------------
# Standardization
# -----------------------------------------------------
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# -----------------------------------------------------
# Min-Max Scaling
# -----------------------------------------------------
def min_max_scale(X):
    '''
    :param X: Design matrix
    :return: min-max scaled X
    '''
    # TODO


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
    # TODO


# -----------------------------------------------------
# Mean Squared Error
# -----------------------------------------------------
def MSE(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: MSE
    '''
    # TODO


# -----------------------------------------------------
# Root Mean Square Error
# -----------------------------------------------------
def RMSE(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: RMSE
    '''
    # TODO


# -----------------------------------------------------
# R2 - Statistics
# -----------------------------------------------------
def R2(t_true, t_hat):
    '''
    :param t_true: target values
    :param t_hat: predicted t
    :return: R2
    '''
    # TODO


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



