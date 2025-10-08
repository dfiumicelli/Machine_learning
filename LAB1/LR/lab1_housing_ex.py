
from time import time

import numpy as np
from matplotlib import pyplot as plt
from LAB1.LR.lab1_load_data import load_housing_data
from LAB1.LR.lab1_linear_regression import standardize, RSS, RSE, MSE, MAE, RMSE, R2, R2_adj
from LAB1.LR.lab1_linear_regression import linear_regression_fit_GD, linear_regression_fit_NE
from LAB1.LR.lab1_linear_regression import hyp
# Load housing datasets
features_file = 'datasets/ex1data2.txt'
X, y = load_housing_data(features_file)


split=0.8
tr_size=int(X.shape[0]*split)
X_tr=X[0:tr_size]
t_tr=y[0:tr_size]
X_te=X[tr_size:]
t_te=y[tr_size:]
X_tr, mean, std = standardize(X_tr)
X_te = standardize(X_te, mean=mean, std=std)
### Normalization
print(X_tr)

### Intercept term
x_0_tr = np.ones((X_tr.shape[0],1))
X_tr = np.append(x_0_tr, X_tr, axis=1)

x_0_te = np.ones((X_te.shape[0],1))
X_te = np.append(x_0_te, X_te, axis=1)
# divide train and test sets


### Compute theta with Gradient Descent and Normal Equation
start_NE = time()
# call NE linear regression
w_NE = linear_regression_fit_NE(X_tr, t_tr)
print(w_NE)
stop_NE = time()
start_GD = time()
# call GD linear regression
w_GD = 0 #TODO
stop_GD = time()
print("theta from Normal Equation: ", w_NE, " in ", (stop_NE - start_NE) * 1000, ' ms')
print("theta from Gradient Descent: ", w_GD, " in ", (stop_GD - start_GD) * 1000, ' ms')

### Predict using the theta from Gradient Descent and Normal Equation
t_hat_NE = hyp(X_te, w_NE)
t_hat_GD = 0  # TODO

plt.figure('Hypotheses')
plt.title('Hypotheses')
plt.scatter(X_te[:, 1], t_te)
plt.plot(X_te[:, 1], t_hat_NE, 'r', marker='o', label='NE',linestyle="None")
#plt.plot(X_te[:, 1], t_hat_GD, 'g', marker='o', label='GD',linestyle="None")
plt.legend()
plt.show()

###  Compute performance
print("Performance of Normal Equation: ")
print("RSS = ", RSS(t_te, t_hat_NE))
print("RSE = ", RSE(t_te, t_hat_NE, X_te))
print("MSE = ", MSE(t_te, t_hat_NE))
print("MAE = ", MAE(t_te, t_hat_NE))
print("RMSE = ", RMSE(t_te, t_hat_NE))
print("R2 = ", R2(t_te, t_hat_NE))
print("R2_adj = ", R2_adj(t_te, t_hat_NE, X_te))
print("Performance of Gradient Descent: ")
print("RSS = ", RSS(t_te, t_hat_GD))
print("RSE = ", RSE(t_te, t_hat_GD, X))
print("MSE = ", MSE(t_te, t_hat_GD))
print("MAE = ", MAE(t_te, t_hat_GD))
print("RMSE = ", RMSE(t_te, t_hat_GD))
print("R2 = ", R2(t_te, t_hat_GD))
print("R2_adj = ", R2_adj(t_te, t_hat_GD, X_te))

plt.figure('Predictions')
plt.title('Predictions')
plt.plot(t_hat_GD, label=r'$\hat{t}_{GD}$')
plt.plot(t_hat_NE, label=r'$\hat{t}_{NE}$')
plt.plot(t, label='t')
plt.legend()
plt.show()