import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from LAB3_solutions.LAB3.logistic_regression import sigmoid, logistic_cost, vectorized_gradient, predict

# Load test dataset
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

# visualize data:
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#set up input and output matrices
X = data[['Exam 1', 'Exam 2']].values
m, n = X.shape
X = np.concatenate((np.ones((m,1)), X), axis=1 )
n +=1
Y = data['Admitted'].values
Y = np.expand_dims(Y, 1)

# Test sigmoid function
z = np.linspace(-10,10,100)
out = sigmoid(z)
plt.figure()
plt.plot(z, out)
plt.show()

# Test logistic cost function
W = np.ones((3,1))*0.1
print ('Test cost function: ', logistic_cost(W, X, Y))

c = vectorized_gradient(W,X,Y)

# Test logistic regression gradient
import scipy.optimize as opt
result = opt.fmin_tnc(func=logistic_cost, x0=W, fprime=vectorized_gradient, args=(X, Y))
print ('Logistic cost after optimization: ',logistic_cost(result[0], X, Y))

# Predict with computed weights
accuracy1, Y_hat = predict(result[0], X, np.array(Y).reshape(m,))
# accuracy1 = sum(Y_hat == np.array(Y).reshape(m,))/(1.0*len(Y))
print("Accuracy with Ours: ", accuracy1)

# Test results with LIBLINEAR
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y.reshape((m,)).T)

result = logreg.predict(X)
accuracy2 = sum(result == np.array(Y).reshape(m,))/(1.0*len(Y))
print("Accuracy with LIBLINEAR: ", accuracy2)
