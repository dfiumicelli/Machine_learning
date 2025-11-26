import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score

from LAB6.plot_data import plot_data
from LAB6.gaussian_kernel import gaussian_kernel
from LAB6.visualize_boundary_linear import visualize_boundary_linear
from LAB6.visualize_boundary import visualize_boundary
from LAB6.model_selection import model_selection


# =============== Part 1: Loading and Visualizing Data ================
print('Loading and Visualizing Dataset 1...')

# Load data
data = np.loadtxt("svm_data.csv", delimiter=",")
X = data[:, :2]
t = data[:, 2].ravel()
# Plot training data
plt.figure()
plot_data(X, t)
plt.title("Dataset 1")
plt.show()


# ==================== Part 2: Training Linear SVM ====================


print('Training Linear SVM...')
# Change the C value below and see how the decision boundary varies (e.g., try C = 1000).
C = 1
clf = svm.LinearSVC(C=C) #per classificazione lineare usa svm.SVC(kernel='linear', C=C), per regressione usa svm.SVR
clf.fit(X, t)
y_train_hat = clf.predict(X)
print ('score:', accuracy_score(t, y_train_hat))

plt.figure()
visualize_boundary_linear(X, t, clf)
plt.title("Dataset 1")
plt.show()


# =============== Part 3: Implementing Gaussian Kernel ===============
print ('Evaluating the Gaussian Kernel...')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)

print ('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {}: \n\t{}\n' \
      '(for sigma = 2, this value should be about 0.324652)'.format(sigma, sim))


# =============== Part 4: Visualizing Dataset 2 ================
print ('Loading and Visualizing Dataset 2...')

data = np.loadtxt("svm_data2.csv", delimiter=",")
X = data[:, :2]
t = data[:, 2].ravel()
# Plot training data
plt.figure()
plot_data(X, t)
plt.title("Dataset 1")
plt.show()

# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
print ('Training SVM with RBF Kernel...')

# SVM Parameters
clf = svm.SVC(C=1, kernel='rbf', gamma=40.5) #per classificazione lineare usa svm.SVC(kernel='linear', C=C), perregressione usa svm.SVR
clf.fit(X, t)
y_train_hat = clf.predict(X)
print ('score:', accuracy_score(t, y_train_hat))
plt.figure()
visualize_boundary(X, t, clf)
plt.title("Dataset 2")
plt.show()

# =============== Part 6: Visualizing Dataset 3 ================
print ('Loading and Visualizing Data 3...')
data = np.loadtxt("svm_data3.csv", delimiter=",")
X = data[:, :2]
t = data[:, 2].ravel()
# Plot training data
plt.figure()
plot_data(X, t)
plt.title("Dataset 3")
plt.show()


# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
data = np.loadtxt("svm_data_val.csv", delimiter=",")
X_val = data[:, :2]
t_val = data[:, 2].ravel()

# Try different SVM Parameters here
C, gamma = model_selection(X, t, X_val, t_val)
print('Best C:', C)
print('Best gamma:', gamma)
# Train the SVM
clf = svm.SVC(C=C, kernel='rbf', gamma=gamma) #per classificazione lineare usa svm.SVC(kernel='linear', C=C), perregressione usa svm.SVR
clf.fit(X, t)
y_train_hat = clf.predict(X)
print ('score:', accuracy_score(t, y_train_hat))
# Visualize the boundary
plt.figure()
visualize_boundary(X, t, clf)
plt.title("Dataset 3")
plt.show()