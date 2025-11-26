import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

data = 'pulsar_data_train.csv'

df = pd.read_csv(data)

print(df.shape)


# let's preview the dataset

print(df.head())

# We can see that there are 9 variables in the dataset. 8 are continuous variables and 1 is discrete variable.
# The discrete variable is `target_class` variable. It is also the target variable.

#Renaming columns
df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness',
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
# view the renamed column names

print(df.columns)

# check for missing values in variables

#TODO

# check distribution of target_class column

#TODO

# view the percentage distribution of target_class column

#TODO

X = df.drop(['target_class'], axis=1)

y = df['target_class']

# split X and y into training and testing sets

#TODO


#Normalization

#TODO

# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score, f1_score, classification_report

# instantiate classifier with default hyperparameters
#TODO

# fit classifier to training set
#TODO

# make predictions on test set
#TODO

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(t_test, t_pred)))
print('Classification report with default hyperparameters: \n', classification_report(t_test, t_pred))


# instantiate classifier with rbf kernel and C=100 and test it
#TODO


# instantiate classifier with linear kernel and C=1.0 amd test it
#TODO


# instantiate classifier with linear kernel and C=100.0 amd test it
#TODO

#Compare the train-set and test-set accuracy to check for overfitting.

#TODO

# instantiate classifier with polynomial kernel and C=1.0 and test it
#TODO

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix
#TODO

# Use GridSearch CV to perform model selection and test it

from sklearn.model_selection import GridSearchCV

# import SVC classifier
from sklearn.svm import SVC

# instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
svc=SVC()

# declare parameters for hyperparameter tuning
#TODO
