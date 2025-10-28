import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df = pd.read_csv('processed.cleveland.data', header=None, index_col=None, names=cols, na_values=['?'])
print(df.head())
# check missing values

# drop rows with missing values
df = df.dropna()
label = 'num'
# binarize output
out = 1-pd.get_dummies(df[label])[0]
features = cols[0:-1]
X_training = df[features].values
t_training = out.values

from sklearn.preprocessing import StandardScaler
# normalize
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
from sklearn.linear_model import LogisticRegression

# Train simple Logit model as a benchmark
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_training, t_training)
accuracy = logreg.score(X_training, t_training)

# print Accuracy

print("Training accuracy (sklearn): ", accuracy)
# Test
df_te = pd.read_csv('processed.hungarian.data', header=None, index_col=None, names=cols, na_values=['?'])
out = 1-pd.get_dummies(df_te[label])[0]
df_te = df_te.dropna()

X_test = df_te[features].values
t_test = out.values

X_test = scaler.transform(X_test)
test_accuracy = logreg.score(X_test, t_test)

# print Test accuracy
print("Test accuracy (sklearn): ", test_accuracy)