import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
cols = ['age','sex', 'cp','trestbps','chol',
        'fbs','restecg','thalach','exang',
        'oldpeak','slope','ca','thal','num']
df = pd.read_csv('processed.cleveland.data',
                 header=None, index_col=None, names=cols, na_values=['?'])

df = df.dropna()
features = ['age','sex', 'cp','trestbps','chol',
        'fbs','restecg','thalach','exang',
        'oldpeak','slope','ca','thal']
label = 'num'
# binarize otput
out = 1-pd.get_dummies(df[label])[0] #inverte l'output per avere 1 = malato, 0 = sano

X_training = df[features].values
t_training = out.values

# normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)

# Train simple Logit model as a benchmark
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_training, t_training)

# print Accuracy
t_hat_training = logreg.predict(X_training)
from sklearn.metrics import classification_report
print("Classification Report on Training set: \n",
      classification_report(t_training, t_hat_training))

# Test
df_te = pd.read_csv('processed.hungarian.data',
                    header=None, index_col=None, names=cols, na_values=['?'])
df_te = df_te.ffill() #forward fill in case there are NaN in the middle
df_te = df_te.bfill() #backfill in case there are NaN at the beginning

out_te = 1-pd.get_dummies(df_te[label])[0]

X_test = df_te[features].values
t_test = out_te.values

X_test = scaler.transform(X_test)

# print Test accuracy
t_hat_test = logreg.predict(X_test)
print("Classification Report on Test set: \n",
      classification_report(t_test, t_hat_test))
