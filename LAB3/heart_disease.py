import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
cols = ['age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df = pd.read_csv('processed.cleveland.data', header=None, index_col=None, names=cols, na_values=['?'])
#TODO

# binarize otput
#TODO

# normalize
#TODO

# Train simple Logit model as a benchmark
#TODO

# print Accuracy
#TODO

# Test
df_te = pd.read_csv('processed.hungarian.data', header=None, index_col=None, names=cols, na_values=['?'])
#TODO

# print Test accuracy
#TODO
