import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format
import matplotlib.pyplot as plt

telecom_cust = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

#TODO: EDA

# Splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, t_train, t_test = train_test_split(X, t, train_size=0.7, test_size=0.3, random_state=100)

#TODO: Normalize features if necessary

#TODO: perform model selectiom and test the best model on the test data
