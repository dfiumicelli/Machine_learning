import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format
import matplotlib.pyplot as plt
from numpy.ma.core import nonzero

telecom_cust = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
pd.set_option('display.expand_frame_repr', False)
print(telecom_cust.head())
print(telecom_cust.info())

telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
not_numeric = telecom_cust[pd.isnull(telecom_cust['TotalCharges'])]
print(not_numeric)
telecom_cust = telecom_cust.dropna()
telecom_cust = telecom_cust.drop(columns=['customerID'], axis=1)
telecom_cust['Churn'].replace(to_replace='Yes', value=1, inplace=True)
telecom_cust['Churn'].replace(to_replace='No', value=0, inplace=True)

telecom_cust_dummies = pd.get_dummies(telecom_cust, drop_first=True)
#corr_matrix = telecom_cust_dummies.corr(numeric_only=True)
#sns.heatmap(corr_matrix, annot=True, cmap='Blues')
#plt.show()

#telecom_cust_temp_dummies = pd.get_dummies(telecom_cust[['MultipleLines', 'PhoneService', 'OnlineBackup', 'InternetService', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']], drop_first=True)
corr_matrix = telecom_cust_dummies.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.show()

telecom_cust_dummies = telecom_cust_dummies.drop([], axis=1)
#so rimasto indietro, soluzione su unistudium
# Splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, t_train, t_test = train_test_split(X, t, train_size=0.7, test_size=0.3, random_state=100)

#TODO: Normalize features if necessary

#TODO: perform model selectiom and test the best model on the test data
