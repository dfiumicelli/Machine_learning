import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
# import matplotlib
# matplotlib.use('TkAgg')
pd.set_option('display.expand_frame_repr', False)


random.seed(1211)


def my_k_fold(df, k_fold=5):
    indexes = list(range(len(df)))
    random.shuffle(indexes)
    fold_sizes = int(len(df) / k_fold)
    splits = []
    for k in range(k_fold):
        ts_idx = indexes[k * fold_sizes:(k + 1) * fold_sizes]
        tr_idx = [i for i in indexes if i not in ts_idx]
        splits.append((tr_idx, ts_idx))
    return splits

def my_leave_one_out(df):
    return my_k_fold(df, k_fold=len(df))


if __name__ == "__main__":
    df = pd.read_csv('winequality-red.csv', sep=";")
    print(df.head())

    list_k_fold_splits = my_k_fold(df, k_fold=5)

    # dop.drop(columns='DRUPE_COLOR', inplace=True)

    ####################################################################################################################
    #                                       CROSS-VALIDATION FOR MODEL ASSESSMENT                                      #
    ####################################################################################################################
    f1_cv_ASSESSMENT = []
    # Assessment
    splits = list_k_fold_splits
    for tr_idx, ts_idx in splits:
        #TODO: use KNNClassifier
        boh = 0  # remove this line when you implement the TODO
## Report
    print('The cross-validated F1-score of your algorithm is ', np.mean(np.asarray(f1_cv_ASSESSMENT)))
#
#     ####################################################################################################################
#     #                                       CROSS-VALIDATION FOR MODEL SELECTION                                       #
#     ####################################################################################################################
    f1_cv_SEL = []
    # Define parameter search space
    k_parameter = range(1, 8, 2)
    # Selection
    for k_par in k_parameter:
        splits = list_k_fold_splits
        f1_temp = []
        for tr_idx, ts_idx in splits:
            dop_tr = df.iloc[tr_idx]
            dop_ts = df.iloc[ts_idx]
            t_tr = dop_tr['quality']
            X_tr = dop_tr.drop(columns='quality', axis=1)
            curr_knn = KNeighborsClassifier(n_neighbors=k_par)
            curr_knn.fit(X_tr, t_tr)
            t_ts = dop_ts['quality']
            X_ts = dop_ts.drop(columns='quality', axis=1)
            t_hat = curr_knn.predict(X_ts)
            f1_temp.append(f1_score(t_ts, t_hat, average='macro'))
        f1_cv_SEL.append(np.mean(np.asarray(f1_temp)))

    # Report
    print('The cross-validated F1-scores of your algorithm with the explored parameters are: ')
    for i in range(len(k_parameter)):
        print('For k = ', k_parameter[i], ' --> F1-score = ', f1_cv_SEL[i])
    print('Overall, the best value for parameter k is ', k_parameter[np.argmax(np.asarray(f1_cv_SEL))],
          ' since it leads to F1-score = ', f1_cv_SEL[np.argmax(np.asarray(f1_cv_SEL))])

#
# ########################################################################################################################
# # Uhm... Actually! Sklearn already has the functions we implemented.
# ########################################################################################################################
#
#     from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
#     splits = KFold(n_splits=5, shuffle=True).split(dop)
#

knn = KNeighborsClassifier(n_neighbors=5)
hyper_parameters = {'n_neighbors': [1, 3, 5, 7], 'weights': ['uniform', 'distance']}
cv_cnn = GridSearchCV(knn, hyper_parameters, cv=5, scoring='f1_macro')
cv_cnn.fit(df.drop(columns='quality', axis=1), df['quality'])
print('Best hyper-parameters found by GridSearchCV are: ', cv_cnn.best_params_)