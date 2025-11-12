import numpy as np
import scipy.sparse as spr
from LAB4.naive_bayes_lab import NaiveBayes

path = './' # inserite il percorso alla cartella con i dati preparati

# Load features prima riga ID mail, seconda riga ID parola, terza riga frequenza
data_tr = np.loadtxt(path + 'train_features_n.txt', dtype='int')
data_spr_tr = spr.csr_matrix((data_tr[:,2], (data_tr[:,0], data_tr[:,1])))[1:,1:]
#per ottenere la matrice densa altrimenti avremmo problemi di dimensioni visto che è molto sparsa
data_full_tr = data_spr_tr.toarray() #converte la matrice sparsa in densa

# Load labels metà 0 non spam, metà 1 spam
labels_tr = np.loadtxt(path + 'train-labels.txt', dtype='int')

naive_bayes_class = NaiveBayes()

#estimate parameters
X_tr = data_full_tr > 0 #trasformiamo le frequenze in booleani per indicare presenza o assenza della parola
t_tr = labels_tr

phi_X, phi_t = naive_bayes_class.fit_v(X_tr, t_tr)

# Load test set
data_te = np.loadtxt(path + 'test_features_n.txt', dtype='int')
data_spr_te = spr.csr_matrix((data_te[:,2], (data_te[:,0], data_te[:,1])))[1:,1:]
data_full_te = data_spr_te.toarray()
labels_te = np.loadtxt(path + 'test-labels.txt', dtype='int')

X_te = data_full_te > 0
t_te = labels_te

#prediction

t_hat = naive_bayes_class.predict_v(X_te)
N_te = t_hat.shape[0]
t_te = t_te.reshape(N_te, 1)
print(sum(t_te == t_hat)/float(N_te))

# predict probabilities

t_prob = naive_bayes_class.predict_proba_v(X_te)