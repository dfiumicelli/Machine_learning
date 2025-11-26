from sklearn import svm
from sklearn.metrics import accuracy_score

def model_selection(X, t, X_val, t_val):
    """
    Returns the best choice of C and gamma for SVM with RBF kernel.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training samples, where n_samples is the number of samples and n_features is the number of features.
    t : ndarray, shape (n_samples,)
        Labels for training set.
    X_val : ndarray, shape (n_val_samples, n_features)
        Cross validation samples, where n_val_samples is the number of cross validation samples and n_features is the
        number of features.
    t_val : ndarray, shape (n_val_samples,)
        Labels for cross validation set.

    Returns
    -------
    C : float
        The best choice of penalty parameter C of the error term.
    gamma : float
        The best choice of kernel coefficient for 'rbf'.
    """
    # Definiamo i valori da testare per C e gamma
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]
    # come valori iniziali per il miglior punteggio e i migliori parametri
    best_score = 0
    best_C = C_values[0]
    best_gamma = gamma_values[0]

    # Cicliamo su tutte le combinazioni di C e gamma
    for C in C_values:
        for gamma in gamma_values:
            # Creiamo e addestriamo il modello SVM con kernel RBF
            clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            clf.fit(X, t)

            # Valutiamo il modello sul set di validazione
            score = accuracy_score(t_val, clf.predict(X_val))

            # Se il punteggio è migliore del migliore finora, aggiorniamo i parametri migliori
            if score > best_score:
                best_score = score
                best_C = C
                best_gamma = gamma

    return best_C, best_gamma
