import numpy as np
import time

class NaiveBayes:

    def __init__(self):
        self.phi_t = -1
        self.phi_X = np.zeros((1,2))
        self.lphi_X1 = np.zeros((1,2))
        self.lphi_X0 = np.zeros((1,2))
        self.lphi_t = np.zeros((1,2))

    def fit(self,X,t):

        start = time.time()

        N, D = X.shape #N = 700 email, D = 2501 parole
        self.phi_t = np.sum(t > 0)/float(N)
        self.phi_X = np.zeros(shape=(D,2), dtype=float)

        N_1 = np.sum(t>0)
        N_0 = N - N_1

        for j in range(D):
            self.phi_X[j,1] = np.sum(np.logical_and(X[:,j], t>0)+1)/float(N_1+2)
            self.phi_X[j,0] = np.sum(np.logical_and(X[:,j], t<1)+1)/float(N_0+2)
        #dobbiamo applicare Laplace smoothing altrimenti se una parola non compare mai in una certa classe la probabilità diventa 0 per quella classe
        print("Execution time of non vectorized version: ", time.time() - start)

        return self.phi_t, self.phi_X

    def fit_v(self,X,t):

        start = time.time()

        N, D = X.shape
        X = np.vstack((np.zeros(shape=(2,D)), np.ones(shape=(2,D)), X)) #aggiungiamo due righe di 0 e due righe di 1 per il Laplace smoothing
        t = np.hstack((np.array((0,1,0,1)), t)) #aggiungiamo due etichette 0 e due etichette 1 per il Laplace smoothing
        N,D = X.shape
        spam_idx = (t>0).reshape(N,1) #indice per le email spam
        hat_idx = (t<1).reshape(N,1) #indice per le email non spam
        self.phi_X = np.zeros((D,2), dtype=float)
        N_1 = sum(spam_idx)
        N_0 = N - N_1
        self.phi_t = N_1/N
        self.phi_X[:,1] = (np.sum(X*spam_idx, axis=0))/float(N_1)
        self.phi_X[:,0] = (np.sum(X*hat_idx, axis=0))/float(N_0)
        print("Execution time of vectorized version: ", time.time() - start)
        return self.phi_t, self.phi_X

    def predict(self,X):

        start_time = time.time()
        N, D = X.shape
        numerator1 = np.zeros(shape=(N,1))
        numerator0 = np.zeros(shape=(N,1))
        for i in range(N):
            for j in range(D):
                numerator1[i] += X[i,j]*np.log(self.phi_X[j,1]) + (1 - X[i,j])*np.log(1 - self.phi_X[j,1])
                numerator0[i] += X[i,j]*np.log(self.phi_X[j,0]) + (1 - X[i,j])*np.log(1 - self.phi_X[j,0])
            numerator1[i] += np.log(self.phi_t)
            numerator0[i] += np.log(1 - self.phi_t)
        t_hat = numerator1 > numerator0
        print("prediction time: ", time.time()-start_time)
        return t_hat

    def predict_v(self,X):
        start_time = time.time()
        numerator1 = np.dot(X, np.log(self.phi_X[:,1])) + np.dot((1 - X), np.log(1 - self.phi_X[:,1]))
        numerator1 += np.ones(numerator1.shape)*np.log(self.phi_t)
        numerator0 = np.dot(X, np.log(self.phi_X[:,0])) + np.dot((1 - X), np.log(1 - self.phi_X[:,0]))
        numerator0 += np.ones(numerator0.shape)*np.log(1 - self.phi_t)
        t_hat = numerator1>numerator0
        t_hat = t_hat.reshape((numerator1.shape[0],1))
        print("prediction time: ", time.time() - start_time)
        return t_hat

    def predict_proba(self,X):
        # TODO
        return prob

    def predict_proba_v(self,X):
        #TODO
        return prob


