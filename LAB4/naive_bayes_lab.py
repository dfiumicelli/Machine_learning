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

        #TODO
        print("Execution time of vectorized version: ", time.time() - start)
        return self.phi_t, self.phi_X

    def predict(self,X):

        start_time = time.time()
        #TODO
        print("prediction time: ", time.time()-start_time)
        return t_hat

    def predict_v(self,X):
        start_time = time.time()
        #TODO
        print("prediction time: ", time.time() - start_time)
        return t_hat

    def predict_proba(self,X):
        # TODO
        return prob

    def predict_proba_v(self,X):
        #TODO
        return prob


