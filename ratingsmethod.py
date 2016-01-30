import numpy as np
from numpy.linalg import inv


#SVD, implementation training via ALS
class SVD_ALS:
    def __init__(self, user_movie, lambda_p = 0.2, lambda_q = 0.001, N=20, K=10):
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.N = N
        self.K = K
        self.N_user = user_movie[0]
        self.N_movies = user_movie[1]

    
    def fit(self, data):
        mu_rating = np.zeros((self.N_movies, self.N_user))
        for u, i, r in data:
            mu_rating[i - 1, u - 1] = r
        mu_index = mu_rating.astype(np.bool)

        self.Q = 0.1 * np.random.random((self.N_movies, self.K))
        self.P = 0.1 * np.random.random((self.N_user, self.K))
        
        for i in range(self.N):
            for k, (index, r) in enumerate(zip(mu_index.T, mu_rating.T)):
                Qbuf = self.Q[index]
                A = np.dot(Qbuf.T, Qbuf)
                d = np.dot(Qbuf.T, r[index])
                self.P[k] = np.dot(inv(self.lambda_p * Qbuf.shape[0] * np.eye(self.K) + A), d)

            for k, (index, r) in enumerate(zip(mu_index, mu_rating)):
                Pbuf = self.P[index]
                if Pbuf.shape[0] == 0:
                    continue
                A = np.dot(Pbuf.T, Pbuf)
                d = np.dot(Pbuf.T, r[index])
                self.Q[k] = np.dot(inv(self.lambda_q * Pbuf.shape[0] * np.eye(self.K) + A), d)
            
            
    def predict(self, data):
        u = data[:, 0] - 1
        i = data[:, 1] - 1
        return np.sum(self.P[u] * self.Q[i], axis=1)