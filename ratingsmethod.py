import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix
import sys

#SVD, implementation training via ALS
class SVD_ALS:
    def __init__(self, user_movie, lambda_p = 0.2, lambda_q = 0.001, N=20, K=10, verbose=0):
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.N = N
        self.K = K
        self.N_user = user_movie[0]
        self.N_movies = user_movie[1]
        self.verbose = verbose
 
    def fit(self, data):
        mu_rating = np.zeros((self.N_movies, self.N_user))
        for u, i, r in data:
            mu_rating[i, u] = r
        mu_index = mu_rating.astype(np.bool)

        self.Q = 0.1 * np.random.random((self.N_movies, self.K))
        self.P = 0.1 * np.random.random((self.N_user, self.K))
        
        for i in range(self.N):
            if self.verbose == 1:
                print('iteration', i)
                sys.stdout.flush()

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
        u = data[:, 0]
        i = data[:, 1]
        return np.sum(self.P[u] * self.Q[i], axis=1)


class SVDplusplus:
    def __init__(self, user_item, regB=0.1, regI=0.1, regU=0.1, lRate=0.01, N=100, K=5, verbose=0):
        self.N_user = user_item[0]
        self.N_items = user_item[1]
        self.regB = regB
        self.regI = regI
        self.regU = regU
        self.lRate = lRate
        self.N = N
        self.K = K
        self.verbose = verbose

    def fit(self, data):
        self.globalmean = np.mean(data[:, 2])
        globmax = np.max(data[:, 2])
        globmin = np.min(data[:, 2])
        self.usermean = globmin + (globmax - globmin) * np.random.random(self.N_user) - self.globalmean 
        self.itemmean = globmin + (globmax - globmin) * np.random.random(self.N_items) - self.globalmean 
        self.P = np.random.random((self.N_user, self.K))
        self.Q = np.random.random((self.N_items, self.K))
        self.Y = np.random.random((self.N_items, self.K))
        
        self.indexY = []
        for u in range(self.N_user):
            self.indexY.append(data[data[:, 0] == u + 1, 1] - 1)
        # self.indexY = coo_matrix((np.ones(data.shape[0], dtype=int), (data[:, 0] - 1, data[:, 1] - 1)), 
        #                             shape=(self.N_user, self.N_items))

        self.w = np.zeros(self.N_user)
        for u in range(self.N_user):
            self.w[u] = np.sum(data[:, 0] == u + 1)
        self.w = np.sqrt(self.w)

        if np.sum(self.w == 0) > 0:
            print("w == 0")

        for j in range(self.N):
            loss = 0
            for u, i, r in data:
                u -= 1
                i -= 1
                
                pred_rui = self.__predict_ui(u, i)
                eui = r - pred_rui

                bu = self.usermean[u]
                self.usermean[u] += self.lRate * (eui - self.regB * bu)                

                bi = self.itemmean[i]
                self.itemmean[i] += self.lRate * (eui - self.regB * bi)

                pu = self.P[u]
                qi = self.Q[i]

                self.P[u] += self.lRate * (eui * qi - self.regU * pu)
                self.Q[i] += self.lRate * (eui * (pu + np.sum(self.Y[self.indexY[u]], axis=0) / self.w[u]) - self.regI * qi)
                
                if self.verbose == 2:
                    loss += eui * eui
                    loss += self.regB * bu * bu
                    loss += self.regB * bi * bi
                    loss += np.sum(self.regU * pu * pu + self.regI * qi * qi)
                    loss += self.regU * np.sum(self.Y[self.indexY[u]] ** 2)

                self.Y[self.indexY[u]] = self.lRate * (eui  / self.w[u] * qi[np.newaxis, :]  - self.regU * self.Y[self.indexY[u]])

        
            if self.verbose == 1:
                print("iteration", j)
                sys.stdout.flush()
            elif self.verbose == 2:
                print ("iteration", j, "loss", loss)
                sys.stdout.flush()
    
    def __predict_ui(self, u, i):
        w = self.w[u]
        result = self.globalmean + self.usermean[u] + self.itemmean[i] 
        if w == 0:
            result += np.sum(self.Q[i] * self.P[u])
        else:
            result += np.sum(self.Q[i] * (self.P[u] + np.sum(self.Y[self.indexY[u]], axis=0) / w))
        return result
        
    def predict(self, data):
        result = []
        for u, i in data:
            result.append(self.__predict_ui(u - 1, i - 1))
        return np.array(result)
        
