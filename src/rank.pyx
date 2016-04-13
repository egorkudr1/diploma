import cython

import numpy as np
cimport numpy as np



cdef extern from "cplusplusrank.h":
    cdef void fast_climf_fit(double*, double*, int, int, int, int*, int*, int, double, double, int)
    cdef void fast_bpr_mf_fit(double*, double*, int, int, int, int* , int*, int, double, double, double, double, int)
    cdef void fast_tfmap_fit(double*, double*, int, int, int, int*, int*, int, double, double, int, int)

class CLiMF:
    def __init__(self, user_item, K=10, reg=0.001, lrate=0.0001, maxiter=10, verbose=0):
        self.K = K
        self.reg = reg
        self.lrate = lrate
        self.maxiter = maxiter
        self.verbose = verbose
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        

    def fit(self, data):
        cdef np.ndarray[double, ndim=2, mode="c"] U = 0.01 * np.random.random((self.N_users, self.K))
        cdef np.ndarray[double, ndim=2, mode="c"] V = 0.01 * np.random.random((self.N_items, self.K))

        edge_u = []
        edge_i = []
        for u, items in enumerate(data):
            edge_u.extend([u] * len(items))
            edge_i.extend(items)
        edge_u = np.array(edge_u).astype(np.int32)
        edge_i = np.array(edge_i).astype(np.int32)

        cdef np.ndarray[int, ndim=1, mode="c"] input_u = edge_u
        cdef np.ndarray[int, ndim=1, mode="c"] input_i = edge_i

        fast_climf_fit(&U[0,0], &V[0,0], self.N_users, self.N_items, self.K, &input_u[0], &input_i[0], len(edge_u),
                            self.reg, self.lrate, self.maxiter)

        self.U = U
        self.V = V
        
    def get_list(self, u):
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij) 


    def get_f(self, u):
        return np.dot(self.V, self.U[u])


class BPR_MF:
    def __init__(self, user_item, K, lrate, regU, regIpos, regIneg, maxiter, verbose=0):
        self.K = K
        self.lrate = lrate
        self.regU = regU
        self.regIpos = regIpos
        self.regIneg = regIneg
        self.maxiter = maxiter
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        self.verbose = verbose


    def fit(self, data):
        cdef np.ndarray[double, ndim=2, mode="c"] U = 0.1 * np.random.random((self.N_users, self.K))
        cdef np.ndarray[double, ndim=2, mode="c"] V = 0.1 * np.random.random((self.N_items, self.K))

        edge_u = []
        edge_i = []
        for u, items in enumerate(data):
            edge_u.extend([u] * len(items))
            edge_i.extend(items)
        edge_u = np.array(edge_u).astype(np.int32)
        edge_i = np.array(edge_i).astype(np.int32)

        cdef np.ndarray[int, ndim=1, mode="c"] input_u = edge_u
        cdef np.ndarray[int, ndim=1, mode="c"] input_i = edge_i

        fast_bpr_mf_fit(&U[0,0], &V[0,0], self.N_users, self.N_items, self.K, &input_u[0], &input_i[0], len(edge_u),
                            self.regU, self.regIpos, self.regIneg, self.lrate, self.maxiter)
        self.U = U
        self.V = V


    def get_list(self, u):
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij)


    def get_f(self, u):
        return np.dot(self.V, self.U[u])


class iMF:
    def __init__(self, user_item, K, lmbd, alpha, maxiter, verbose=0):
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        self.K = K
        self.lmbd = lmbd
        self.alpha = alpha
        self.maxiter = maxiter
        self.verbose = verbose
        

    def fit(self, data):
        self.U = 0.01 * np.random.random((self.N_users, self.K))
        self.V = 0.01 * np.random.random((self.N_items, self.K))
        itemdata = [[] for i in range(self.N_items)]
        for u, items in enumerate(data):
            for i in items:
                itemdata[i].append(u) 

        for t in range(self.maxiter):
            W = np.dot(self.V.T, self.V)
            for u, items in enumerate(data):
                self.U[u] = self._update_latent(self.V, W, items)

            W = np.dot(self.U.T, self.U)
            for i, users in enumerate(itemdata):
                self.V[i] = self._update_latent(self.U, W, users)


    def _update_latent(self, V, W, index):
        localV = V[index,  :]
        if localV.shape[0] > 0:
            M = W + self.alpha * np.dot(localV.T, localV) + np.diag(self.lmbd * np.ones(self.K))
            res = np.dot(np.linalg.inv(M), (1 + self.alpha) * np.sum(localV, axis=0))
        else:
            res = np.zeros(self.K)
        return res


    # def count_loss(self, data):
    #     loss = 0
    #     for u, items in enumerate(data):
    #         c = np.ones(self.N_items)
    #         p = np.zeros(self.N_items)
    #         c[items] += self.alpha
    #         p[items] = 1
    #         ulist = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
    #         loss += np.sum(c * (p - ulist) ** 2)
    #     loss += self.lmbd * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
    #     return loss


    def get_list(self, u):
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij)


    def get_f(self, u):
        return np.dot(self.V, self.U[u])


class TFMAP:
    def __init__(self, user_item, K=10, reg=0.001, lrate=0.001, n_sample=100, maxiter=20, verbose=0):
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        self.K = K
        self.reg = reg
        self.lrate = lrate
        self.n_sample = n_sample
        self.maxiter = maxiter
        self.verbose = verbose

    def fit(self, data):
        cdef np.ndarray[double, ndim=2, mode="c"] U = 0.1 * np.random.random((self.N_users, self.K))
        cdef np.ndarray[double, ndim=2, mode="c"] V = 0.1 * np.random.random((self.N_items, self.K))

        edge_u = []
        edge_i = []
        for u, items in enumerate(data):
            edge_u.extend([u] * len(items))
            edge_i.extend(items)
        edge_u = np.array(edge_u).astype(np.int32)
        edge_i = np.array(edge_i).astype(np.int32)

        cdef np.ndarray[int, ndim=1, mode="c"] input_u = edge_u
        cdef np.ndarray[int, ndim=1, mode="c"] input_i = edge_i

        fast_tfmap_fit(&U[0,0], &V[0,0], self.N_users, self.N_items, self.K, &input_u[0], &input_i[0], len(edge_u),
                            self.reg,  self.lrate, self.n_sample, self.maxiter)

        self.U = U
        self.V = V
    

    def get_list(self, u):
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij)


    def get_f(self, u):
        return np.dot(self.V, self.U[u]) 