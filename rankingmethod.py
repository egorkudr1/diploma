import numpy as np
from scipy.special import expit
import random
import sys

class PopRec:
    def __init__(self, user_item):
        self.N_users = user_item[0]
        self.N_items = user_item[1]
    
    def fit(self, data):
        votes = np.bincount(np.array(data).ravel())
        self.most_pop = np.argsort(-votes)
        
        
    def get_list(self, u):
        return self.most_pop


class RandomRec:
    def __init__(self, user_item):
        self.N_users = user_item[0]
        self.N_items = user_item[1]

    def fit(self, data):
        pass

    def get_list(self, u):
        res = np.arange(self.N_items)
        np.random.shuffle(res)
        return res


class CLiMF:
    def __init__(self, user_item, K=10, reg=0.001, lrate=0.0001, maxiter=30, verbose=0):
        self.K = K
        self.reg = reg
        self.lrate = lrate
        self.maxiter = maxiter
        self.verbose = verbose
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        self.U = 0.1 * np.random.random((self.N_users, self.K))
        self.V = 0.1 * np.random.random((self.N_items, self.K))


    def fit(self, data):
        for t in range(self.maxiter):
            for u, items in enumerate(data):
                curU = self.U[u]
                curV = self.V[items]
                f_ij = np.sum(self.U[u][np.newaxis, :] * curV, axis=1)
                diff_f_ij = f_ij[:, np.newaxis] - f_ij[np.newaxis, :]
                diff_V_ij = curV[:, np.newaxis, :] - curV[np.newaxis, :, :]
                g_ij = expit(-f_ij)

                dU = np.sum(g_ij[:, np.newaxis] * curV, axis=0)

                diverV =  expit(diff_f_ij)*expit(-diff_f_ij)

                denominanor  = 1 / (1 - expit(diff_f_ij))

                tmp = diverV * denominanor
                dU -= np.sum(tmp[:, :, np.newaxis] * diff_V_ij, axis=(0, 1))
                dU -= self.reg * curU
                self.U[u] += self.lrate * dU

                for j, index in enumerate(items):
                    coef = g_ij[j]
                    coef += np.sum(diverV[j] * (denominanor[:, j] - denominanor[j, :]))
                    dV = coef * self.U[u] - self.reg * curV[j]
                    self.V[index] += self.lrate * dV

            if self.verbose == 1:
                print("iteration", t)
                sys.stdout.flush()
            elif self.verbose == 2:
                print("iteration", t, "loss?", self.get_loss(data))
                sys.stdout.flush()


    def get_loss(self, data):
        loss = 0
        for u, items in enumerate(data):
            fij =  np.sum(self.U[u][np.newaxis, :] * self.V[items], axis=1)
            loss += np.sum(np.log(expit(fij)))
            loss += np.sum(np.log(1 -expit(fij[:, np.newaxis] - fij[np.newaxis, :])))
        loss -= self.reg / 2 * (np.sum(self.U * self.U) + np.sum(self.V + self.V))
        return loss


    def get_list(self, u):
        curU = self.U[u]
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij)


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
        self.U = 0.1 * np.random.random((self.N_users, self.K))
        self.V = 0.1 * np.random.random((self.N_items, self.K))
        self.verbose = verbose


    def fit(self, data):
        num_pos_feedback = 0
        for items in data:
            num_pos_feedback += len(items)
            
        for t in range(self.maxiter):
            loss = 0
            for l in range(num_pos_feedback * 10):
                u = random.randint(0, self.N_users - 1)
                i = random.choice(data[u])
                while(True):
                    j = random.randint(0, self.N_items - 1)
                    if j in data[u]:
                        continue
                    break
             
                curU = self.U[u, :].copy()
                curVi = self.V[i, :].copy()
                curVj = self.V[j, :].copy()

                xuij = np.sum(curU * curVi) - np.sum(curU * curVj)
                
                coef = expit(-xuij)
                
                self.U[u] += self.lrate * (coef * (curVi - curVj) - self.regU * curU)
                self.V[i] += self.lrate * (coef * (curU) - self.regIpos * curVi)
                self.V[j] += self.lrate * (- coef * curU - self.regIneg * curVj)

                if verbose == 2:
                    loss += -np.log(expit(xuij))
                    loss += self.regU * np.sum(curU ** 2) + self.regIpos * np.sum(curVi ** 2) + self.regIneg * np.sum(curVj ** 2)

            if self.verbose == 1:
                print("iteration", t)
                sys.stdout.flush()
            elif self.verbose == 2:
                print("iteration", t, "loss", loss)
                sys.stdout.flush()


    def get_list(self, u):
        curU = self.U[u]
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij)

