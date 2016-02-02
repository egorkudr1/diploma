import numpy as np
from scipy.special import expit

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


    def fit(self, data):
        self.U = np.random.random((self.N_users, self.K))
        self.V = np.random.random((self.N_items, self.K))

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
            elif self.verbose == 2:
                print("iteration", t, "loss?", self.get_loss(data))


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




                # diff_f_ij = f_ij[:, np.newaxis] - f_ij[np.newaxis, :]
                # dU += expit(diff_f_ij) / (1 - expit(diff_f_ij)) * 
                # for i in items:
                #     f_ij = np.sum(self.V[i] * curU)
                #     dU += expit(-f_ij) * self.V[i]
                #     diff_f_ij = f_ij - np.sum(curU[np.newaxis, :] * self.V[items], axis=1)
                #     tmp = expit(diff_f_ij) * expit(-diff_f_ij) / (1- expit(diff_f_ij))
                #     diffV = self.V[i][:, np.newaxis] - self.V[items].T
                #     tmp = tmp[np.newaxis:,] * diffV
                #     dU += np.sum(tmp, axis=1)
                
                # dU -= self.reg * curU
                # self.U[u] += self.lrate * dU

                # for i in items:
