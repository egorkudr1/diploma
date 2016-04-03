import numpy as np
from scipy.special import expit
import random
import sys
import rankingmeasure
import ratingsmethod


class PopRec:
    def __init__(self, user_item):
        self.N_users = user_item[0]
        self.N_items = user_item[1]
    

    def fit(self, data):
        votes = np.zeros(self.N_items)
        for items in data:
            votes += np.bincount(items, minlength=self.N_items)
        # votes = np.bincount(np.array(data).ravel())
        most_pop = np.argsort(-votes)
        if most_pop.shape[0] < self.N_items:
            mask = np.in1d(np.arange(self.N_items), most_pop)
            mask = np.logical_not(mask)
            most_pop = np.concatenate((most_pop, np.arange(self.N_items)[mask]))
        self.most_pop = most_pop
        

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
    def __init__(self, user_item, K=10, reg=0.001, lrate=0.0001, maxiter=10, verbose=0):
        self.K = K
        self.reg = reg
        self.lrate = lrate
        self.maxiter = maxiter
        self.verbose = verbose
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        

    def fit(self, data):
        self.U = 0.01 * np.random.random((self.N_users, self.K))
        self.V = 0.01 * np.random.random((self.N_items, self.K))

        for t in range(self.maxiter):
            for u, items in enumerate(data):
                curU = self.U[u]
                curV = self.V[items]

                f_ij = np.sum(self.U[u][np.newaxis, :] * curV, axis=1)
                diff_f_ij = f_ij[:, np.newaxis] - f_ij[np.newaxis, :]
                diff_V_ij = curV[:, np.newaxis, :] - curV[np.newaxis, :, :]
                g_ij = expit(-f_ij)
                diverV =  expit(diff_f_ij)*expit(-diff_f_ij)
                denominanor  = 1 / (1 - expit(diff_f_ij))

                dU = np.sum(g_ij[:, np.newaxis] * curV, axis=0)
                tmp = diverV * denominanor
                dU -= np.sum(tmp[:, :, np.newaxis] * diff_V_ij, axis=(0, 1))
                dU -= self.reg * curU
                self.U[u] += self.lrate * dU

                for j, index in enumerate(items):
                    f_ij = np.sum(self.U[u][np.newaxis, :] * curV, axis=1)
                    diff_f_ij = f_ij[:, np.newaxis] - f_ij[np.newaxis, :]
                    diff_V_ij = curV[:, np.newaxis, :] - curV[np.newaxis, :, :]
                    g_ij = expit(-f_ij)
                    diverV =  expit(diff_f_ij)*expit(-diff_f_ij)
                    denominanor  = 1 / (1 - expit(diff_f_ij))
                  
                    coef = np.sum(diverV[j, :] * (denominanor[:, j] - denominanor[j, :]))
                    dV = (coef + g_ij[j]) * self.U[u]  - self.reg * curV[j]
                    self.V[index] += self.lrate * dV
                
            if self.verbose == 1:
                print("iteration", t)
                sys.stdout.flush()
            elif self.verbose == 2:
                print("iteration", t, "loss", self.get_loss(data))
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
        self.U = 0.1 * np.random.random((self.N_users, self.K))
        self.V = 0.1 * np.random.random((self.N_items, self.K))
        num_pos_feedback = 0
        for items in data:
            num_pos_feedback += len(items)
        
        # data_neg = []
        # for u, items in enumerate(data):
        #     locn = int(self.rate_neg_sample * len(items))
        #     loc_list = []
        #     for l in range(locn):
        #         while True:
        #             j = random.randint(0, self.N_items - 1)
        #             if j in data[u]:
        #                 continue
        #             break
        #         loc_list.append(j)
        #     data_neg.append(loc_list)


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
                #j = random.choice(data_neg[u])
                curU = self.U[u, :].copy()
                curVi = self.V[i, :].copy()
                curVj = self.V[j, :].copy()

                xuij = np.sum(curU * curVi) - np.sum(curU * curVj)
                
                coef = expit(-xuij)
                
                self.U[u] += self.lrate * (coef * (curVi - curVj) + self.regU * curU)
                self.V[i] += self.lrate * (coef * (curU) + self.regIpos * curVi)
                self.V[j] += self.lrate * (- coef * curU + self.regIneg * curVj)

                if self.verbose == 2:
                    loss += -np.log(expit(xuij))
                    loss += self.regU * np.sum(curU ** 2) + self.regIpos * np.sum(curVi ** 2) + self.regIneg * np.sum(curVj ** 2)

            if self.verbose == 1:
                print("iteration", t)
                sys.stdout.flush()
            elif self.verbose == 2:
                print("iteration", t, "loss", loss)
                sys.stdout.flush()


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

            if self.verbose == 1:
                print('iteration', t)
                sys.stdout.flush()
            elif self.verbose == 2:
                print('iteration', t, 'loss', self.count_loss(data))
                sys.stdout.flush()


    def _update_latent(self, V, W, index):
        localV = V[index,  :]
        if localV.shape[0] > 0:
            M = W + self.alpha * np.dot(localV.T, localV) + np.diag(self.lmbd * np.ones(self.K))
            res = np.dot(np.linalg.inv(M), (1 + self.alpha) * np.sum(localV, axis=0))
        else:
            res = np.zeros(self.K)
        return res


    def count_loss(self, data):
        loss = 0
        for u, items in enumerate(data):
            c = np.ones(self.N_items)
            p = np.zeros(self.N_items)
            c[items] += self.alpha
            p[items] = 1
            ulist = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
            loss += np.sum(c * (p - ulist) ** 2)
        loss += self.lmbd * (np.sum(self.U ** 2) + np.sum(self.V ** 2))
        return loss


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
        self.U = 0.01 * np.random.random((self.N_users, self.K))
        self.V = 0.01 * np.random.random((self.N_items, self.K))    
        itemdata = [[] for i in range(self.N_items)]
        for u, items in enumerate(data):
            for i in items:
                itemdata[i].append(u) 

        for t in range(self.maxiter):
            for u, items in enumerate(data):
                curV = self.V[items]
                fui = np.dot(curV, self.U[u])
                diff_fui = fui[:, np.newaxis] - fui[np.newaxis, :]
                gi = expit(fui)
                gij = expit(diff_fui)
                diff_gi = expit(fui) * expit(-fui)
                diff_gij = expit(diff_fui) * expit(-diff_fui)
                delta =  diff_gi * np.sum(gij, axis=0) - gi * np.sum(diff_gij, axis=0)
                tmp = np.sum(gi[np.newaxis, :] * diff_gij, axis=1)
                dU = (np.sum(delta[:, np.newaxis] * curV)  + np.sum(tmp[:, np.newaxis] * curV)) / len(items) - self.reg * self.U[u]
                self.U[u] += self.lrate * dU

            for u, items in enumerate(data):
                buf_items = self._buffer_constract(u, items)
                for i, index in enumerate(buf_items):
                    fui = np.dot(self.V[buf_items], self.U[u])
                    diff_fui = fui - fui[i] 
                    gi = expit(fui[i])
                    gij = expit(diff_fui)
                    diff_gi = expit(fui) * expit(-fui)
                    diff_gij = expit(gij) * expit(-gij)
                    
                    coef = np.sum(diff_gi[i] * gij  + (gij - gij[i]) * diff_gij) / len(buf_items)
                    dV = coef * self.U[u] - self.reg * self.V[index]
                    self.V[index] += self.lrate * dV

            if self.verbose == 1:
                print('iteration', t)
                sys.stdout.flush()
            elif self.verbose == 2:
                print('iteration', t, 'loss', self.count_loss(data))
                sys.stdout.flush()
            elif self.verbose == 3:
                print('iteration', t, 'loss', rankingmeasure.get_MAP(data, data, self, skip_train=False))
                sys.stdout.flush()


    def _buffer_constract(self, u, items):
        p = np.min(np.dot(self.V[items], self.U[u]))
        fij = np.dot(self.V, self.U[u])
        yones = np.zeros(self.N_items)
        yones[items] = 1
        S = np.where(np.logical_and(fij >= p, yones == 0))[0]
        if S.shape[0] > self.n_sample:
            S = np.random.choice(S, self.n_sample)
        B_minus = S[np.argsort(-fij[S])]
        if B_minus.shape[0] > len(items):
            B_minus = B_minus[:len(items)]
        return np.concatenate((items, B_minus))
    

    def count_loss(self, data):
        loss = 0
        for u, items in enumerate(data):
            fi = np.dot(self.V[items], self.U[u])
            diff_fi = fi[:, np.newaxis] - fi[np.newaxis, :]
            gi = expit(fi)
            gij = expit(diff_fi)
            loss += np.sum(gi * np.sum(gij, axis=0))/len(items)
        loss -= self.reg/2 * (np.sum(self.U ** 2) +  np.sum(self.V ** 2))
        return loss


    def get_list(self, u):
        fij = np.sum(self.U[u][np.newaxis, :] * self.V, axis=1)
        return np.argsort(-fij)


    def get_f(self, u):
        return np.dot(self.V, self.U[u]) 


class pureSVD:
    def __init__(self, user_item, lambda_p=0.2, lambda_q=0.001, maxiter=20, K=10, rate_neg_sample=1.0, verbose=0):
        self.cf = ratingsmethod.SVD_ALS(user_movie=user_item, lambda_p=lambda_p, lambda_q=lambda_q, N=maxiter, K=K, verbose=verbose)
        self.N_users = user_item[0]
        self.N_items = user_item[1]
        self.rate_neg_sample = rate_neg_sample


    def fit(self, data):
        new_data = []
        for u, items in enumerate(data):
            for i in items:
                new_data.append([u, i, 1])

            n_neg_sample = int(items.shape[0] * self.rate_neg_sample)
            for t in range(n_neg_sample):
                while(True):
                    j = random.randint(0, self.N_items - 1)
                    if j in items:
                        continue
                    break
                new_data.append([u, j, 0])
        new_data = np.array(new_data)
        print(new_data.shape)
        self.cf.fit(new_data)
            

    def get_list(self, u):
        fij = np.sum(self.cf.P[u][np.newaxis, :] * self.cf.Q, axis=1)
        return np.argsort(-fij)

    def get_f(self, u):
        return np.dot(self.V, self.U[u])