import numpy as np
import pandas as pd
import copy
import sys
import copy
import random
from sklearn import linear_model
import rankingmeasure

 
class TopEnsemble:
    def __init__(self, topK = 15, verbose=0):
        self.verbose = verbose
        self.topK = topK
    
    def fit(self, list_cf):
        self.list_cf = copy.deepcopy(list_cf)
    
    def get_list(self, u):
        N_items = self.list_cf[1].N_items
        N_methods = len(self.list_cf)
        res = np.zeros((N_methods, N_items))
        for i, num_cf in enumerate(np.random.choice(N_methods, N_methods, replace=False)):
            res[i] = self.list_cf[num_cf].get_list(u)
        
        total_res = res.T.ravel()
        ans = []
        for elem in total_res:
            if elem not in ans:
                ans.append(elem)
            if len(ans) == self.topK:
                break
        return np.array(ans)


class RatingEnsemble:
    def __init__(self,  weights = None, alpha=0.5, verbose=0):
        self.alpha = alpha
        self.verbose = verbose
        self.weights = weights
        
    def fit(self, list_cf):
        self.list_cf = copy.deepcopy(list_cf)
        if self.weights is None:
            self.weights = np.ones(len(list_cf))/len(list_cf)
        else:
            assert(self.weights.shape[0] == len(list_cf))

        self.N_items = self.list_cf[0].N_items
        self.N_methods = len(self.list_cf)
        self.index = np.exp(-self.alpha * np.arange(self.N_items))

    def get_list(self, u):
        res = np.zeros(self.N_items)
        for i in range(self.N_methods):
            ranks = self.list_cf[i].get_list(u)
            res[ranks] += self.weights[i] * self.index 
        return np.argsort(-res)


class InnerValueEnsemble:
    def __init__(self, weights = None, verbose=0):
        self.verbose = verbose
        self.weights = weights

    def fit(self, list_cf, validation, train,  trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.weights = np.ones(len(list_cf))/len(list_cf)
        

        self.N_items = self.list_cf[0].N_items
        self.N_methods = len(self.list_cf)

    def get_list(self, u):
        res = np.zeros(self.N_items)
        for i in range(len(self.list_cf)):
            tmp = self.list_cf[i].get_f(u)
            tmp -= np.min(tmp)
            tmp /= np.max(tmp)
            res += self.weights[i] * tmp
        return np.argsort(-res)


class ValEns:
    def __init__(self, weights, metric, tiedrank=False, verbose=0):
        self.verbose = verbose
        self.weights = weights
        self.metric = metric
        self.tiedrank = tiedrank


    def fit(self, list_cf, validation, train,  trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0].N_items
        self.N_users = self.list_cf[0].N_users

        best_value_metric = -1
        best_w = None
        for w in self.weights:
            self.w = w
            value_metric = self.metric(train, validation, self)
            if self.verbose == 2:
                print(value_metric)
                sys.stdout.flush()
            if value_metric > best_value_metric:
                best_value_metric = value_metric
                best_w = w
        if self.verbose == 1:
            print("best w is", best_w)
            sys.stdout.flush()
        self.w = best_w

    def get_list(self, u):
        res = np.zeros(self.N_items)
        for i, alpha in enumerate(self.w):
            if not self.tiedrank:
                fij = self.list_cf[i].get_f(u)
                fij -= np.min(fij)
                fij /= np.max(fij)
            else:
                fij = self.list_cf[t].get_tiedrank(u)
            
            res += alpha * fij 
        return np.argsort(-res)

    def get_f(self, u):
        res = np.zeros(self.N_items)
        for i, alpha in enumerate(self.w):
            if not self.tiedrank:
                fij = self.list_cf[i].get_f(u)
                fij -= np.min(fij)
                fij /= np.max(fij)
            else:
                fij = self.list_cf[t].get_tiedrank(u)
            res += alpha * fij
        return res


class BoostValEns:
    def __init__(self,  metric, index, tiedrank=False, num_weights=51, verbose=0):
        self.verbose = verbose
        self.num_weights = num_weights
        self.metric = metric
        self.index = index
        self.tiedrank = tiedrank


    def fit(self, list_cf, validation, train, trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0]
        self.N_users = self.list_cf[0].N_users

        weights = np.array([[alpha, 1 - alpha] for alpha in np.linspace(0, 1, self.num_weights)])

        self.res_model = [copy.deepcopy(self.list_cf[self.index[0]])]
        for i in range(1, len(list_cf)):
            self.res_model.append(ValEns(weights, self.metric, self.tiedrank))
            self.res_model[i].fit([self.res_model[i - 1], self.list_cf[self.index[i]]], validation, train, trainvalidation)
        

    def get_list(self, u):
        return self.res_model[-1].get_list(u)


class TreeValEns:
    def __init__(self, metric, index, tiedrank=False, num_weights=51, verbose=0):
        self.verbose = verbose
        self.num_weights = num_weights
        self.metric = metric
        self.index = index
        self.tiedrank = tiedrank

    def fit(self, list_cf, validation, train, trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0]
        self.N_users = self.list_cf[0].N_users        

        weights = np.array([[alpha, 1 - alpha] for alpha in np.linspace(0, 1, self.num_weights)])

        model1 = ValEns(weights, self.metric, self.tiedrank)
        model1.fit([list_cf[self.index[0][0]], list_cf[self.index[0][1]]], validation, train, trainvalidation)
        model2 = ValEns(weights, self.metric, self.tiedrank)
        model2.fit([list_cf[self.index[1][0]], list_cf[self.index[1][1]]], validation, train, trainvalidation)

        self.res_model = ValEns(weights, self.metric, self.tiedrank)
        self.res_model.fit([model1, model2], validation, train, trainvalidation)

    def get_list(self, u):
        return self.res_model.get_list(u)


class RegressionEnsemble:
    def __init__(self, model, tiedrank = False, ratio_neg = 5, verbose=0):
        self.model = copy.deepcopy(model)
        self.ratio_neg = ratio_neg
        self.verbose = verbose
        self.tiedrank = tiedrank

    def fit(self, list_cf, validation, train, trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0].N_items
        self.N_users = len(validation)
        X = []
        y = []

        fij = np.zeros((self.N_items, len(list_cf)))
        for u, items in enumerate(validation):
            if not self.tiedrank:
                for j in range(len(self.list_cf)):
                    fij[:, j] = self.list_cf[j].get_f(u)
                fij -= np.min(fij, axis = 0)[np.newaxis, :]
                fij /= np.max(fij, axis = 0)[np.newaxis, :]
            else:
                for j in range(len(self.list_cf)):
                    fij[:, j] = self.list_cf[j].get_tiedrank(u)
            for i in items:
                X.append(fij[i, :])
                y.append(1)

            for i in range(self.ratio_neg):
                while(True):
                    neg_i = random.randint(0, self.N_items - 1)
                    if neg_i in trainvalidation[u]:
                        continue
                    break
                X.append(fij[neg_i, :])
                y.append(0)

        X = np.array(X)
        y = np.array(y)

        if self.verbose == 1:
            print("data preparation is done")
            sys.stdout.flush()
        self.model.fit(X, y)
        if self.verbose == 1:
            print("fit is done", self.model.coef_)
            sys.stdout.flush()

    def get_list(self, u):
        fij = np.zeros((self.N_items, len(self.list_cf)))
        for j in range(len(self.list_cf)):
            fij[:, j] = self.list_cf[j].get_f(u)
        fij -= np.min(fij, axis = 0)[np.newaxis, :]
        fij /= np.max(fij, axis = 0)[np.newaxis, :]

        res = self.model.predict(fij)
        return np.argsort(-res)




