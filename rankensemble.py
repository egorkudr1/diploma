import numpy as np
import pandas as pd
import copy
import sys
from sklearn import linear_model
import copy
import random
import rankingmeasure


# Это класс, который хранит алгоритмы ранжирования вместе.
# Позволяет хранить методы в обученном состоянии, а также проводить некоторый аналитику
class ensemble:
    def __init__(self, list_cf, verbose=0):
        self.list_cf = copy.deepcopy(list_cf)
        self.verbose = verbose
    
    #обучение всех алгоритмов обучения
    def fit(self, data):
        for i in range(len(self.list_cf)):
            self.list_cf[i].fit(data)
            if self.verbose == 1:
                print("iteration", i)
                sys.stdout.flush()
    
    #показ первых topK предметов, которые выдал каждый из алгоритмов ранжирования
    def show_all_ulist(self, u, topK=5):
        for i in range(len(self.list_cf)):
            print(type(self.list_cf[i]).__name__, self.list_cf[i].get_list(u)[:topK])

    #попарное сравнение первых topK множеств предметов, которые выдл каждый 
    #из алгоритмов ранжирования, для пользователя u.
    #не учитывает поряд.
    def matrix_similar(self, u, topK=5):
        res = []
        for i in range(len(self.list_cf)):
            res.append(self.list_cf[i].get_list(u)[:topK])

        ans = np.zeros((len(res), len(res)))
        for i in range(ans.shape[0]):
            for j in range(i + 1, ans.shape[1]):
                ans[i, j] = np.mean(np.in1d(res[i], res[j]))
                ans[j, i] = ans[i, j]
        return ans

    #Уредненное значение метода matrix_similar по всем пользователям.
    def mean_matrix_similar(self, topK=5):
        N_users = self.list_cf[0].N_users
        for u in range(N_users):
            if u == 0:
                ans = matrix_similar(self, u, topK)
            else:
                ans += matrix_similar(self, u, topK)
        ans /= N_users
        return ans

 
class top_ensemble:
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


class rating_ensemble:
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
        #self.index = np.arange(self.N_items)[::-1] + 1
        self.index = np.exp(-self.alpha * np.arange(self.N_items))


    def get_list(self, u):
        res = np.zeros(self.N_items)
        for i in range(self.N_methods):
            ranks = self.list_cf[i].get_list(u)
            res[ranks] += self.weights[i] * self.index 
        return np.argsort(-res)


class inner_value_ensemble:
    def __init__(self, weights = None, verbose=0):
        self.verbose = verbose
        self.weights = weights


    def fit(self, list_cf, validation, train,  trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        if self.weights is None:
            self.weights = np.ones(len(list_cf))/len(list_cf)
        else:
            assert(self.weights.shape[0] == len(list_cf))

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


class val_ens:
    def __init__(self, weights, metric, verbose=0):
        self.verbose = verbose
        self.weights = weights
        self.metric = metric


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
            fij = self.list_cf[i].get_f(u)
            fij -= np.min(fij)
            fij /= np.max(fij)
            res += alpha * fij 
        return np.argsort(-res)


    def get_f(self, u):
        res = np.zeros(self.N_items)
        for i, alpha in enumerate(self.w):
            fij = self.list_cf[i].get_f(u)
            fij -= np.min(fij)
            fij /= np.max(fij)
            res += alpha * fij
        return res


class boost_val_ens:
    def __init__(self,  metric, index, num_weights = 51, verbose=0):
        self.verbose = verbose
        self.num_weights = num_weights
        self.metric = metric
        self.index = index


    def fit(self, list_cf, validation, train, trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0]
        self.N_users = self.list_cf[0].N_users

        weights = np.array([[alpha, 1 - alpha] for alpha in np.linspace(0, 1, self.num_weights)])

        self.res_model = [copy.deepcopy(self.list_cf[self.index[0]])]
        for i in range(1, len(list_cf)):
            self.res_model.append(val_ens(weights, self.metric))
            self.res_model[i].fit([self.res_model[i - 1], self.list_cf[self.index[i]]], validation, train, trainvalidation)
        

    def get_list(self, u):
        return self.res_model[-1].get_list(u)


class tree_val_ens:
    def __init__(self, metric, index, num_weights=51, verbose=0):
        self.verbose = verbose
        self.num_weights = num_weights
        self.metric = metric
        self.index = index


    def fit(self, list_cf, validation, train, trainvalidation):
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0]
        self.N_users = self.list_cf[0].N_users        

        weights = np.array([[alpha, 1 - alpha] for alpha in np.linspace(0, 1, self.num_weights)])

        model1 = val_ens(weights, self.metric, [0, 1])
        model1.fit([list_cf[self.index[0][0]], list_cf[self.index[0][1]]], validation, train, trainvalidation)
        model2 = val_ens(weights, self.metric, [0, 1])
        model2.fit([list_cf[self.index[1][0]], list_cf[self.index[1][1]]], validation, train, trainvalidation)

        self.res_model = val_ens(weights, self.metric, [0, 1])
        self.res_model.fit([model1, model2], validation, train, trainvalidation)


    def get_list(self, u):
        return self.res_model.get_list(u)

# def find_alpha(a1, a2, index = None):
#     if index is None:
#         all_pairs = np.transpose([np.tile(a1, a2.shape[0]), np.tile(a2, a1.shape[0]), 
#                       np.repeat(a1, a2.shape[0]), np.repeat(a2, a1.shape[0])])
#     else:
#         all_pairs = np.transpose([np.tile(a1[index], a2.shape[0]), np.tile(a2[index], a1.shape[0]), 
#                       np.repeat(a1, len(index)), np.repeat(a2, len(index))])
#     # print(all_pairs.shape, all_pairs)
#     alpha = (- all_pairs[:, 1]  + all_pairs[:, 3]) / (all_pairs[:, 0] - all_pairs[:, 1] - all_pairs[:, 2]  + all_pairs[:, 3])
#     alpha = alpha[np.logical_not(np.isnan(alpha))]
#     alpha = alpha[alpha > 0 ]
#     alpha = alpha[alpha < 1]
#     return np.unique(alpha)


# class smart_val_ens:
#     def __init__(self, maxn = 1000, topK = 5, verbose=0):
#         self.maxn = maxn
#         self.verbose = verbose
#         self.topK = topK


#     def fit(self, list_cf, validation, train, trainvalidation):
#         self.list_cf = copy.deepcopy(list_cf)
#         self.N_items = self.list_cf[0].N_items
#         self.N_users = self.list_cf[0].N_users


#         alpha = []
#         all_items = np.arange(self.N_items)
#         for u in range(self.N_users):
#             rank1 = self.list_cf[0].get_list(u)[:self.topK]
#             rank2 = self.list_cf[1].get_list(u)[:self.topK]

#             f1 = self.list_cf[0].get_f(u)
#             f2 = self.list_cf[1].get_f(u)

#             changed_index2 = f2 >= np.min(f2[rank1])
#             changed_index1 = f1 >= np.min(f1[rank2])

#             f1_1 = f1[changed_index1]
#             f2_1 = f2[changed_index1]
#             rank1 = np.argsort(-f1_1)

#             f1_2 = f1[changed_index2]
#             f2_2 = f2[changed_index2]
#             rank2 = np.argsort(-f2_2)   

#             totaln1 = min(np.sum(changed_index1), self.maxn)
#             totaln2 = min(np.sum(changed_index2), self.maxn)

#             l1 = find_alpha(f1_1, f2_1, rank1) 
#             l2 = find_alpha(f2_2, f1_2, rank2)
#             print(l1, l2)
#             print(l1.shape, l2.shape)
#             loc_alpha = np.unique(loc_alpha)
#             if self.verbose == 2:
#                 print(u, loc_alpha.shape, len(alpha))
#                 sys.stdout.flush()
#             alpha.expand(loc_alpha)




class regression_ensemble:
    def __init__(self, model, ratio_neg = 5, verbose=0):
        self.model = copy.deepcopy(model)
        self.ratio_neg = ratio_neg
        self.verbose = verbose


    def fit(self, list_cf, validation, train, trainvalidation):
        # numpos = 0
        # for items in validation:
        #     numpos += len(items)
        # numpos *= len(list_cf)
        # X = np.zeros((numpos * (1 + ratio_neg), len(list_cf)))
        # y = np.zeros((numpos * (1 + ratio_neg), len(list_cf)))
        self.list_cf = copy.deepcopy(list_cf)
        self.N_items = self.list_cf[0].N_items
        self.N_users = len(validation)
        X = []
        y = []

        fij = np.zeros((self.N_items, len(list_cf)))
        for u, items in enumerate(validation):
            for j in range(len(self.list_cf)):
                fij[:, j] = self.list_cf[j].get_f(u)
            fij -= np.min(fij, axis = 0)[np.newaxis, :]
            fij /= np.max(fij, axis = 0)[np.newaxis, :]
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

        # print(fij.shape)
        # print(self.N_items)
        # res = self.model.predict_proba(fij)[1]
        res = self.model.predict(fij)
        # print(res.shape)
        return np.argsort(-res)




