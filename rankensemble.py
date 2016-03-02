import numpy as np
import pandas as pd
import copy
import sys


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