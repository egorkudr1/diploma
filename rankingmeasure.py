import numpy as np
from sklearn.metrics import roc_auc_score


def __prec_K(mask):
    return np.mean(mask)


def __one_recal_K(mask): 
    return int(True in mask)


def __MRR(mask):
    index = np.where(mask == 1)
    return 1 / (np.min(index) + 1)


def __AUC(mask):
    ranks = np.arange(mask.shape[0])[::-1]
    return roc_auc_score(mask,  ranks)   


def __MAP(mask):
    n_rel = np.sum(mask)
    if n_rel > 0:
        res = 0
        for k, elem in enumerate(mask):
            res += np.mean(mask[:k + 1]) * elem
        res /= n_rel
    else:
        res = 0
    return res


def __NGDC(mask):
    D = 1 / np.log(np.arange(mask.shape[0]) + 2)
    maxNDCG = np.sum(D)
   
    return np.sum(mask * D) / maxNDCG


def wrapper_kmetrics(func):
    def metric(train, test, cls, skip_train=True, top=None, K=5):
        for u, (test_cur, train_cur) in enumerate(zip(test, train)):
            ulist = cls.get_list(u)
            if  skip_train:
                mask = np.in1d(ulist, train_cur)
                ulist = ulist[~mask]
            
            mask = np.in1d(ulist[:K], test_cur)
            if not skip_train:
                mask2 = np.in1d(ulist[:K], train_cur)
                mask = np.logical_or(mask, mask2)
            
            if top is not None:
                masktmp = np.in1d(ulist[:K], top)
                mask = np.logical_and(mask, np.logical_not(masktmp))
            
            if u == 0:
                res = func(mask)
            else:
                res += func(mask)
            
        res /= len(test)
        return res
    return metric


@wrapper_kmetrics
def get_prec_K(mask):
    return __prec_K(mask)


@wrapper_kmetrics
def get_one_recal_K(mask): 
    return __one_recal_K(mask)


@wrapper_kmetrics
def get_NDCG(mask):
    return __NGDC(mask)


@wrapper_kmetrics
def get_MAP(mask):
    return __MAP(mask)


@wrapper_kmetrics
def get_Kmetrics(mask):
    return np.array([__prec_K(mask), __one_recal_K(mask), __NGDC(mask), __MAP(mask)])


def wrapper_listmetrics(func):
    def metric(train, test, cls, skip_train=True, top=None, K=5):
        for u, (test_cur, train_cur) in enumerate(zip(test, train)):
            ulist = cls.get_list(u)
            if  skip_train:
                mask = np.in1d(ulist, train_cur)
                ulist = ulist[~mask]
            
            mask = np.in1d(ulist, test_cur)
            if not skip_train:
                mask2 = np.in1d(ulist, train_cur)
                mask = np.logical_or(mask, mask2)
            
            if top is not None:
                masktmp = np.in1d(ulist, top)
                mask = np.logical_and(mask, np.logical_not(masktmp))
            
            if u == 0:
                res = func(mask)
            else:
                res += func(mask)
            
        res /= len(test)
        return res
    return metric


@wrapper_listmetrics
def get_MRR(mask):
    return __MRR(mask)


@wrapper_listmetrics
def get_AUC(mask):
    return __AUC(mask)


@wrapper_listmetrics
def get_listmetrics(mask):
    return np.array([__MRR(mask), __AUC(mask)])







