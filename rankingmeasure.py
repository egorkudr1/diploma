import numpy as np
from sklearn.metrics import roc_auc_score


def wrapper_kmetrics(func):
    def metric(train, test, cls, skip_train=True, top=None, K=5):
        res = 0
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
            
            res += func(mask)
            
        res /= len(test)
        return res
    return metric


@wrapper_kmetrics
def get_prec_K(mask):
    return np.mean(mask)


@wrapper_kmetrics
def get_one_recal_K(mask): 
    return int(True in mask)
        
    
def get_RR(train, test, ulist, skip_train=True, top=None):
    n_skip = 0 
    for rank, item in enumerate(ulist):
        if top is not None and item in top:
            continue
        if item in test:
            return 1 / (rank + 1 - n_skip)
        if item in train:
            if skip_train:
                n_skip += 1
            else:
                return 1 / (rank  + 1)
    return 0


def get_MRR(train, test,  cls, skip_train=True, top=None, K=5):
    res = 0
    for u, (test_cur, train_cur) in enumerate(zip(test, train)):
        ulist = cls.get_list(u)
        locres = get_RR(train_cur, test_cur, ulist, skip_train, top)
        res += locres
    res /= len(test)
    return res


def get_AUC(train, test, cls, skip_train=True, top=None, K=5):
    res = 0
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

        res += roc_auc_score(mask, np.arange(ulist.shape[0])[::-1])
    res /= len(test)
    return res


def get_NDCG(train, test, cls, skip_train=True, top=None, K=5):
    res = 0
    D = 1 / np.log(np.arange(K) + 2)
    maxNDCG = np.sum(D)
    for u, (test_cur, train_cur) in enumerate(zip(test, train)):
        ulist = cls.get_list(u)
        if  skip_train:
            mask = np.in1d(ulist, train_cur)
            ulist = ulist[~mask]
        localK = min(K, ulist.shape[0])
        mask = np.in1d(ulist[:localK], test_cur)
        if not skip_train:
            mask2 = np.in1d(ulist[:localK], train_cur)
            mask = np.logical_or(mask, mask2)
        if top is not None:
            masktmp = np.in1d(ulist[:localK], top)
            mask = np.logical_and(mask, np.logical_not(masktmp))
        res += np.sum(mask * D)
    res /= len(test) * maxNDCG
    return res


@wrapper_kmetrics
def get_MAP(mask):
    n_rel = np.sum(mask)
    if n_rel > 0:
        res = 0
        for k, elem in enumerate(mask):
            res += np.mean(mask[:k + 1]) * elem
        res /= n_rel
    else:
        res = 0
    return res