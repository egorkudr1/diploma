import numpy as np


#don't forget about bugs with skip_local_top!
def get_prec_K(K, train, test,  cls,  skip_train=False, top=None, skip_local_top=0):
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
        
        if skip_local_top>0:
            mask[:skip_local_top] = False
        
        res += np.mean(mask)
        
    res /= len(test)
    return res


def get_one_recal_K(K, train, test,  cls,  skip_train=False, top=None, skip_local_top=0):
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
            
        if skip_local_top>0:
            mask[:skip_local_top] = False
            
        res += int(True in mask)
        
    res /= len(test)
    return res


def get_RR(train, test, ulist, skip_train=False, top=None, skip_local_top=0):
    for rank, item in enumerate(ulist):
        if rank < skip_local_top:
            continue
        if top is not None and item in top:
            continue
        if item in test:
            return 1 / (rank + 1)
        if not skip_train and item in train:
            return 1 / (rank  + 1)
    return 0


def get_MRR(train, test,  cls, skip_train=False, top=None, skip_local_top=0):
    res = 0
    for u, (test_cur, train_cur) in enumerate(zip(test, train)):
        ulist = cls.get_list(u)
        locres = get_RR(train_cur, test_cur, ulist, skip_train, top, skip_local_top)
        res += locres
    res /= len(test)
    return res