import numpy as np
import sys
import pickle
from scipy.sparse import csr_matrix
import rankingmeasure


def all_measures(train, test, method, skip_train=True, top=None, K=5):
    list_metrics = [
        ('precK',  rankingmeasure.get_prec_K),
        ('1-call5', rankingmeasure.get_one_recal_K),
        ('MRR', rankingmeasure.get_MRR),
        ('AUC', rankingmeasure.get_AUC),
        ('NGDC', rankingmeasure.get_NDCG),
        ('MAP', rankingmeasure.get_MAP)
    ]
    for name_metric, metric in list_metrics:
        print(name_metric, metric(train, test, method, skip_train=skip_train, top=top, K=K))
        sys.stdout.flush()


def make_valid_data(data, thres=25, reindex=True):
    list_good_user = list(np.where(np.bincount(data[:, 0]) > thres)[0])
    set_good_user = set(list_good_user)
    epinion = []
    for i,j in data:
        if i in set_good_user:
            epinion.append([i, j])
    epinion = np.array(epinion)
    if reindex:
        for new_u, u in enumerate(np.sort(list_good_user)):
            epinion[epinion[:, 0] == u, 0] = new_u
        list_good_item = np.unique(epinion[:, 1])
        for new_i, i in enumerate(np.sort(list_good_item)):
            epinion[epinion[:, 1] == i, 1] = new_i
    user_item = np.max(epinion, axis=0) + 1
    return epinion, user_item


def create_listarray(data):
    xlist = []
    for u in np.sort(np.unique(data[:, 0])):
        xlist.append(data[data[:, 0] == u, 1])
    return xlist


def givenK_train_test(data, K, data_is_xlist=False):
    if data_is_xlist:
        xlist = data.copy()
    else:
        xlist = create_listarray(data)
    
    train = []
    test = []
    for x in xlist:
        train_ind = np.random.choice(x.shape[0], K, replace=False)
        tmp = np.ones(x.shape[0])
        tmp[train_ind] = 0
        test_ind = np.arange(x.shape[0])[tmp == 1]
        
        train.append(x[train_ind])
        test.append(x[test_ind])
        
    return (train, test, xlist)


def ratio_train_test(data, ratio_test, data_is_xlist=False):
    if data_is_xlist:
        xlist = data.copy()
    else:
        xlist = create_listarray(data)

    train = []
    test = []
    for x in xlist:
        train_ind = np.random.choice(x.shape[0], round(ratio_test * x.shape[0]), replace=False)
        tmp = np.ones(x.shape[0])
        tmp[train_ind] = 0
        test_ind = np.arange(x.shape[0])[tmp == 1]
        
        train.append(x[train_ind])
        test.append(x[test_ind])
    
    return (train, test, xlist)        


def create_csr(data, user_item, file_name):
    inf = []
    row = []
    col = []
    for u, items in enumerate(data):
        for i in items:
            inf.append(1)
            row.append(u)
            col.append(i)
    new_data = csr_matrix((inf,(row, col)), shape=user_item).astype('float')
    with open(file_name, 'wb') as f:
        pickle.dump(new_data, f, 0)


def create_original_sample(data, index):
    res = []
    for i in index:
        res.append(data[i])
    return res

