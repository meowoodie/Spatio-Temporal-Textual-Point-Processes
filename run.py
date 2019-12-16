#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Marked Point Process Learning via EM algorithm
'''

import sys
import arrow
import utils
import numpy as np
from vhawkes import VecMarkedMultivarHawkes

burglary_with_random_indice   = list(range(0, 50))
pedrobbery_with_random_indice = list(range(3700, 3900))
others_with_random_indice     = list(range(10000, 10056))
indice = burglary_with_random_indice + pedrobbery_with_random_indice + others_with_random_indice

def exp_baselines(
        retrieval_range=np.linspace(100, 1000, 51).astype(np.int32), n=10056,
        category='other', epoches=1, iters=5,
        csv_filename='data/beats_graph.csv'):
    # load dataset
    t, _, m, l, u, u_set, true_labels = utils.load_police_training_data(n=n, category=category)
    # only select a small set of data for category other
    if category == 'other':
        t  = np.array([ t[idx] for idx in indice ])
        u  = np.array([ u[idx] for idx in indice ])
        m  = np.array([ m[idx] for idx in indice ])
        l  = [ l[idx] for idx in indice ]
    # init results
    precisions = []
    recalls    = []
    # data preparation and configuration
    t     = np.expand_dims((t - min(t) + 1000.) / (max(t) - min(t) + 2000.), -1) # time normalization
    u     = np.expand_dims(u, -1)
    m     = m / 1000.
    seq   = np.concatenate([t, u, m], axis=1)
    n_dim = len(np.unique(u))
    T     = 1.

    # build model
    hawkes = VecMarkedMultivarHawkes(n_dim=n_dim, T=T, seq=seq)
    hawkes.em_fit(iters=2)

    # experiments
    for N in retrieval_range:
        print('---------N = %d ----------' % N)
        precision = []
        recall    = []
        for e in range(epoches):
            p, r = utils.retrieval_test(hawkes, l, true_labels=true_labels, first_N=N)
            print(p, r)
            precision.append(p)
            recall.append(r)
        precisions.append(precision)
        recalls.append(recall)
    
    # save exp results
    # np.savetxt("result/newsttpp+gbrbm1k_%s_precision_N_from%dto%d.txt" % \
    #     (category, min(retrieval_range), max(retrieval_range)), precisions, delimiter=',')
    # np.savetxt("result/newsttpp+gbrbm1k_%s_recalls_N_from%dto%d.txt" % \
    #     (category, min(retrieval_range), max(retrieval_range)), recalls, delimiter=',')



if __name__ == '__main__': 
    # np.random.seed(0)
    # np.set_printoptions(suppress=True)

    exp_baselines(
        retrieval_range=np.linspace(100, 1000, 51).astype(np.int32),
        n=10056, category='robbery', epoches=2)