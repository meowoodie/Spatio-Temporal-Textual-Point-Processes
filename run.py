#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Marked Point Process Learning via EM algorithm
'''

import sys
import arrow
import utils
import numpy as np

from mppem import MPPEM

if __name__ == '__main__':
    # np.random.seed(0)
    # np.set_printoptions(suppress=True)

    epoches  = 5
    iters    = 5

    category = 'other'
    t, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=10056, category=category)

    init_precision  = 0
    init_recall     = 0
    precisions      = []
    recalls         = []

    burglary_with_random_indice   = list(range(0, 50))
    pedrobbery_with_random_indice = list(range(3700, 3900))
    others_with_random_indice     = list(range(10000, 10056))
    indice = burglary_with_random_indice + pedrobbery_with_random_indice + others_with_random_indice
    t  = np.array([ t[idx] for idx in indice ])
    u  = np.array([ u[idx] for idx in indice ])
    m  = np.array([ m[idx] for idx in indice ])
    l  = [ l[idx] for idx in indice ]

    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=1., beta_2=1.)
    init_em.init_Mu(alpha=1e+2)

    for beta in np.linspace(-15, 0, 51):
        precision = []
        recall    = []
        print('---------beta = 10^%f ----------' % beta)
        for e in range(epoches):
            em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=10**beta, beta_2=10**2)
            em.Mu = init_em.Mu
            init_p, init_r, p, r = em.fit(T=t[-1], tau=t[0], epoches=iters, first_N=500, specific_labels=specific_labels)
            precision.append(p)
            recall.append(r)
        precisions.append(precision)
        recalls.append(recall)

    np.savetxt("result/%s_precision_beta1_from-15to0.txt" % category, precisions, delimiter=',')
    np.savetxt("result/%s_recalls_beta1_from-15to0.txt" % category, recalls, delimiter=',')

    print(init_p)
    print(init_r)

    category = 'robbery'
    t, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=350, category=category)

    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=1., beta_2=1.)
    init_em.init_Mu(alpha=1e+2)

    init_precision  = 0
    init_recall     = 0
    precisions      = []
    recalls         = []

    for beta in np.linspace(-15, 0, 51):
        precision = []
        recall    = []
        print('---------beta = 10^%f ----------' % beta)
        for e in range(epoches):
            em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=10**beta, beta_2=10**2)
            em.Mu = init_em.Mu
            init_p, init_r, p, r = em.fit(T=t[-1], tau=t[0], epoches=iters, first_N=500, specific_labels=specific_labels)
            precision.append(p)
            recall.append(r)
        precisions.append(precision)
        recalls.append(recall)

    np.savetxt("result/%s_precision_beta1_from-15to0.txt" % category, precisions, delimiter=',')
    np.savetxt("result/%s_recalls_beta1_from-15to0.txt" % category, recalls, delimiter=',')

    print(init_p)
    print(init_r)
