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

burglary_with_random_indice   = list(range(0, 50))
pedrobbery_with_random_indice = list(range(3700, 3900))
others_with_random_indice     = list(range(10000, 10056))
indice = burglary_with_random_indice + pedrobbery_with_random_indice + others_with_random_indice

def visualize_on_map(category='other', alpha=1e+2, n=10056):
    # load dataset
    t, s, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=n, category=category)
    # init results
    precisions = []
    recalls    = []
    logliks    = []
    lowerbs    = []
    # only select a small set of data for category other
    if category == 'other':
        t  = np.array([ t[idx] for idx in indice ])
        s  = np.array([ s[idx] for idx in indice ])
        u  = np.array([ u[idx] for idx in indice ])
        m  = np.array([ m[idx] for idx in indice ])
        l  = [ l[idx] for idx in indice ]

    # init mu
    # in order to save time, do the initialization of mu one-time at first
    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set))
    init_em.init_Mu(alpha=alpha)

    utils.plot_intensities4beats(init_em.Mu, u_set, locations=s, labels=l, html_path='%s_intensity_map.html' % category)

def exp_convergence(
        gamma=1., beta=1e-10, alpha=1e+2,
        category='other', epoches=1, iters=20, n=10056):
    # load dataset
    t, _, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=n, category=category)
    # init results
    precisions = []
    recalls    = []
    logliks    = []
    lowerbs    = []
    # only select a small set of data for category other
    if category == 'other':
        t  = np.array([ t[idx] for idx in indice ])
        u  = np.array([ u[idx] for idx in indice ])
        m  = np.array([ m[idx] for idx in indice ])
        l  = [ l[idx] for idx in indice ]
    # init mu
    # in order to save time, do the initialization of mu one-time at first
    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set))
    init_em.init_Mu(gamma=gamma)
    # experiments
    for e in range(epoches):
        em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), alpha=alpha, beta=beta, gamma=gamma)
        em.Mu = init_em.Mu
        ps, rs, lls, lbs = em.fit(T=t[-1], tau=t[0], iters=iters, first_N=500, specific_labels=specific_labels)
        precisions.append(ps)
        recalls.append(rs)
        logliks.append(lls)
        lowerbs.append(lbs)
    precisions = np.array(precisions).mean(axis=0)
    recalls    = np.array(recalls).mean(axis=0)
    logliks    = np.array(logliks).mean(axis=0)
    lowerbs    = np.array(lowerbs).mean(axis=0)
    # save exp results
    np.savetxt("result/%s_precision_convergence.txt" % category, precisions, delimiter=',')
    np.savetxt("result/%s_recalls_convergence.txt" % category, recalls, delimiter=',')
    np.savetxt("result/%s_loglik_convergence.txt" % category, logliks, delimiter=',')
    np.savetxt("result/%s_lowerb_convergence.txt" % category, lowerbs, delimiter=',')

def exp_alpha(
        alpha_range=np.linspace(-15, 0, 51), beta=1e+2, gamma=1.,
        category='other', epoches=5, iters=5,
        csv_filename='data/beats_graph.csv'):
    # load dataset
    t, _, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=10056, category=category)
    # only select a small set of data for category other
    if category == 'other':
        t  = np.array([ t[idx] for idx in indice ])
        u  = np.array([ u[idx] for idx in indice ])
        m  = np.array([ m[idx] for idx in indice ])
        l  = [ l[idx] for idx in indice ]
    # init results
    precisions = []
    recalls    = []
    # in order to save time, do the initialization of mu one-time at first
    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set))
    # init A
    distance_matrix = utils.calculate_beats_pairwise_distance(u_set, csv_filename)
    init_em.init_A(distance_matrix, gamma=gamma)
    # init Mu
    init_em.init_Mu(gamma=gamma)
    # experiments
    for alpha in alpha_range:
        precision = []
        recall    = []
        print('---------alpha = %f ----------' % alpha)
        for e in range(epoches):
            em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta=beta, alpha=alpha)
            em.Mu = init_em.Mu
            em.A  = init_em.A
            ps, rs, _, _ = em.fit(T=t[-1], tau=t[0], iters=iters, first_N=500, specific_labels=specific_labels)
            precision.append(ps[-1])
            recall.append(rs[-1])
        precisions.append(precision)
        recalls.append(recall)
    # save exp results
    np.savetxt("result/%s_precision_alpha_from%dto%d.txt" % (category, min(alpha_range), max(alpha_range)), precisions, delimiter=',')
    np.savetxt("result/%s_recalls_alpha_from%dto%d.txt" % (category, min(alpha_range), max(alpha_range)), recalls, delimiter=',')

def exp_retrival(
        retrieval_range=np.linspace(100, 1000, 51).astype(np.int32), n=10056,
        category='other', epoches=5, iters=5,
        csv_filename='data/beats_graph.csv'):
    # load dataset
    t, _, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=n, category=category)
    # only select a small set of data for category other
    if category == 'other':
        t  = np.array([ t[idx] for idx in indice ])
        u  = np.array([ u[idx] for idx in indice ])
        m  = np.array([ m[idx] for idx in indice ])
        l  = [ l[idx] for idx in indice ]
    # init results
    precisions = []
    recalls    = []

    # in order to save time, do the initialization of mu one-time at first
    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set))
    # init A
    distance_matrix = utils.calculate_beats_pairwise_distance(u_set, csv_filename)
    init_em.init_A(distance_matrix, gamma=1.)
    # init Mu
    init_em.init_Mu(gamma=1.)

    # experiments
    for N in retrieval_range:
        precision = []
        recall    = []
        print('---------N = %d ----------' % N)
        for e in range(epoches):
            em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta=1e+2, alpha=2.)
            em.Mu = init_em.Mu
            em.A  = init_em.A
            ps, rs, _, _ = em.fit(T=t[-1], tau=t[0], iters=iters, first_N=N, specific_labels=specific_labels)
            precision.append(ps[-1])
            recall.append(rs[-1])
        precisions.append(precision)
        recalls.append(recall)
    # save exp results
    np.savetxt("result/%s_precision_N_from%dto%d.txt" % (category, min(retrieval_range), max(retrieval_range)), precisions, delimiter=',')
    np.savetxt("result/%s_recalls_N_from%dto%d.txt" % (category, min(retrieval_range), max(retrieval_range)), recalls, delimiter=',')



if __name__ == '__main__':
    # np.random.seed(0)
    # np.set_printoptions(suppress=True)

    # visualize data on map
    # visualize_on_map(category='other', n=10056)

    # exp_alpha(
    #     alpha_range=np.linspace(0, 20, 51), beta=1e+2, gamma=1.,
    #     category='robbery', epoches=3, iters=1)
    # exp_alpha(
    #     alpha_range=np.linspace(0, 20, 51), beta=1e+2, gamma=1.,
    #     category='burglary', epoches=3, iters=1)
    # exp_alpha(
    #     alpha_range=np.linspace(0, 20, 51), beta=1e+2, gamma=1.,
    #     category='other', epoches=3, iters=1)

    # exp_convergence(beta_1=1., beta_2=1e+2, alpha=1e+2, category='burglary', epoches=1, iters=25, n=350)
    exp_retrival(
        retrieval_range=np.linspace(100, 1000, 51).astype(np.int32), n=10056
        category='robbery', epoches=3, iters=2)
    exp_retrival(
        retrieval_range=np.linspace(100, 1000, 51).astype(np.int32), n=350
        category='burglary', epoches=3, iters=2)
    exp_retrival(
        retrieval_range=np.linspace(100, 1000, 51).astype(np.int32), n=10056
        category='other', epoches=3, iters=2)
