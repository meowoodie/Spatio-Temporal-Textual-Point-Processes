#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Marked Point Process Learning via EM algorithm
'''

import sys
import arrow
import utils
import numpy as np

class MPPEM(object):
    '''
    Marked Point Process Learning via EM algorithm
    '''

    def __init__(self, d, seq_t, seq_u, seq_l, seq_m=None, beta_1=1., beta_2=1.):
        # training data
        self.t      = seq_t # time of each of events
        self.u      = seq_u # component of each of events
        self.m      = seq_m # marks (feature vectors) of each of events
        self.l      = seq_l # labels of each of events
        # basic configuration
        self.T0     = self.t[0]
        self.Tn     = self.t[-1]
        self.n      = len(self.t) # number of events
        self.d      = d           # number of components
        # parameters for intensity kernel
        self.beta_1 = beta_1                                    # parameter for intensity kernel
        self.beta_2 = beta_2
        self.A      = np.random.uniform(0, 1, (self.d, self.d)) # influential matrix for intensity kernel
        self.A_mask = np.ones((self.d, self.d))                 # mask for influential matrix
        self.Mu     = np.random.uniform(0, 1, self.d)           # background rates for intensity kernel
        self.P      = np.ones((self.n, self.n)) * -1            # transition probability matrix
                                                                # -1 means uninitiated value
        # normalization
        self.t      = (self.t - self.T0) / (self.Tn - self.T0)

    def _init_P(self, t_indices):
        '''init transition probability matrix'''
        # values for initiated P
        valid_Pij  = [ self.P[i][j]
            for i in t_indices
            for j in t_indices[t_indices<i]
            if self.P[i][j] != -1 ]
        # positions for uninitiated P
        invalid_ij = [ [i, j]
            for i in t_indices
            for j in t_indices[t_indices<i]
            if self.P[i][j] == -1 ]
        # initiate P for the positions where P is uninitiated
        if len(invalid_ij) > 0:
            init_val = (1. - sum(valid_Pij)) / len(invalid_ij)
            for i, j in invalid_ij:
                self.P[i][j] = init_val + np.random.normal(0, 1e-8, 1)[0]

    def init_Mu(self, alpha):
        '''init background rate Mu for each of the components'''
        # all data will be used in estimating the initial Mu.
        u_window  = self.u # [t_indices]
        U_freq    = [ len(u_window[u_window == uj]) for uj in range(self.d) ]
        U_dist    = [ U_freq[uj] / sum(U_freq) for uj in range(self.d) ]
        self.Mu   = np.array(U_dist) * alpha

    def _slide_window_indices(self, T, tau):
        '''select the indices of the sequence within the slide window'''
        return np.where((self.t < T) & (self.t >= tau))[0]

    def _loglik_subterm_1(self, i, t_indices):
        '''subterm 1 in log-likelihood function'''
        terms = [
            self.A[self.u[i]][self.u[j]] * self.beta_1 * np.exp(-1 * self.beta_2 * (self.t[i] - self.t[j]) + np.inner(self.m[i], self.m[j]))
            for j in t_indices[t_indices<i] ]
        return np.array(terms)

    def _loglik_subterm_2(self, uj, t_indices, T):
        '''subterm 2 in log-likelihood function'''
        terms = [
            self.A[self.u[i]][uj] * (1 - np.exp(- self.beta_2 * (T - self.t[i])) + np.inner(self.m[i], self.m[t_indices[-1]]))
            for i in t_indices]
        return np.array(terms)

    def log_likelihood(self, T, tau):
        '''log-likelihood function given t (time sequence) and u (component sequence)'''
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        term_1 = [
            np.log(self.Mu[self.u[i]] + self._loglik_subterm_1(i, t_indices).sum())
            for i in t_indices ]
        term_2 = [ self.Mu[uj] * (T - tau) for uj in range(self.d) ]
        term_3 = [ self._loglik_subterm_2(uj, t_indices, T).sum() for uj in range(self.d)]
        loglik = sum(term_1) + sum(term_2) + sum(term_3)
        return loglik

    def jensens_lower_bound(self, T, tau):
        '''lower bound for the log-likelihood function according to the jensen's inequality'''
        # get time indices of the indicated window
        t_indices        = self._slide_window_indices(T, tau)
        t_indices_before = lambda i: t_indices[t_indices<i]
        term_1 = [
            self.P[i][i] * np.log(self.Mu[self.u[i]]) + \
            (self.P[i][t_indices_before(i)] * np.log(self._loglik_subterm_1(i, t_indices) + 1e-100)).sum() - \
            (self.P[i][t_indices_before(i)] * np.log(self.P[i][t_indices_before(i)] + 1e-100)).sum()
            for i in t_indices ]
        term_2 = [ self.Mu[uj] * (T - tau) for uj in range(self.d) ]
        term_3 = [ self._loglik_subterm_2(uj, t_indices, T).sum() for uj in range(self.d)]
        lowerb = sum(term_1) + sum(term_2) + sum(term_3)
        return lowerb

    def _update_P(self, i, j, T, tau):
        '''update transition probability matrix P'''
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        if i > j and j in t_indices and i in t_indices:
            numerator    = self.A[self.u[i]][self.u[j]] * self.beta_1 * np.exp(-1 * self.beta_2 * (self.t[i] - self.t[j]) + np.inner(self.m[i], self.m[j]))
            denominator  = self.Mu[self.u[i]] + self._loglik_subterm_1(i, t_indices).sum()
            # print('%f, %f' % (numerator, denominator))
            self.P[i][j] = numerator / ( denominator * len(t_indices) )
            # print('P(%d, %d) = %f' % (i, j, self.P[i][j]))

    def _update_A(self, u, v, T, tau):
        '''update influential matrix A'''
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        # check if A_{u,v} is available
        if self.A_mask[u][v]:
            numerator = []
            for i in t_indices[np.where(self.u[t_indices] == u)[0]]:
                for j in t_indices[np.where(self.u[t_indices] == v)[0]]:
                    if j < i:
                        numerator.append(self.P[i][j])
            denominator = [ 1 - np.exp(- self.beta_2 * (T - self.t[j]))
                for j in t_indices if self.u[j] == v ]
            if len(numerator) == 0 or len(denominator) == 0:
                self.A_mask[u][v] = 0
                return
            numerator    = sum(numerator)
            denominator  = sum(denominator)
            self.A[u][v] = numerator / denominator
            # print('A(%d, %d) = %f' % (u, v, self.A[u][v]))

    def retrieval_test(self, t_indices, specific_labels=None, first_N=100):
        '''get precision and recall of retrieval test'''
        # only do the test on the specific labels if specific_labels is not None
        if specific_labels:
            specific_label_cond = lambda i: self.l[i] in specific_labels
        else:
            specific_label_cond = lambda i: True
        # alerts = [ (i, j) for i in t_indices for j in range(i) if self.P[i][j] >= threshold ]
        # miss   = [ (i, j) for i in t_indices for j in range(i) if self.P[i][j] < threshold and self.l[i] == self.l[j] ]
        # recall = len(hits) / (len(hits) + len(miss)) if (len(hits) + len(miss)) != 0 else 0.
        # get all the valid pairs
        pairs = [ [ self.P[i][j], i, j ] for i in t_indices for j in range(i) ]
        pairs = np.array(pairs)
        pairs = pairs[pairs[:, 0].argsort()]
        print(len(pairs))
        # get retrieve, hits and relevant
        retrieve  = pairs[-first_N:, [1, 2]].astype(np.int32)
        hits      = [ (i, j) for i, j in retrieve if self.l[i] == self.l[j] and specific_label_cond(i) ]
        relevant  = [ (i, j) for i in t_indices for j in range(i) if self.l[i] == self.l[j] and specific_label_cond(i) ]
        print(len(relevant))
        # get precision and recall
        precision = len(hits) / len(retrieve) if len(retrieve) != 0 else 0.
        recall    = len(hits) / len(relevant) if len(relevant) != 0 else 0.
        return len(retrieve), precision, recall

    def fit(self, T, tau, epoches=100, first_N=100, specific_labels=None):
        '''
        maximize the lower bound of the loglikelihood function by estimating
        matrix P and matrix A iteratively.
        '''
        # F-1 score
        F_1 = lambda p, r: 2 * p * r / (p + r) if (p + r) != 0. else 0.
        # normalization
        T   = (T - self.T0) / (self.Tn - self.T0)
        tau = (tau - self.T0) / (self.Tn - self.T0)
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        print('[%s] %d points will be fitted.' % (arrow.now(), len(t_indices)))
        # init P
        self._init_P(t_indices)
        self.check_P(t_indices)
        n_alerts, init_precision, init_recall = self.retrieval_test(t_indices, specific_labels=specific_labels, first_N=first_N)
        print('[%s] epoch %d\tlower bound:\t%f' % (arrow.now(), 0, self.log_likelihood(T, tau)))
        print('[%s] \t\tnum of alerts:%d,\tprecision:\t%f,\trecall:\t%f,\tF-1 score:\t%f.' % \
            (arrow.now(), n_alerts, init_precision, init_recall, F_1(init_precision, init_recall)))
        # training epoches
        for e in range(epoches):
            # update P matrix:
            for i in t_indices:
                for j in t_indices[t_indices<i]:
                    self._update_P(i, j, T, tau)
            # update A matrix:
            for u in range(self.d):
                for v in range(self.d):
                    self._update_A(u, v, T, tau)
            self.check_P(t_indices)
            n_alerts, precision, recall = self.retrieval_test(t_indices, specific_labels=specific_labels, first_N=first_N)
            print('[%s] epoch %d\tlower bound:\t%f' % (arrow.now(), e+1, self.log_likelihood(T, tau)))
            print('[%s] \t\tnum of alerts:%d,\tprecision:\t%f,\trecall:\t%f,\tF-1 score:\t%f.' % \
                (arrow.now(), n_alerts, precision, recall, F_1(precision, recall)))
        return init_precision, init_recall, precision, recall

    def save(self, path):
        np.savetxt(path + "P.txt", self.P, delimiter=',')
        np.savetxt(path + "A.txt", self.A, delimiter=',')
        np.savetxt(path + "Mu.txt", self.Mu, delimiter=',')

    def check_P(self, t_indices):
        test = [ self.P[i][j]
            for i in t_indices
            for j in t_indices[t_indices<i] ]
        # print(test)
        print(sum(test))

if __name__ == '__main__':
    # np.random.seed(0)
    # np.set_printoptions(suppress=True)
    epoches  = 5
    iters    = 1
    category = 'other'
    t, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=10056, category=category)

    init_precision  = 0
    init_recall     = 0
    precisions      = []
    recalls         = []

    init_em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=1., beta_2=1.)
    init_em.init_Mu(alpha=1e+2)

    # t = t[0:50] + t[3700:3900] + t[10000, 10056]
    # u = u[0:50] + u[3700:3900] + u[10000, 10056]
    # l = l[0:50] + l[3700:3900] + l[10000, 10056]
    # m = m[0:50] + m[3700:3900] + m[10000, 10056]
    burglary_with_random_indice   = list(range(0, 50))
    pedrobbery_with_random_indice = list(range(3700, 3900))
    others_with_random_indice     = list(range(10000, 10056))
    indice = burglary_with_random_indice + pedrobbery_with_random_indice + others_with_random_indice
    t  = np.array([ t[idx] for idx in indice ])
    u  = np.array([ u[idx] for idx in indice ])
    m  = np.array([ m[idx] for idx in indice ])
    l  = [ l[idx] for idx in indice ]

    for beta in np.linspace(0, 10, 11):
        precision = []
        recall    = []
        print('---------beta = 10^%f ----------' % beta)
        for e in range(epoches):
            em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=beta, beta_2=10**2)
            em.Mu = init_em.Mu
            init_p, init_r, p, r = em.fit(T=t[-1], tau=t[0], epoches=iters, first_N=500, specific_labels=specific_labels)
            precision.append(p)
            recall.append(r)
        precisions.append(precision)
        recalls.append(recall)

    np.savetxt("result/%s_precision_beta1_from-5to10.txt" % category, precisions, delimiter=',')
    np.savetxt("result/%s_recalls_beta1_from-5to10.txt" % category, recalls, delimiter=',')

    print(init_p)
    print(init_r)

    category = 'robbery'
    t, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=350, category=category)

    init_precision  = 0
    init_recall     = 0
    precisions      = []
    recalls         = []

    for beta in np.linspace(0, 10, 11):
        precision = []
        recall    = []
        print('---------beta = 10^%f ----------' % beta)
        for e in range(epoches):
            em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set), beta_1=beta, beta_2=10**2)
            em.Mu = init_em.Mu
            init_p, init_r, p, r = em.fit(T=t[-1], tau=t[0], epoches=iters, first_N=500, specific_labels=specific_labels)
            precision.append(p)
            recall.append(r)
        precisions.append(precision)
        recalls.append(recall)

    np.savetxt("result/%s_precision_beta1_from-5to10.txt" % category, precisions, delimiter=',')
    np.savetxt("result/%s_recalls_beta1_from-5to10.txt" % category, recalls, delimiter=',')

    print(init_p)
    print(init_r)
