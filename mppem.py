#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Marked Point Process Learning via EM algorithm
'''

import arrow
import numpy as np

class VecMarkedMultivarHawkes(object):
    '''
    Multivariate Hawkes Processes with Vectorized Marks 

    Reference: https://arxiv.org/pdf/1902.00440.pdf
    '''

    # def __init__(self, d, seq_t, seq_u, seq_l, seq_m=None, beta=1.):
    #     # training data
    #     self.t      = seq_t # time of each of events
    #     self.u      = seq_u # component of each of events
    #     self.m      = seq_m # marks (feature vectors) of each of events
    #     self.l      = seq_l # labels of each of events
    #     # basic configuration
    #     self.T0     = self.t[0]
    #     self.Tn     = self.t[-1]
    #     self.n      = len(self.t) # number of events
    #     self.d      = d           # number of components
    #     # parameters for intensity kernel
    #     self.beta   = beta                             # parameter for intensity kernel
    #     self.A      = np.zeros((self.d, self.d))       # influential matrix for intensity kernel
    #     self.A_mask = np.ones((self.d, self.d))        # mask for influential matrix
    #     self.Mu     = np.random.uniform(0, 1, self.d)  # background rates for intensity kernel
    #     self.P      = np.ones((self.n, self.n)) * -1   # transition probability matrix
    #                                                    # -1 means uninitiated value
    #     # normalization
    #     self.t      = (self.t - self.T0) / (self.Tn - self.T0)

    def __init__(self, n_dim, T, seq, beta=1.):
        '''
        Params:
        - n_dim:   the number of dimensions of the sequence.
        - T:       the time horizon of the sequence.
        - seq:     the sequence of points with shape (n_point, 1+1+len_vec) where 
                   the last dimension indicates time, dimension and mark, respectively.
        - beta:    (optional) temporal decaying factor.
        '''
        self.seq   = seq
        self.T     = T
        self.n_dim = n_dim
        self.beta  = beta
        # configuration
        n_point = self.seq.shape[0]     # the number of points in the sequence
        len_vec = self.seq.shape[1] - 2 # the length of the mark vector
        self.P  = self._random_init_P(n_point)
        self.A  = self._random_init_A(self.n_dim)
        self.Mu = self._prop_init_Mu(self.n_dim, self.seq, self.T)

    @staticmethod
    def _random_init_P(n_point):
        '''
        Uniformly initiate triggerring probability matrix P where each entry p_{ij}
        indicates how likely point i is triggered by point j (j < i). In addition,
        p_{ii} indicate the point i is triggered by the background. 

        The matrix also requires that the sum of entries P_{ij}, j < i for each i 
        equals to 1. 
        '''
        P = np.random.uniform(low=0., high=1., size=(n_point, n_point))
        for i in range(n_point):
            P[i, :i+1] /= P[i, :i+1].sum()
            P[i, i+1:] = 0
        return P

    @staticmethod
    def _random_init_A(n_dim, scale=1e-5):
        '''
        Uniformly initiate covariance matrix A_{uv} where each entry indicate the 
        covariance between dimension u and dimension v.
        '''
        return np.random.uniform(low=0., high=1.*scale, size=(n_dim, n_dim))

    @staticmethod
    def _prop_init_Mu(n_dim, seq, T):
        '''
        Initiate the background intensity Mu using the number of points in data 
        that falls onto the dimension u divided by the size of the time horizon 
        (i.e. lambda value if we consider each dimension as an individual poisson 
        process). The initialized Mu vector is proportional to numbers of points
        in each dimension.
        '''
        # all data will be used in estimating the initial Mu.
        Mu            = np.zeros(n_dim)
        set_s, counts = np.unique(seq[:, 1].astype(np.int32), return_counts=True)
        for s, c in zip(set_s, counts):
            Mu[s] = c / T
        return Mu

    def _h(self, i, j):
        '''
        h_{ij} denotes a special term: 
            A_{s_i, s_j} * beta * exp(- beta * (t_i - t_j)) h_i' * h_j.
        This term will be further used in P matrix update. 

        See the definition of h_{ij} at Appendix D in the reference.
        '''
        h_ij = self.A[self.seq[i, 1], self.seq[j, 1]] * \
               self.beta * np.exp(- self.beta * (self.seq[i, 0] - self.seq[j, 0])) * \
               np.dot(self.seq[i, 2:], self.seq[j, 2:])
        return h_ij
    
    def _update_P(self, i, j):
        '''
        Update P_{ij} using an EM-like algorithm by maximizing the lower bound 
        of the log-likelihood function. 
        '''
        if i == j:
            self.P[i, i] = self.Mu[i] / self. 

    # def _slide_window_indices(self, T, tau):
    #     '''select the indices of the sequence within the slide window'''
    #     return np.where((self.t < T) & (self.t >= tau))[0]

    # def _loglik_subterm_1(self, i, t_indices):
    #     '''subterm 1 in log-likelihood function'''
    #     terms = [
    #         self.A[self.u[i]][self.u[j]] * 1. * np.exp(-1 * self.beta * (self.t[i] - self.t[j]) + np.inner(self.m[i], self.m[j]))
    #         for j in t_indices[t_indices<i] ]
    #     return np.array(terms)

    # def _loglik_subterm_2(self, uj, t_indices, T):
    #     '''subterm 2 in log-likelihood function'''
    #     terms = [
    #         self.A[self.u[i]][uj] * (1 - np.exp(- self.beta * (T - self.t[i])) + np.inner(self.m[i], self.m[t_indices[-1]]))
    #         for i in t_indices]
    #     return np.array(terms)

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
            numerator    = (self.A[self.u[i]][self.u[j]]) * 1. * np.exp(-1 * self.beta * (self.t[i] - self.t[j]) + np.inner(self.m[i], self.m[j]))
            denominator  = self.Mu[self.u[i]] + self._loglik_subterm_1(i, t_indices).sum()
            self.P[i][j] = numerator / denominator

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
            denominator = [ 1 - np.exp(- self.beta * (T - self.t[j]))
                for j in t_indices if self.u[j] == v ]
            if len(numerator) == 0 or len(denominator) == 0:
                self.A_mask[u][v] = 0
                return
            numerator    = sum(numerator)
            denominator  = sum(denominator)
            self.A[u][v] = (numerator / denominator)
            # print('A(%d, %d) = %f' % (u, v, self.A[u][v]))

    def retrieval_test(self, t_indices, specific_labels=None, first_N=100):
        '''get precision and recall of retrieval test'''
        # only do the test on the specific labels if specific_labels is not None
        if specific_labels:
            specific_label_cond = lambda i: self.l[i] in specific_labels
        else:
            specific_label_cond = lambda i: True
        # get all the valid pairs
        pairs = [ [ self.P[i][j], i, j ] for i in t_indices for j in range(i) ]
        pairs = np.array(pairs)
        pairs = pairs[pairs[:, 0].argsort()]
        # print(len(pairs))
        # get retrieve, hits and relevant
        retrieve  = pairs[-first_N:, [1, 2]].astype(np.int32)
        hits      = [ (i, j) for i, j in retrieve if self.l[i] == self.l[j] and specific_label_cond(i) ]
        relevant  = [ (i, j) for i in t_indices for j in range(i) if self.l[i] == self.l[j] and specific_label_cond(i) ]
        # print(len(relevant))
        # get precision and recall
        precision = len(hits) / len(retrieve) if len(retrieve) != 0 else 0.
        recall    = len(hits) / len(relevant) if len(relevant) != 0 else 0.
        return len(retrieve), precision, recall

    def fit(self, T, tau, iters=100, first_N=100, specific_labels=None):
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
        # n_alerts, init_precision, init_recall = self.retrieval_test(t_indices, specific_labels=specific_labels, first_N=first_N)
        # print('[%s] iter %d\tlower bound:\t%f' % (arrow.now(), 0, self.log_likelihood(T, tau)))
        # print('[%s] \t\tnum of alerts:%d,\tprecision:\t%f,\trecall:\t%f,\tF-1 score:\t%f.' % \
        #     (arrow.now(), n_alerts, init_precision, init_recall, F_1(init_precision, init_recall)))
        # training iters
        precisions = []
        recalls    = []
        logliks    = []
        lowerbs    = []
        for e in range(iters):
            # update P matrix:
            for i in t_indices:
                for j in t_indices[t_indices<i]:
                    self._update_P(i, j, T, tau)
            # update A matrix:
            for u in range(self.d):
                for v in range(self.d):
                    self._update_A(u, v, T, tau)
            # check sum of P
            self.check_P(t_indices)
            # get retrieval test results
            n_alerts, precision, recall = self.retrieval_test(t_indices, specific_labels=specific_labels, first_N=first_N)
            loglik = self.log_likelihood(T, tau)
            lowerb = self.jensens_lower_bound(T, tau)
            # logging
            print('[%s] iter %d\tlog likli: %f,\tlower bound: %f' % (arrow.now(), e+1, loglik, lowerb))
            print('[%s] \t\tnum of alerts: %d,\tprecision: %f,\trecall: %f,\tF-1 score: %f.' % \
                (arrow.now(), n_alerts, precision, recall, F_1(precision, recall)))
            precisions.append(precision)
            recalls.append(recall)
            logliks.append(loglik)
            lowerbs.append(lowerb)
        return precisions, recalls, logliks, lowerbs

    def save(self, path):
        np.savetxt(path + "P.txt", self.P, delimiter=',')
        np.savetxt(path + "A.txt", self.A, delimiter=',')
        np.savetxt(path + "Mu.txt", self.Mu, delimiter=',')

    def check_P(self, t_indices):
        test = [ self.P[i][j]
            for i in t_indices
            for j in t_indices[t_indices<i] ]
        print('[%s] sum of Pij is %f' % (arrow.now(), sum(test)))
        # for i in t_indices[0:10]:
        #     test = [ self.P[i][j] for j in t_indices[t_indices<i] ]
        #     print('[%s] sum of Pij is %f' % (arrow.now(), sum(test)))

if __name__ == '__main__':
    # generate synthetic data
    # - configuration
    T       = 1.
    n_dim   = 3
    n_point = 10
    # - data generation
    seq       = np.random.uniform(low=0., high=T, size=(n_point, 3))
    t_order   = seq[:, 0].argsort()
    seq       = seq[t_order, :]
    seq_s     = np.random.choice(n_dim, n_point)
    seq[:, 1] = seq_s
    print(seq)
    # model unittest
    hawkes = VecMarkedMultivarHawkes(n_dim=n_dim, T=T, seq=seq)
    print(hawkes.P)
    print(hawkes.A)
    print(hawkes.Mu)
