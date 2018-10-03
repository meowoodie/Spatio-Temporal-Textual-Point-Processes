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

    def __init__(self, d, seq_t, seq_u, seq_l, seq_m=None, beta=1.):
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
        # self.w_dur  = window_duration # duration for the slide window
        # parameters for intensity kernel
        self.beta   = beta                                      # parameter for intensity kernel
        self.A      = np.random.uniform(0, 1, (self.d, self.d)) # influential matrix for intensity kernel
        self.A_mask = np.ones((self.d, self.d))                 # mask for influential matrix
        self.Mu     = np.random.uniform(0, 1, self.d)           # background rates for intensity kernel
        self.P      = np.ones((self.n, self.n)) * -1            # transition probability matrix
                                                                # -1 means uninitiated value
        # normalization
        self.t      = (self.t - self.T0) / (self.Tn - self.T0)
        # self.w_dur  = self.w_dur / (self.Tn - self.T0)

    def _init_P(self, t_indices):
        '''init transition probability matrix'''
        # get time indices of the indicated window
        # t_indices  = self._slide_window_indices(T, tau)
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
            self.A[self.u[i]][self.u[j]] * self.beta * np.exp(-1 * self.beta * (self.t[i] - self.t[j])) * np.inner(self.m[i], self.m[j])
            for j in t_indices[t_indices<i] ]
        return np.array(terms)

    def _loglik_subterm_2(self, uj, t_indices, T):
        '''subterm 2 in log-likelihood function'''
        terms = [
            self.A[self.u[i]][uj] * (1 - np.exp(- self.beta * (T - self.t[i]))) * np.inner(self.m[i], self.m[t_indices[-1]])
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
        t_indices = self._slide_window_indices(T, tau)
        t_indices_before = lambda i: t_indices[t_indices<i]
        term_1 = [
            self.P[i][i] * np.log(self.Mu[self.u[i]]) + \
            (self.P[i][t_indices_before(i)] * np.log(self._loglik_subterm_1(i, t_indices))).sum() - \
            (self.P[i][t_indices_before(i)] * np.log(self.P[i][t_indices_before(i)])).sum()
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
            numerator    = self.A[self.u[i]][self.u[j]] * self.beta * np.exp(-1 * self.beta * (self.t[i] - self.t[j])) * np.inner(self.m[i], self.m[j])
            denominator  = self.Mu[self.u[i]] + self._loglik_subterm_1(i, t_indices).sum()
            self.P[i][j] = numerator / denominator
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
            denominator = [ 1 - np.exp(- self.beta * (T - self.t[j]))
                for j in t_indices if self.u[j] == v ]
            if len(numerator) == 0 or len(denominator) == 0:
                self.A_mask[u][v] = 0
                return
            numerator    = sum(numerator)
            denominator  = sum(denominator)
            self.A[u][v] = numerator / denominator
            # print('A(%d, %d) = %f' % (u, v, self.A[u][v]))

    def retrieval_test(self, t_indices, threshold=0.05):
        alerts = [ (i, j) for i in t_indices for j in range(i) if self.P[i][j] >= threshold ]
        hits   = [ (i, j) for i, j in alerts if self.l[i] == self.l[j] ]
        miss   = [ (i, j) for i in t_indices for j in range(i) if self.P[i][j] < threshold and self.l[i] == self.l[j] ]
        precision = len(hits) / len(alerts)
        recall    = len(miss) / (len(hits) + len(miss))
        return precision, recall

    def fit(self, T, tau, epoches=100):
        '''
        maximize the lower bound of the loglikelihood function by estimating
        matrix P and matrix A iteratively.
        '''
        # normalization
        T   = (T - self.T0) / (self.Tn - self.T0)
        tau = (tau - self.T0) / (self.Tn - self.T0)
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        print('[%s] %d points will be fitted.' % (arrow.now(), len(t_indices)))
        # init P
        self._init_P(t_indices)
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
            precision, recall = self.retrieval_test(t_indices)
            print('[%s] epoch %d\tlower bound:\t%f' % (arrow.now(), e, self.jensens_lower_bound(T, tau)))
            print('[%s] \t\tprecision:\t%f,\trecall:\t%f' % (arrow.now(), precision, recall))

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
    np.set_printoptions(suppress=True)

    # T0 = 1467763200
    # Tn = 1505779200
    win_len = 100 # 3600 * 24 * 90
    step    = 10
    cold_start_epoches = 100
    regular_epoches    = 5

    t, m, l, u, u_set = utils.load_police_training_data(n=10056)
    em = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(set(u)))

    tau_i = 0
    T_i   = 100
    print('[%s] --- cold start with extra training epoches = %d in (%d, %d) ---' % \
        (arrow.now(), cold_start_epoches, t[tau_i], t[T_i]))
    # init Mu (use the entire data )
    em.init_Mu(alpha=1.)
    # cold start with extra training epoches
    em.fit(T=t[T_i], tau=t[tau_i], epoches=cold_start_epoches)
    while T_i < len(t):
        # tau_i += step
        T_i   += step
        print('[%s] --- regular fit with training epoches = %d in (%d, %d) ---' % \
            (arrow.now(), regular_epoches, t[tau_i], t[T_i]))
        em.fit(T=t[T_i], tau=t[tau_i], epoches=regular_epoches)



    # utils.plot_intensities4beats(em.Mu, u_set)
