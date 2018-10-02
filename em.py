#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Maximum Likelihood of Spatio-Temporal Marked Point Process by EM
'''

import numpy as np

class EM(object):

    def __init__(self, seq_t, seq_u, d, beta=1., window_duration=3600*24*30):
        # training data
        self.t      = seq_t # time of each of events
        self.u      = seq_u # component of each of events
        # basic configuration
        self.T0     = self.t[0]
        self.Tn     = self.t[-1]
        self.n      = len(self.t)     # number of events
        self.d      = d               # number of components
        self.w_dur  = window_duration # duration for the slide window
        # parameters for intensity kernel
        self.beta   = beta                                      # parameter for intensity kernel
        self.A      = np.random.uniform(0, 1, (self.d, self.d)) # influential matrix for intensity kernel
        self.A_mask = np.ones((self.d, self.d))                 # mask for influential matrix
        self.Mu     = np.random.uniform(0, 1, self.d)           # background rates for intensity kernel
        # TODO: change ones to normal distribution
        self.P      = np.ones((self.n, self.n)) * \
                      (1. / (self.w_dur / \
                      ((self.Tn - self.T0) / self.n)))          # transition probablity matrix
        # normalization
        self.t      = (self.t - self.T0) / (self.Tn - self.T0)
        self.w_dur  = self.w_dur / (self.Tn - self.T0)

    def _slide_window_indices(self, T, tau):
        '''select the indices of the sequence within the slide window'''
        # # normalization
        # T   = (T - self.T0) / self.Tn
        # tau = (tau - self.T0) / self.Tn
        return np.where((self.t < T) & (self.t >= tau))[0]

    def _loglik_subterm_1(self, i, t_indices):
        '''subterm 1 in log-likelihood function'''
        terms = [
            self.A[self.u[i]][self.u[j]] * self.beta * np.exp(-1 * self.beta * (self.t[i] - self.t[j]))
            for j in t_indices[t_indices<i] ]
        return np.array(terms)

    def _loglik_subterm_2(self, uj, t_indices, T):
        '''subterm 2 in log-likelihood function'''
        terms = [
            self.A[self.u[i]][uj] * (1 - np.exp(- self.beta * (T - self.t[i])))
            for i in t_indices]
        return np.array(terms)

    def log_likelihood(self, T, tau):
        '''log-likelihood function given t (time sequence) and u (component sequence)'''
        # normalization
        T   = (T - self.T0) / (self.Tn - self.T0)
        tau = (tau - self.T0) / (self.Tn - self.T0)
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
        # normalization
        T   = (T - self.T0) / (self.Tn - self.T0)
        tau = (tau - self.T0) / (self.Tn - self.T0)
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
        # normalization
        T   = (T - self.T0) / (self.Tn - self.T0)
        tau = (tau - self.T0) / (self.Tn - self.T0)
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        if i > j and j in t_indices and i in t_indices:
            numerator    = self.A[self.u[i]][self.u[j]] * self.beta * np.exp(-1 * self.beta * (self.t[i] - self.t[j]))
            denominator  = self.Mu[self.u[i]] + self._loglik_subterm_1(i, t_indices).sum()
            self.P[i][j] = numerator / denominator
            # return self.P[i][j]

    def _update_A(self, u, v, T, tau):
        '''update influential matrix A'''
        # normalization
        T   = (T - self.T0) / (self.Tn - self.T0)
        tau = (tau - self.T0) / (self.Tn - self.T0)
        # get time indices of the indicated window
        t_indices = self._slide_window_indices(T, tau)
        # check if A_{u,v} is available
        if self.A_mask[u][v]:
            numerator = []
            for i in np.where(self.u == u)[0].tolist():
                for j in np.where(self.u == v)[0].tolist():
                    if j < i and j in t_indices and i in t_indices:
                        numerator.append(self.P[i][j])
            denominator = [ 1 - np.exp(- self.beta * (T - self.t[j]))
                for j in t_indices if self.u[j] == v ]
            if len(numerator) == 0 or len(denominator) == 0:
                self.A_mask[u][v] = 0
                # return len(numerator), len(denominator)
            numerator    = sum(numerator)
            denominator  = sum(denominator)
            self.A[u][v] = numerator / denominator
            # return self.A[u][v]

    def fit(self, T, tau):
        # update P matrix:
        for i in range(self.n):
            for j in range(i):
                self._update_P(i, j)
            print(self.P[i, :].sum())
        # update A matrix:
        for u in range(self.d):
            for v in range(self.d):
                self._update_A(u, v)



def project2components(s, m):
    '''
    project spatio points into discrete multi-components according to the
    alignment of the components (d = m * m)
    '''
    max_x, min_x = s[:, 0].max(), s[:, 0].min()
    max_y, min_y = s[:, 1].max(), s[:, 1].min()
    print('max x %f, min x %f.' % (max_x, min_x))
    print('max y %f, min y %f.' % (max_y, min_y))

    x_bins = np.linspace(min_x-1e-3, max_x+1e-3, m+1)
    y_bins = np.linspace(min_y-1e-3, max_y+1e-3, m+1)
    x_inds = np.digitize(s[:, 0], x_bins) - 1
    y_inds = np.digitize(s[:, 1], y_bins) - 1
    u      = x_inds * m + y_inds
    return u

if __name__ == '__main__':
    m = 10
    # load data
    points = np.loadtxt('data/10k.points.txt', delimiter=',')
    points = points[:1000, ]
    # prepare training data
    t = points[:, 0]   # time sequence
    t_order = t.argsort()
    t = t[t_order]
    s = points[:, 1:3] # spatio sequence
    s = s[t_order]
    # project spatio sequence into components
    u = project2components(s, m)
    print(u)
    T = t.max() + 1.
    print(t.shape)
    print(u.shape)
    em = EM(seq_t=t, seq_u=u, d=m*m)
    # print(em.log_likelihood(T=1505779200, tau=1476230400))
    # print(em.jensens_lower_bound(T=1505779200, tau=1476230400))
    # print(em._update_P(100, 0))
    for i in range(m*m):
        for j in range(m*m):
            print(em._update_A(i, j, T=1505779200, tau=1476230400))
