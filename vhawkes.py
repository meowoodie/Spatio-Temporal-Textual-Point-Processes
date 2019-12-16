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

    def __init__(self, n_dim, T, seq, beta=1.):
        '''
        Params:
        - n_dim:   the number of dimensions of the sequence.
        - T:       the time horizon of the sequence.
        - seq:     the sequence of points with shape (n_point, 1+1+len_vec) where 
                   the last dimension indicates time, dimension and mark, respectively.
        - beta:    (optional) temporal decaying factor.
        '''
        # configuration
        self.seq     = seq
        self.T       = T
        self.n_dim   = n_dim
        self.beta    = beta
        self.n_point = self.seq.shape[0]     # the number of points in the sequence
        self.len_vec = self.seq.shape[1] - 2 # the length of the mark vector
        # model initialization
        self.P  = self._random_init_P(self.n_point)
        self.A  = self._random_init_A(self.n_dim)
        self.Mu = self._prop_init_Mu(self.n_dim, self.seq, self.T)
        self.M  = self._calculate_pairwise_mark_inner_prod(self.seq[:, 2:])

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

    @staticmethod
    def _calculate_pairwise_mark_inner_prod(seq_m):
        '''
        Calculate matrix M that stores the inner product between m_i and m_j for 
        all i and j. 
        '''
        n_point = seq_m.shape[0]
        M       = np.zeros((n_point, n_point))
        for i in range(n_point):
            for j in range(i+1):
                M[i, j] = M[j, i] = np.dot(seq_m[i, :], seq_m[j, :])
        return M

    def _h(self, i, j):
        '''
        h_{ij} denotes a special term: 
            h_{ij} = A_{s_i, s_j} * beta * exp(- beta * (t_i - t_j)) h_i' * h_j.
        This term will be further used in P matrix update. 

        See the definition of h_{ij} at Appendix D in the reference.
        '''
        h_ij = self.A[int(self.seq[i, 1]), int(self.seq[j, 1])] * \
               self.beta * np.exp(- self.beta * (self.seq[i, 0] - self.seq[j, 0])) * \
               self.M[i, j]
        return h_ij
    
    # Intermediate vectors which will be used in updating parameters

    def _update_sum_h(self):
        '''
        Update vector _H that stores the sum of h_{ij} from j=1 to j=i-1, for all i.
              _H_{ij}
            = sum_{j=1}^{i-1} h_{ij}
            = sum_{j=1}^{i-1} A_{s_i, s_j} * beta * exp(- beta * (t_i - t_j)) h_i' * h_j.
        This term will be further used in P matrix update.
        '''
        self._H = np.zeros(self.n_point)
        for i in range(self.n_point):
            _h_i = [ self._h(i, j) for j in range(i) ]
            self._H[i] = sum(_h_i)
    
    def _calculate_alpha_denominator(self):
        '''
        Calculate all possible the first term of the denominator of alpha_{uv}. 
            1 - exp(- beta * (T - t_j))
        This term will be further used in A matrix update
        '''
        self._DA = np.zeros(self.n_point)
        for j in range(self.n_point):
            self._DA[j] = 1 - np.exp(- self.beta * (self.T - self.seq[j, 0]))
                
    def _update_P(self, i, j):
        '''
        Update P_{ij} using an EM-like algorithm by maximizing the lower bound of the 
        log-likelihood function w.r.t. P
        '''
        if i == j:
            self.P[i, i] = self.Mu[int(self.seq[i, 1])] / (self.Mu[int(self.seq[i, 1])] + self._H[i])
        elif j < i: 
            self.P[i, j] = self._h(i, j) / (self.Mu[int(self.seq[i, 1])] + self._H[i])

    def _update_A(self, u, v):
        '''
        Update A_{uv} using an EM-like algorithm by maximizing the lower bound of the 
        log-likelihood function w.r.t. A
        of 
        '''
        seq_s = self.seq[:, 1]
        # indicator of s_j == v and s_i == u
        ind_sj_v = (seq_s.astype(np.int32) == v).astype(np.float32)
        ind_si_u = (seq_s.astype(np.int32) == u).astype(np.float32)
        denominator  = (ind_sj_v * self._DA * self.M.sum(axis=0)).sum()
        numerator    = (np.outer(ind_si_u, ind_sj_v) * self.P).sum()
        self.A[u, v] = numerator / denominator

    # def retrieval_test(self, t_indices, specific_labels=None, first_N=100):
    #     '''get precision and recall of retrieval test'''
    #     # only do the test on the specific labels if specific_labels is not None
    #     if specific_labels:
    #         specific_label_cond = lambda i: self.l[i] in specific_labels
    #     else:
    #         specific_label_cond = lambda i: True
    #     # get all the valid pairs
    #     pairs = [ [ self.P[i][j], i, j ] for i in t_indices for j in range(i) ]
    #     pairs = np.array(pairs)
    #     pairs = pairs[pairs[:, 0].argsort()]
    #     # print(len(pairs))
    #     # get retrieve, hits and relevant
    #     retrieve  = pairs[-first_N:, [1, 2]].astype(np.int32)
    #     hits      = [ (i, j) for i, j in retrieve if self.l[i] == self.l[j] and specific_label_cond(i) ]
    #     relevant  = [ (i, j) for i in t_indices for j in range(i) if self.l[i] == self.l[j] and specific_label_cond(i) ]
    #     # print(len(relevant))
    #     # get precision and recall
    #     precision = len(hits) / len(retrieve) if len(retrieve) != 0 else 0.
    #     recall    = len(hits) / len(relevant) if len(relevant) != 0 else 0.
    #     return len(retrieve), precision, recall

    def em_fit(self, iters=100):
        '''
        maximize the lower bound of the loglikelihood function by estimating
        matrix P and matrix A iteratively.
        '''
        # # F-1 score
        # F_1 = lambda p, r: 2 * p * r / (p + r) if (p + r) != 0. else 0.
        # # normalization
        # T   = (T - self.T0) / (self.Tn - self.T0)
        # tau = (tau - self.T0) / (self.Tn - self.T0)
        # # get time indices of the indicated window
        # t_indices = self._slide_window_indices(T, tau)
        # print('[%s] %d points will be fitted.' % (arrow.now(), len(t_indices)))
        # # init P
        # self._init_P(t_indices)
        # n_alerts, init_precision, init_recall = self.retrieval_test(t_indices, specific_labels=specific_labels, first_N=first_N)
        # print('[%s] iter %d\tlower bound:\t%f' % (arrow.now(), 0, self.log_likelihood(T, tau)))
        # print('[%s] \t\tnum of alerts:%d,\tprecision:\t%f,\trecall:\t%f,\tF-1 score:\t%f.' % \
        #     (arrow.now(), n_alerts, init_precision, init_recall, F_1(init_precision, init_recall)))
        # # training iters
        # precisions = []
        # recalls    = []
        # logliks    = []
        # lowerbs    = []

        # calculate the denominator of A_{uv}
        self._calculate_alpha_denominator()
        for e in range(iters):
            print('[%s] iter %d' % (arrow.now(), e))
            # update the denominator of P_{ij}
            self._update_sum_h()
            # update P matrix
            for i in range(self.n_point):
                for j in range(i+1):
                    self._update_P(i, j)
            # update A matrix
            for u in range(self.n_dim):
                for v in range(self.n_dim):
                    self._update_A(u, v)

            # # check sum of P
            # self.check_P(t_indices)
            # # get retrieval test results
            # n_alerts, precision, recall = self.retrieval_test(t_indices, specific_labels=specific_labels, first_N=first_N)
            # loglik = self.log_likelihood(T, tau)
            # lowerb = self.jensens_lower_bound(T, tau)
            # # logging
            # print('[%s] iter %d\tlog likli: %f,\tlower bound: %f' % (arrow.now(), e+1, loglik, lowerb))
            # print('[%s] \t\tnum of alerts: %d,\tprecision: %f,\trecall: %f,\tF-1 score: %f.' % \
            #     (arrow.now(), n_alerts, precision, recall, F_1(precision, recall)))
            # precisions.append(precision)
            # recalls.append(recall)
            # logliks.append(loglik)
            # lowerbs.append(lowerb)
        # return precisions, recalls, logliks, lowerbs

    def save(self, path):
        np.savetxt(path + "P.txt", self.P, delimiter=',')
        np.savetxt(path + "A.txt", self.A, delimiter=',')
        np.savetxt(path + "Mu.txt", self.Mu, delimiter=',')
    


if __name__ == '__main__':
    # generate synthetic data
    # - configuration
    T       = 1.
    n_dim   = 3
    n_point = 5
    # - data generation
    seq       = np.random.uniform(low=0., high=T, size=(n_point, 10))
    t_order   = seq[:, 0].argsort()
    seq       = seq[t_order, :]
    seq_s     = np.random.choice(n_dim, n_point)
    seq[:, 1] = seq_s
    print(seq)
    # model unittest
    hawkes = VecMarkedMultivarHawkes(n_dim=n_dim, T=T, seq=seq)
    print(hawkes.P.sum(axis=1))
    # print(hawkes.A)
    # print(hawkes.Mu)
    hawkes.em_fit()
