#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Marked Point Process Learning via EM algorithm
'''

import arrow
import numpy as np
import utils

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

    # Initialization functions

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

    # Update rules for P matrix and A matrix
                
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

    # log-likelihood value

    def log_likelihood(self):
        '''
        Log-likelihood of the input sequence given current parameters.
        '''
        term1 = np.array([ 
            np.log(self.Mu[int(self.seq[i, 1])] + self._H[i])
            for i in range(self.n_point) ]).sum()
        term2 = - self.Mu.sum() * self.T * len(np.unique(self.seq[:, 2:], axis=0))
        term3 = - np.array([
            np.array([ self.A[k, int(self.seq[j, 1])] for j in range(self.n_point) ]) * \
            self._DA * self.M.sum(axis=0)
            for k in range(self.n_dim) ]).sum()
        loglik = term1 + term2 + term3
        return loglik

    # model fitting using EM like algorithm

    def em_fit(self, iters=2, verbose=True):
        '''
        maximize the lower bound of the loglikelihood function by estimating
        matrix P and matrix A iteratively.
        '''
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
            if verbose:
                print('[%s] log-likelihood %f' % (arrow.now(), self.log_likelihood()))

    def save(self, path):
        np.savetxt(path + "P.txt", self.P, delimiter=',')
        np.savetxt(path + "A.txt", self.A, delimiter=',')
        np.savetxt(path + "Mu.txt", self.Mu, delimiter=',')
    


if __name__ == '__main__':
    # UNITTEST 1: RANDOM DATA
    # - configuration
    T       = 1.
    n_dim   = 3
    n_point = 100
    # - data generation
    seq       = np.random.uniform(low=0., high=T, size=(n_point, 10))
    t_order   = seq[:, 0].argsort()
    seq       = seq[t_order, :]
    seq_s     = np.random.choice(n_dim, n_point)
    seq[:, 1] = seq_s
    # - test model
    hawkes    = VecMarkedMultivarHawkes(n_dim=n_dim, T=T, seq=seq)
    hawkes.em_fit()

    # # UNITTEST 2: SMALL REAL DATA
    # # - fetch real data
    # t, s, m, l, u, u_set, specific_labels = utils.load_police_training_data(n=500, category='burglary')
    # # - data preparation and configuration
    # t   = np.expand_dims((t - min(t) + 1000.) / (max(t) - min(t) + 2000.), -1) # time normalization
    # u   = np.expand_dims(u, -1)
    # seq = np.concatenate([t, u, m], axis=1)
    # print(seq) 
    # print(seq.shape)
    # n_dim = len(np.unique(u))
    # T     = 1.
    # # - test model
    # hawkes = VecMarkedMultivarHawkes(n_dim=n_dim, T=T, seq=seq)
    # hawkes.em_fit()

