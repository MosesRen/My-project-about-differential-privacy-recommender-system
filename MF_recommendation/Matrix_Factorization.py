#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : Matrix_Factorization.py
@Time : 2018/12/07 14:57:21
@Author : JiahuiRen 
'''
# here put the import lib


import numpy as np
import pandas as pd
import random


class Matrix_Factorization(object):

    def __init__(self, K=10, alpha=0.01, beta=0.02, epoch=1000, regularization=True, random_state=100):

        self.R = None
        self.K = K
        self.P = None
        self.Q = None
        self.r_index = None
        self.r = None
        self.length = None
        self.alpha = alpha
        self.beta = beta
        self.epoch = epoch
        self.regularization = regularization
        self.random_state = random_state
        self.max_rating = None
        self.min_rating = None


    def fit(self, R):

        np.random.seed(self.random_state)

        self.R = R.values
        self.max_rating = self.R.max()
        self.min_rating = self.R.min()
        print(self.max_rating)
        print(self.min_rating)
        M, N = self.R.shape
        self.P = np.random.rand(M, self.K)
        self.Q = np.random.rand(N, self.K)

        self.r_index = self.R.nonzero()
        self.r = self.R[self.r_index[0], self.r_index[1]]
        self.length = len(self.r)



    def _comp_descent(self, index):

        r_i = self.r_index[0][index]
        r_j = self.r_index[1][index]

        p_i = self.P[r_i]
        q_j = self.Q[r_j]

        r_ij_hat = p_i.dot(q_j)
        e_ij = self.R[r_i, r_j] - r_ij_hat


        if self.regularization == True:
            descent_p_i = -2 * e_ij * q_j + self.beta * p_i
            descent_q_j = -2 * e_ij * p_i + self.beta * q_j
        else:
            descent_p_i = -2 * e_ij * q_j
            descent_q_j = -2 * e_ij * p_i

        return r_i, r_j, p_i, q_j, descent_p_i, descent_q_j


    def _update(self, p_i, q_j, descent_p_i, descent_q_j):

        p_i_new = p_i - self.alpha * descent_p_i
        q_j_new = q_j - self.alpha * descent_q_j

        return p_i_new, q_j_new



    def _update_DP(self, index):
        r_i = self.r_index[0][index]
        r_j = self.r_index[1][index]

        p_i = self.P[r_i]
        q_j = self.Q[r_j]
        r_ij_hat = p_i.dot(q_j)

        # print(p_i)
        # print(q_j)

        e_ij = self.R[r_i, r_j] - r_ij_hat
        if e_ij > 2:
            e_ij = 2
        elif e_ij < -2:
            e_ij = -2

        p_i_new = p_i + self.alpha*(e_ij*q_j - self.beta*p_i)
        q_j_new = q_j + self.alpha*(e_ij*p_i - self.beta*q_j)

        return r_i, r_j, p_i_new,q_j_new


    def _estimate_r_hat(self):

        r_hat = self.P.dot(self.Q.T)[self.r_index[0], self.r_index[1]]

        return r_hat


    def start(self):

        epoch_num = 1
        while epoch_num <= self.epoch:
            for index in range(0, self.length):
                
                r_i, r_j, p_i, q_j, descent_p_i, descent_q_j = self._comp_descent(index)
                p_i_new, q_j_new = self._update(p_i, q_j, descent_p_i, descent_q_j)
                #r_i, r_j, p_i_new, q_j_new = self._update_DP(index)
                self.P[r_i] = p_i_new
                self.Q[r_j] = q_j_new

            r_hat = self._estimate_r_hat()
            e = r_hat - self.r
            error = e.dot(e)
            print('The error is %s=================Epoch:%s' %(error, epoch_num))
            epoch_num += 1

        R_hat = self.P.dot(self.Q.T)
        return R_hat