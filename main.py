#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : main.py
@Time : 2018/12/07 15:33:53
@Author : JiahuiRen 
'''
# here put the import lib
import numpy as np
import pandas as pd

from MF_recommendation import Matrix_Factorization
from measure import measure_method

if __name__ == "__main__":
    train = pd.read_csv('dataset/ml-100k/user-rating.csv', index_col=0)
    test = pd.read_csv('dataset/ml-100k/user-rating_test.csv', index_col=0)
    MF_estimate = Matrix_Factorization.Matrix_Factorization(K=3, epoch=10,beta=0.06)
    MF_estimate.fit(train)
    R_hat = MF_estimate.start()
    non_index = test.values.nonzero()
    pred_MF = R_hat[non_index[0], non_index[1]]
    actual = test.values[non_index[0], non_index[1]]
    print('MSE of MF is %s' % measure_method.comp_mse(pred_MF, actual))
    print('RMSE of MF is %s' % measure_method.comp_rmse(pred_MF, actual))    