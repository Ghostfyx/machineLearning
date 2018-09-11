#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/23 上午8:41
@Author  : fanyuexiang
@Site    : 
@File    : AdaLineGD.py
@Software: PyCharm
@version: 1.0
@describe: 自适应线性神经元（Adaline）
'''

import numpy as np

class AdaLineGD(object):
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1]+1)
        self.cost_ = []
        for i in range(self.n_iter):
            # 批量更新权重，批量梯度下降
            output = self.net_input(X=X)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        z = self.w_[1:] + self.w_[0]
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)