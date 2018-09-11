#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/24 上午8:56
@Author  : fanyuexiang
@Site    : 
@File    : AdalineSGD.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import numpy as np

class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self,X,y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self,m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        out = self.net_input(xi)
        error = target - out
        self.w_[1:] = self.eta * xi.dot(error)
        self.w[0] = self.eta * error
        cost = error ** 2 /2.0
        return cost

    def net_input(self,X):
        return np.dot(X, self.w_[1:]+self.w_[0])