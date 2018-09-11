#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/9/3 上午9:04
@Author  : fanyuexiang
@Site    : 
@File    : SBS.py
@Software: PyCharm
@version: 1.0
@describe: sklearn实现sbs（序列向后选择算法），进行特征抽取，基于贪心算法选择特征
'''
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split

class SBS(object):
    def __init__(self, estimator, k_features, scoring = accuracy_score, test_size=0.25,
                 random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        # 特征的总数量
        dim = X_train.shape[1]
        # tuple：python元组
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subset = []
            # combinations函数实现排列组合
            for p in combinations(self.indices_, r = dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subset.append(p)
            best = np.argmax(scores)
            # indices_记录特征的索引
            self.indices_ = subset[best]
            # 记录在特征数量一定的前提下，最好特征索引的集合
            self.subsets_.append(self.indices_)
            dim -= 1
            # 记录在特征数量一定的前提下，特征衡量函数的值
            self.scores_.append(scores[best])
        # 选择K个特征中最好的一个
        self.k_score = self.scores_[-1]
        return self


    def _calc_score(self,X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score