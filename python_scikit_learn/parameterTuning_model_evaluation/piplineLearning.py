#!/usr/bin/env python
# encoding: utf-8
'''
@author: fanyuexiang
@contact: xxxxxxxxxx@163.com
@Software: PyCharm
@version: 1.0
@file: piplineLearning.py
@time: 2018/9/13 22:27
@desc: 对sklearn的工作流实现
'''
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from python_scikit_learn.readData import get_breast_data

X_train, X_test, y_train, y_test = get_breast_data()
# random_state 随机数种子，默认为无，仅在正则化优化算法为sag,liblinear时有用
pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('PCA', PCA(n_components=2)),
    ('logistic', LogisticRegression())
])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' %pipe_lr.score(X_test, y_test))
