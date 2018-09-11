#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/30 上午9:11
@Author  : fanyuexiang
@Site    : 
@File    : knn.py
@Software: PyCharm
@version: 1.0
@describe: scikit-learn KNeighborsClassifier(KNN分类树)的实现，
此外scikit-learn还有KNeighborsRegressor（KNN回归树）
'''
from python_scikit_learn.decisionregions import plot_decision_regions
from python_scikit_learn.readData import get_iris_data
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plot

X_tain, X_test, y_train, y_test, X, y = get_iris_data()
knn = KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski')
knn.fit(X_tain, y_train)
plot_decision_regions(X, y, knn, test_idx=range(105, 150))
plot.xlabel('petal length')
plot.ylabel('petal width')
# legend函数用于说明图上的图例
plot.legend(loc = 'upper left')
# n_neighbors：KNN中的k值，根据metric参数给出的距离计算方式，找出最近的k个样本
knn2 = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
knn2.fit(X_tain, y_train)
plot_decision_regions(X, y, knn2, test_idx=range(105, 150))
plot.xlabel('petal length')
plot.ylabel('petal width')
# legend函数用于说明图上的图例
plot.legend(loc = 'upper left')
plot.show()
