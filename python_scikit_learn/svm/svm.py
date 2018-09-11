#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/29 上午8:52
@Author  : fanyuexiang
@Site    : 
@File    : svm.py
@Software: PyCharm
@version: 1.0
@describe:
'''
from sklearn.svm import SVC
import matplotlib.pyplot as plot
from python_scikit_learn.decisionregions import plot_decision_regions
from python_scikit_learn.readData import get_iris_data

# kernel: 核函数，rbf表示高斯核函数；gamma：核函数待优化参数的值； C：控制类间隔的大小
svm = SVC(kernel="rbf", random_state=0, gamma=0.1, C=1.0)
X_tain, X_test, y_train, y_test, X, y = get_iris_data()
svm.fit(X_tain, y_train)
plot_decision_regions(X, y, svm, test_idx=range(105, 150))
plot.xlabel('petal length')
plot.ylabel('petal width')
# legend函数用于说明图上的图例
plot.legend(loc = 'upper left')
plot.show()