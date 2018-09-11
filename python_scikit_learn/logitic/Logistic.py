#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/28 上午8:33
@Author  : fanyuexiang
@Site    : 
@File    : Logistic.py
@Software: PyCharm
@version: 1.0
@describe: 使用scikit-learn构建logistic回归模型
'''
from pprint import pprint

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from python_scikit_learn.decisionregions import plot_decision_regions

# C表示正则化系数的倒数(目的是：增强正则化学习)，penalty：正则化方法选择（l1,l2）
lr = LogisticRegression(C=0.1, random_state=0, penalty='l2')
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
# 使用scikit_learn 对训练数据和测试数据进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)
# 对数据的特征值进行标准化处理，优化性能
sc = StandardScaler()
# 计算训练数据每个特征的样本均值和标准差
sc.fit(X_train)
# 注意，使用相同的参数缩放数据集，保证它们的值是彼此相等的
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_std = sc.transform(X)
lr.fit(X_train_std, y_train)
pprint(lr.coef_)
plot_decision_regions(X_std, y, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal weight')
plt.legend(loc = 'upper left')
plt.show()