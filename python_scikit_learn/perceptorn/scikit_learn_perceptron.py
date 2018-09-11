#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/24 下午1:56
@Author  : fanyuexiang
@Site    : 
@File    : scikit_learn_perceptron.py
@Software: PyCharm
@version: 1.0
@describe: 使用scikit_learn 感知机API来训练莺尾花数据
'''
from pprint import pprint

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

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
# 使用scikit_learn的感知机API， random_state：每次训练后重排训练数据集，
# 采用随机梯度下降的方法寻找全局最优点
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print("训练完成的模型的准确率为：%.2f" %accuracy_score(y_true=y_test, y_pred=y_pred))



