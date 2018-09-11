#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/9/6 上午8:51
@Author  : fanyuexiang
@Site    : 
@File    : PCA_handler.py
@Software: PyCharm
@version: 1.0
@describe: 使用sklearn PCA压缩数据，然后使用logistic进行分类训练
'''
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from perception.decisionregions import plot_decision_regions
from python_scikit_learn.readData import get_wine_data

pca = PCA(n_components=2)
lr = LogisticRegression(C=0.1, random_state=0, penalty='l2')
X_train_std, X_test_std, y_train, y_test, X_std, y = get_wine_data()
X_train_std_pca = pca.fit_transform(X_train_std)
X_test_std_pca = pca.fit_transform(X_test_std)
lr.fit(X_train_std_pca, y_train)
plot_decision_regions(X_train_std_pca, y_train, classifier=lr)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc = 'lower left')
plt.show()
