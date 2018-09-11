#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/21 下午8:38
@Author  : fanyuexiang
@Site    : 
@File    : decisionregions.py
@Software: PyCharm
@version: 1.0
@describe: 分类matlab可视化
'''
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx = None):
    makers = ['s','x','o','r','v']
    colors = ['red','blue','lightgreen','gray','cyan']
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    # 构建numpy数组
    a = np.arange(x1_min, x1_max, resolution)
    b = np.arange(x2_min,x2_max, resolution)
    # 从坐标向量返回坐标矩阵。
    xx1,xx2 = np.meshgrid(a, b)
    # ravel 将多维数组降为一维
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    # reshape重新构造数组
    Z = Z.reshape(xx1.shape)
    # 画等高线图
    plt.contourf(xx1,xx2,Z,alpha = 0.4, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1,0], y = X[y == c1, 1], alpha=0.8, c = cmap(idx), marker=makers[idx])

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0,
                    linewidths=1, marker='o', s=55)

