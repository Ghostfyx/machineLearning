#!/usr/bin/env python
# encoding: utf-8
'''
@author: fanyuexiang
@Software: PyCharm
@version: 1.0
@file: Kernel_PCA_training.py
@time: 2018/9/13 21:58
@desc: 使用核PCA算法对同心圆训练数据进行处理，并对比PCA算法
'''
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from python_scikit_learn.pca.Kernel_PCA import rbf_kernel_PCA
import numpy as np

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
X_RPCA = rbf_kernel_PCA(X=X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsizae=(7,3))
ax[0].scatter(X_RPCA[y==0, 0], X_RPCA[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_RPCA[y==1, 0], X_RPCA[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_RPCA[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_RPCA[y==1, 0],  np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()

