#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/9/11 上午8:40
@Author  : fanyuexiang
@Site    : 
@File    : Kernel_PCA_training.py
@Software: PyCharm
@version: 1.0
@describe: PCA与核PCA分离同心圆数据，比较两者的降维效果
'''
from pprint import pprint
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
# 构建同心圆数据
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
# scatter 散点图，y==0----》布尔型numpy数组的切分
# plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='^', alpha=0.5)
normal_pca = PCA(n_components=2)
X_npca = normal_pca.fit_transform(X)
# subplots绘制多个子图，图区域被分成 numRows 行和 numCols 列
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_npca[y==0, 0], X_npca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_npca[y==1, 0], X_npca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_npca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_npca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()
