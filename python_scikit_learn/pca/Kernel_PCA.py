#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/9/10 上午8:59
@Author  : fanyuexiang
@Site    : 
@File    : Kernel_PCA.py
@Software: PyCharm
@version: 1.0
@describe: 使用SBF（高斯核）实现核PCA，我们使用scipy和numpy来实现，
SciPy在Numpy的基础上提供了更多的数组计算函数
'''
# scipy linalg:scipy线性代数库
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy import exp
import numpy as np

def rbf_kernel_PCA(X, gamma, n_components):
    """
    :param X: Numpy ndarray, shape = {n_samples, n_features}
    :param gamma: Turing parameter of the RBF kernel
    :param n_components: Number of principal to return
    """
    # pdist是scipy计算距离库中，计算每个向量对之间的距离
    sq_dists = pdist(X, 'sqeuclidean')
    # squareform() 将向量对的距离转换为矩阵，现在矩阵为N*N的矩阵（其中N为n_samples）
    mat_sq_dists = squareform(sq_dists)
    # compute the symmetric kernel matrix（矩阵）
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    # oen_n 相当于公式中的l_n，l_n是一个N*N的矩阵，其所有的值均为1/n
    one_n = np.ones((N, N)) / N
    # 聚集矩阵K
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # 将聚集后的核矩阵按照特征值进行降序排列
    eigvalues, eigvecs = eigh(K)
    # column_stack：numpy数组的拼接
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))
    return X_pc