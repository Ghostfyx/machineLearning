#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/9/5 上午8:56
@Author  : fanyuexiang
@Site    : 
@File    : PCA.py
@Software: PyCharm
@version: 1.0
@describe: 使用PCA（主成分析）对数据进行降维处理，将高纬数据映射到低纬数据，
注意PCA属于特征抽取方法，改变了原始数据
'''
from pprint import pprint
from python_scikit_learn.readData import get_wine_data
import numpy as np
import matplotlib.pyplot as plt
'''
PCA算法的步骤为：
    1. 对原属样本数据集进行标准化处理，目的是将不同度量标准的特征数据具有相同的重要性
    2. 构造样本协方差矩阵，协方差矩阵是d*d维的，存储不同特征值之间的协方差
    3. 选择前K个特征值以及其特征向量
'''

X_train_std, X_test_std, y_train, y_test, X_std, y = get_wine_data()
X_train_std_T = X_train_std.T
# 计算协方差矩阵
cov_mat = np.cov(X_train_std_T)
# 计算协方差矩阵的特征值和特征向量
eigen_values, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigen_values\n%s' %eigen_values)
total = np.sum(eigen_values)
# 计算特征值的贡献率，reverse即为按从大到小排列
var_exp = [(i/total) for i in sorted(eigen_values, reverse=True)]
# 返回给定axis上的累计和, 对于一维数组而言，返回一维数组，里面的数据是前面元素的累加之和
var_exp_cum = np.cumsum(var_exp)
# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center')
# plt.step(range(1,14), var_exp_cum, where='mid', )
# plt.xlabel('Explained variance ratio')
# plt.ylabel('Principal components')
# plt.show()
eigen_pairs = [(np.abs(eigen_values[i], eigen_vecs[:, i])) for i in range(len(eigen_values))]
# 选择前两个特征向量，按照列组合起来
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
X_train_std_pca = X_train_std.dot(w)