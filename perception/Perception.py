#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/20 下午2:14
@Author  : fanyuexiang
@Site    : 
@File    : Perception.py
@Software: PyCharm
@version: 1.0
@describe: python机器学习 感知器Demo，感知机仅仅对线性可分的数据集收敛，感知机存在
原始形式和对偶形式两种
'''
import numpy as np
import pandas as pd

class Perception(object):
    def __init__(self,eta = 0.1, n_iter = 10):
        """
        感知器初始化
        :param eta:学习速率
        :param n_iter: 迭代次数
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self,X,y):
        """
        训练函数
        :param X:训练样本
        :param Y:训练样本目标值
        """
        # 数据集的特征维度+1，w_为特征的权重向量，初始化权重向量
        # 疑问：为什么要多增加一列权重的阈值？
        # 回答：是损失函数使用梯度下降方法优化，用于存放参数b的值，即为将偏置向量b，并入权重向量w
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        # 开始进行迭代学习
        for i in range(self.n_iter):
            errors = 0
            # 遍历训练集，不断更新权重
            for xi,target in zip(X,y):
                # 更新权重的计算，predict函数计算预测的类别
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 表示权重的阈值
                self.w_[0] += update
                # 将判断错误的分类样本数量放入数组中
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        """
        符号函数，将特征向量与权重向量的乘积是否大于0，作为类别划分的边界
        :param X:
        :return:
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
