#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/20 下午3:15
@Author  : fanyuexiang
@Site    : 
@File    : IrisTrain.py
@Software: PyCharm
@version: 1.0
@describe:使用莺尾花数据对感知器进行训练
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perception.Perception import Perception
from perception.decisionregions import plot_decision_regions

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #添加抬头
df = pd.read_csv('iris.csv', names=names) #读取csv数据
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values
# 绘制散点图
# plt.scatter(X[:50,0],X[:50,1],c='red', marker='o')
# plt.scatter(X[50:100,0],X[50:100,1],c='blue', marker='x')
# plt.xlabel('petal length')
# plt.ylabel('sepal.length')
# plt.legend(loc='upper left')
# # plt.show()

ppn = Perception(eta= 0.1,n_iter=10)
ppn.fit(X,y)
# plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker='o',c='black')
# plt.xlabel('Epochs')
# plt.ylabel('Number of misclassifications')
# plt.show()

plot_decision_regions(X,y,ppn)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.show()