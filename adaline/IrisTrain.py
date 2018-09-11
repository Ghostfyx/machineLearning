#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/23 上午9:05
@Author  : fanyuexiang
@Site    : 
@File    : IrisTrain.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from adaline import AdaLineGD
from perception.decisionregions import plot_decision_regions

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #添加抬头
df = pd.read_csv('iris.csv', names=names) #读取csv数据
# 对前100条数据进行操作
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
ada = AdaLineGD.AdaLineGD(n_iter=15, eta= 0.01)
ada.fit(X_std,y)
plot_decision_regions(X_std,y,ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.show()
plt.plot(range(1,len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylable('Sum - squared -error')
plt.show()