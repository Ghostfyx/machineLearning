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
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
