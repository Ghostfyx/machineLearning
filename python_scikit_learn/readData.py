#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/29 上午8:53
@Author  : fanyuexiang
@Site    : 
@File    : readData.py
@Software: PyCharm
@version: 1.0
@describe: 获取莺尾花数据公共类
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_iris_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    # 使用scikit_learn 对训练数据和测试数据进行划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)
    # 对数据的特征值进行标准化处理，优化性能
    sc = StandardScaler()
    # 计算训练数据每个特征的样本均值和标准差
    sc.fit(X_train)
    # 注意，使用相同的参数缩放数据集，保证它们的值是彼此相等的
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_std = sc.transform(X)
    return X_train_std, X_test_std, y_train, y_test, X_std, y


def get_wine_data():
    wine = datasets.load_wine()
    # wine = pd.read_csv(filepath_or_buffer="/Users/yuexiangfan/PycharmProjects/machineLearning/python_scikit_learn/wine.csv", header=None)
    wine.columns = ['Class Label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
             'Total phenols', 'Flavanoids', 'Noflavanoid phenols', 'Proanthocyanins', 'Color intensity',
             'Hue','OD280/OD315 of diluted wines', 'Proline']
    X = wine.iloc[:, 1:].values
    y = wine.iloc[:, 0].values
    # 使用scikit_learn 对训练数据和测试数据进行划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)
    # 对数据的特征值进行标准化处理，优化性能
    sc = StandardScaler()
    # 计算训练数据每个特征的样本均值和标准差
    sc.fit(X_train)
    # 注意，使用相同的参数缩放数据集，保证它们的值是彼此相等的
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_std = sc.transform(X)
    return X_train_std, X_test_std, y_train, y_test, X_std, y