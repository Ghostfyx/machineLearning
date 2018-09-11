#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/31 上午8:59
@Author  : fanyuexiang
@Site    : 
@File    : handle_data.py
@Software: PyCharm
@version: 1.0
@describe: 对缺省数据和标注类标数据进行处理，以莺尾花数据为例
'''
from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'label'] #添加抬头
df = pd.read_csv('../iris.csv', names=names)
# print(df['label'])
# 使用numpy对数据集的类别进行编码（使用枚举的方式）
class_mapping = {lable : idx for idx, lable in enumerate(np.unique(df['label']))}
# 根据字典映射方式，将特征值或类标转换成整数
df.label = df['label'].map(class_mapping)
# print(df)
df_scikit_learn = df = pd.read_csv('../iris.csv', names=names)
# 使用scikit_learn的LabelEncoder类进行字符串类标转换（处理连续有序型特征值）
class_le = LabelEncoder()
y = class_le.fit_transform(df_scikit_learn['label'].values)
# pprint(y)
# 将转换后的类别标签还原转至
# class_label = class_le.inverse_transform(y)
# pprint(class_label)
# 对标称特征进行处理，使用独热编码的方式！！！
df_clothes = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class3']
])
df_clothes.columns = ['color', 'size', 'price', 'label']
# 对size有序特征值进行map映射
size_mapping = {'M':1, 'L':2, 'XL':3}
df_clothes['size'] = df_clothes['size'].map(size_mapping)
print(df_clothes)
# 使用sklearn的独热编码实现类进行编码转换，
# 一定要注意：使用OneHotEncoder类时，所有特征值必须为实数（而不是字符串）
X = df_clothes[['color', 'size', 'price']].values
X[:, 0] = class_le.fit_transform(X[:, 0])
print(X)
ohe = OneHotEncoder(categorical_features=[0])
# fit_transform返回一个稀疏矩阵，为了可视化处理转换成numpy数组

df_clothes_array = ohe.fit_transform(X).toarray()
pprint(df_clothes_array)
