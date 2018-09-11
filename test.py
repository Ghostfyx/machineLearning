#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/30 上午10:41
@Author  : fanyuexiang
@Site    : 
@File    : test.py
@Software: PyCharm
@version: 1.0
@describe:
'''
from itertools import combinations
from pprint import pprint

import numpy as np
import sklearn

from python_scikit_learn.readData import get_wine_data

print(pow(54, 1/3))
a = [2,5,9,4,8,7]
print(np.median(a))
print(5 // 2)
print(sklearn.__version__)
c = tuple(range(10))
d = len(c)
for item in combinations(c, r=d-1):
    print(item)

