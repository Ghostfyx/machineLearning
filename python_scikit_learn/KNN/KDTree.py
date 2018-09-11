#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/8/30 上午11:16
@Author  : fanyuexiang
@Site    : 
@File    : KDTree.py
@Software: PyCharm
@version: 1.0
@describe: 实现KD树的构建和搜索,kd树是一种为了提高KNN算法效率的特殊数据结构，本质上是二叉树
每个节点代表着对K维输入空间上的某一位进行划分；KD树适用于训练样本数目远大于特征维数的数据集。否则在进行搜素时，
效率与线性搜索效率基本一致
'''
from pprint import pprint


class Node(object):
    '''
    初始化一个节点
    '''
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

class KDNode(Node):
    '''
    初始化一个包含kd树数据和方法的节点
    '''
    def __init__(self, data=None, left=None, right=None, axis=None, next_axis=None, dimensions=None):
        """
        为KD树创造一个新的节点，如果该节点在树中被引用，必须给asix与next_asix赋值
        :param data: 当前节点数据
        :param left:
        :param right:
        :param axis: 当前需要被切分的维度
        :param next_axis: 下个需要被切分的维度
        :param dimensions: 数据集的维度
        """
        super(KDNode,self).__init__(data, left, right)
        self.axis = axis
        self.next_axis = next_axis
        self.dimensions = dimensions

def create_tree(point_list=None, dimensions=None, axis=0, sel_axis=None):
        if not dimensions and not point_list:
            raise ValueError('数据集维度和数据集不能为空！')
        elif point_list:
            dimensions = check_dimensionality(point_list, dimensions)
        sel_axis = sel_axis or (lambda prev_axis : (prev_axis+1) % dimensions)
        if not point_list:
            return KDNode(next_axis=sel_axis, axis=axis, dimensions=dimensions)
        point_list = list(point_list)
        point_list.sort(key=lambda point : point[axis])
        # //运算符：取整除————返回商的整数部分（向下取整）
        median = len(point_list) // 2

        loc = point_list[median]
        print(loc)
        print(sel_axis(axis))
        left = create_tree(point_list[:median], dimensions, sel_axis(axis))
        right = create_tree(point_list[median+1:], dimensions, sel_axis(axis))
        return KDNode(data=loc, left=left, right=right, next_axis=sel_axis, axis=axis, dimensions=dimensions)


def check_dimensionality(point_list, dimensions):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('所有样本的特征维度必须一致')
    return dimensions

if __name__ == '__main__':
    point_list = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    tree = create_tree(point_list, dimensions=len(point_list[0]))
    pprint(tree)
