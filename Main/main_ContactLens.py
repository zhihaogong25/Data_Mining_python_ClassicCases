# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:38:30 2021

@author: Zhihao Gong

"""


import numpy as np
import operator
import matplotlib.pyplot as plt

import sys
Lib_path = r'E:\数据挖掘\PythonCode\Lib'
sys.path.append(Lib_path) 


import tree_lib_std as tree
import loadData_Lib as ldD

data_path = r'E:\数据挖掘\PythonCode\lense_data'
data_name = r'\lenses.txt'

lenseData  = ldD.loadStrSet(data_path + data_name, '\t')
lenseLabel = ['age', 'prescript', 'astigmatic', 'tearRate']
lenseTree  = tree.createTree(lenseData, lenseLabel)
tree.createPlot(lenseTree)





"""
# test code for decisiton tree ID3

dataSet = [[1, 1, 'yes'],
           [1, 1, 'yes' ],
           [1, 0, 'no' ],
           [0, 1, 'no' ],
           [0, 1, 'no' ],
           ]
label = ['no surfacing', 'flippers']
testLabel = ['no surfacing', 'flippers']

myTree = tree.createTree(dataSet, label)



numLeafs = tree.getNumLeafs(myTree)
depthMyTree = tree.getTreeDepth(myTree)

tree.createPlot(myTree)

Res = tree.classify(myTree, [1,0], testLabel)



"""

