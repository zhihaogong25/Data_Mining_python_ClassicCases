# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:55:51 2021

@author: P7XX-TM1G
"""
import numpy as np
import operator
import matplotlib.pyplot as plt

import sys
Lib_path = r'E:\数据挖掘\PythonCode\Lib'
sys.path.append(Lib_path) 


import tree_lib_C4_5 as tree
#import tree_lib_std as tree
import loadData_Lib as ldD


dataSet = [ ['yes', 'single',   125, 'no' ],
            ['no',  'married',  100, 'no' ],
            ['no',  'single',   70,  'no' ],
            ['yes', 'married',  120, 'no' ],
            ['no',  'divorced', 95,  'yes'],
            ['no',  'married',  60,  'no' ],
            ['yes', 'divorced', 220, 'no' ],
            ['no',  'single',   85,  'yes'],
            ['no',  'married',  75,  'no' ],
            ['no',  'single',   90,  'yes']]

labels = ['homeowner', 'marriage status', 'Yearly Salaries(K)', 'loan delinquency']

#A = tree.bestChosenFeat(dataSet, labels
                        
mytree = tree.createTree(dataSet, labels)                        

#tree.createPlot(mytree)

testLabels = labels.copy()
testVec = ['no', 'single', 91 ]

classLabel = tree.classify(mytree, testVec, testLabels)