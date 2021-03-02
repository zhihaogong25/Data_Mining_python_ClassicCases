# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:31:17 2021

@author: Zhihao Gong
"""


"""
Loading the data file
"""
import numpy as np


import sys
Lib_path = r'E:\数据挖掘\PythonCode\Lib'
sys.path.append(Lib_path) 


import loadData_Lib as ldD
import pca_lib as pcA


data_path = r'E:\数据挖掘\PythonCode\SemiConduct_MF_data'
data_name = r'\secom.data'

# imput data from datafile
dataMat = ldD.loadDataSet(data_path + data_name, ' ')

# replace the nan values in data with mean value of nonnan values for each feature
reconMat = ldD.replaceNanWithMean(dataMat)

pcA.pca_pre(dataMat)
