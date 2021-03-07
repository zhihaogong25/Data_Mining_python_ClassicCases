# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:33:29 2021

@author: Zhihao Gong
"""

import pandas as pd
import numpy as np
import math 

"""
Directly load the data file
"""

def loadDataSet(fileName, delim='\t'):
    fid = open(fileName)
    # strip() is to delete blank around datas, split() is used to separate 
    # datas
    stringArr = [line.strip().split(delim) for line in fid.readlines()]
    # list(map()) is used in python 3. 
    dataArr   = [list(map(float, line)) for line in stringArr]
    return np.mat(dataArr)

def loadStrSet(fileName, delim='\t'):
    fid = open(fileName)
    stringArr = [line.strip().split(delim) for line in fid.readlines()]
    return stringArr



"""
load csv datafile with DataFrame
"""
def loadDataFrame(fileName):
    dataArr = pd.read_csv(fileName)
    dataFrm = pd.DataFrame(dataArr)
    return dataFrm



def replaceNanWithMean(dataArr):
    numFeat = np.shape(dataArr)[1]
    for i in range(numFeat):
        indCol    = np.nonzero( ~np.isnan(np.array( dataArr[:,i])) )[0] 
        meanVal   = np.mean(  dataArr[ indCol, i ]  )
        indNanCol = np.nonzero(  np.isnan(np.array( dataArr[:,i])) )[0] 
        dataArr[ indNanCol, i ] = meanVal
    reconMat = dataArr
    return reconMat
