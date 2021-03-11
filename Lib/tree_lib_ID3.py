# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:01:42 2021

@author: Zhihao Gong

The purpose of this Lib file is to realize 
the ID3 decision tree

1. calculate the Shannon Entropy
2. split the dataSet with inputting a feature and a value
3. determine the best choose for a feature
4. create the tree structure
5. classify the input vector according to the tree structure

"""

import math
import operator


"""
calculate the Shannon entropy for an input dataSet
    in Calculating the Shannon entropy, the final column 
    of the dataSet is used
   1. formulation for dataSet: feature1 feature2 ... featureN classList
   2. the countSample is used to count the number for each class of "classList"
    
"""
def calcEntropy(dataSet):
    numSample = len(dataSet)
    countSample = {}
    for sample in dataSet:
        if sample[-1] not in countSample.keys():
            countSample[sample[-1]] = 0
        countSample[sample[-1]] += 1
    # calculate the probability for each class
    Entropy = 0.0
    for key in countSample.keys():
        Prob = countSample[key]/float(numSample)
        Entropy -= Prob * math.log(Prob, 2.0)
    return Entropy
     
"""
use feature axis and value to split the dataSet
function "extend" is used to extend each vector with getting rid of "axis" 
function "append" is used to collect the vector for "value"
"""
def splitDataSet(dataSet, axis, value):
    reddataSet = []
    for featRow in dataSet:
        if featRow[axis] == value:
            tempRow = featRow[:axis]
            tempRow.extend(featRow[(axis+1):])
            reddataSet.append(tempRow)
    return reddataSet        
            
"""
use information gain to determine the best feature
"""
def bestChosenFeat(dataSet):
    numFeat = len(dataSet[0]) - 1
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeatAxis = -1
    for axis_i in range(numFeat):
        # calculate the Superposition of entropy of each value for chosen "axis_i"
        SupPositEntropy_i = 0.0
        sampleCol = [sampleCol[axis_i] for sampleCol in dataSet]
        SampleSet = set(sampleCol) # a Set of values
        for value_j in SampleSet:
            redDataSet = splitDataSet(dataSet, axis_i, value_j)
            prob_j     = len(redDataSet)/float(len(dataSet))
            Entropy_ij = calcEntropy(redDataSet)
            SupPositEntropy_i += prob_j * Entropy_ij 
        # the information gain by split dataSet 
        infoGain_i = baseEntropy - SupPositEntropy_i
        if (infoGain_i > bestInfoGain) :
            bestInfoGain = infoGain_i
            bestFeatAxis = axis_i
    return bestFeatAxis  
        
    
def majorityCount(classList):
    classCount = {}
    for key in classList:
        if key not in classCount.keys():
            classCount[key] = 0
        classCount[key] += 1
    sortedClassCount = sorted(classCount.items(),\
                              key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
use iteration to build the tree structure 
"""
def createTree(dataSet, labels):
    # End condition 1: reach a list of pure class
    classList = [sample[-1] for sample in dataSet]
    if(classList.count(classList[0]) == len(classList)):
        return classList[0]
    # End condition 2: No feature is left to split the dataSet
    if len(dataSet[0])  == 1 :
        # to separate the classList with count the number of class 
        return majorityCount(classList)
    # the iteration part
    bestFeatAxis = bestChosenFeat(dataSet)
    bestLabel    = labels[bestFeatAxis]
    myTree       = {bestLabel:{}}
    featValue    = [ example[bestFeatAxis] for example in dataSet]
    FeatSet      = set(featValue)
    for value in FeatSet:
        subLabels = labels[:bestFeatAxis]
        subLabels.extend(labels[bestFeatAxis+1:])        
        subDataSet = splitDataSet(dataSet, bestFeatAxis, value)
        myTree[bestLabel][value] = createTree(subDataSet, subLabels)
    return myTree
   
"""
use iteration to classify the testVec
"""    
def classify(inputMyTree, testVec, testLabels):
    firstStr       = inputMyTree.keys()[0]
    subTree        = inputMyTree[firstStr]
    rootLabelIndex = testLabels.index(firstStr)
    for key in subTree.keys():
        if testVec[rootLabelIndex] == key:
            if type(subTree[key]).__name__ == 'dict':
                classLabel = classify(subTree[key], testVec, testLabels)
            else: classLabel = subTree[key]
    return classLabel
        
    
    
