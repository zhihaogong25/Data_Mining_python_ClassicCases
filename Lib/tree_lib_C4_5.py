# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:29:12 2021

@author: Zhihao Gong


The purpose of this code is to realize C4.5

"""

import numpy as np
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
split continous feature into two classes:
    class 1 including the dataSet smaller than value
    class 2 including the dataSet greater than value
"""

def splitContinuousDataSet(dataSet, axis, value):
    splitdataSetC1 = []
    splitdataSetC2 = []
    for featRow in dataSet:
        featGidRow = featRow[:axis]
        featGidRow.extend(featRow[axis+1:])
        if featRow[axis] < value:
            splitdataSetC1.append(featGidRow)
        else:
            splitdataSetC2.append(featGidRow)
    return splitdataSetC1, splitdataSetC2
            

"""
use information gain ratio to determine the best feature:
    1. including both continous and discrete feature splitting 
    2. first select the features whose information gain is larger than average,
       then select the features which has the most information gain ratio
"""
def bestChosenFeat(dataSet, labels):
    
    numFeat = len(dataSet[0]) - 1
    numTotSample = len(dataSet)
    baseEntropy = calcEntropy(dataSet)
    inlabels = labels.copy()
   # traverse the feature axis 
    InfoFeatMat  = [] # contains information gain and information volumn
    for axis_i in range(numFeat):
        featCol = [sampleRow[axis_i] for sampleRow in dataSet]
        # Consider the contious feature
        if type(featCol[0]).__name__ == 'float' or type(featCol[0]).__name__ == 'int' :
            # sort Feature column to generate the splitList
            sortedFeatCol = sorted(featCol)
            splitVec = []
            for j in range(len(sortedFeatCol)-1):
                splitvalue = ( sortedFeatCol[j] + sortedFeatCol[j+1] ) * 0.5
                splitVec.append(splitvalue)
            # traverse splitList to separate the dataSet to calculate 
            # the conditional entropy, Entropy(D | a), 
            # and the conditional information gain array
            infoMat = []
            for valueSplit in splitVec:
                dataClass1,dataClass2 = splitContinuousDataSet(dataSet, axis_i,valueSplit)
                # calculate information gain and information volumn
                Prob1            = len(dataClass1)/float(numTotSample)
                Prob2            = len(dataClass2)/float(numTotSample)
                conditEntropy    = Prob1 * calcEntropy(dataClass1) + \
                                   Prob2 * calcEntropy(dataClass2) 
                infoGain         = baseEntropy - conditEntropy
                infoVol          = - Prob1 * math.log(Prob1, 2) + \
                                   (- Prob2 * math.log(Prob2, 2) )
                infoMat.append([infoGain, infoVol])                  
            infoGainCol  = [row[0] for row in infoMat ]
            infoGainMean = np.mean(infoGainCol)
            # use the mean value 'infoGainMean' to filter the infoGain
            bestInfoGain      = 0.0 
            bestInfoGainRatio = 0.0
            bestSplitAxis     = -1
            for col in np.nonzero(infoGainCol > infoGainMean)[0] :
                infoGainRatio = infoMat[col][0]/infoMat[col][1] 
                if infoGainRatio > bestInfoGainRatio:
                    bestInfoGainRatio = infoGainRatio
                    bestSplitAxis     = col
                    bestInfoGain      = infoMat[col][0]
            inlabels[axis_i] = [inlabels[axis_i], splitVec[bestSplitAxis]]
            InfoFeatMat.append([bestInfoGain, infoMat[col][1]])  
        else: # consider the discrete feature
            SampleSet = set(featCol) # a Set of values
            infoGain_d  = baseEntropy
            infoVol_d   = 0.0
            for value_j in SampleSet:
                redDataSet = splitDataSet(dataSet, axis_i, value_j)
                prob_j     = len(redDataSet)/float(len(dataSet))
                Entropy_ij = calcEntropy(redDataSet)
                infoGain_d   -= prob_j * Entropy_ij 
                infoVol_d    -= prob_j * math.log(prob_j, 2)
            InfoFeatMat.append([infoGain_d, infoVol_d])
    # determine the best feature according to information gain ratio
    # After the traverse over axis_i
    bestInfoGainRatio_all = 0.0
    bestFeature           = -1
    infoGainCol  = [row[0] for row in InfoFeatMat ]
    infoGainMean_all = np.mean(infoGainCol)
    for col in np.nonzero(infoGainCol > infoGainMean_all)[0] :
        infoGainRatio = InfoFeatMat[col][0]/InfoFeatMat[col][1]
        if infoGainRatio > bestInfoGainRatio_all:
            bestInfoGainRatio_all = infoGainRatio
            bestFeature           = col
            # which means that we only have unit class of samples for this feature
    return inlabels[bestFeature], bestFeature
             

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
    bestLabel, bestFeature = bestChosenFeat(dataSet, labels)
    if bestFeature == -1 :
        return majorityCount(classList)   
    subLabels = labels[:bestFeature]
    subLabels.extend(labels[bestFeature+1:])       
    if type(bestLabel).__name__ == 'list':
        myTree      = {bestLabel[0]:{}}
        splitValue  = bestLabel[1]
        dataC1,dataC2 = splitContinuousDataSet(dataSet, bestFeature,splitValue)
        myTree[bestLabel[0]]['<'+str(splitValue)]  = createTree(dataC1, subLabels)
        myTree[bestLabel[0]]['=>'+str(splitValue)] = createTree(dataC2, subLabels)
    else:
        myTree    = {bestLabel:{}}
        featValue = [ example[bestFeature] for example in dataSet]
        featSet   = set(featValue)
        for value in featSet:
            subDataSet = splitDataSet(dataSet, bestFeature, value)
            myTree[bestLabel][value] = createTree(subDataSet, subLabels)
    return myTree



"""
use iteration to classify the testVec
"""    
def classify(inputMyTree, testVec, testLabels):
    firstStr       = list(inputMyTree.keys())[0]
    subTree        = inputMyTree[firstStr]
    rootLabelIndex = testLabels.index(firstStr)
    testValue      = testVec[rootLabelIndex]
    print(rootLabelIndex)
    for key in subTree.keys():
        # conisder the continous feature
        if key[0] in {'>', '=', '<'}:
            if key[1] in {'>', '=', '<'}:
                splitValue = float( key[2:] )
                if (set(key[:2]) == {'>','='} and testValue >= splitValue) or (set(key[:2]) == {'<','='} and testValue <= splitValue) :
                    if type(subTree[key]).__name__ == 'dict':
                        classLabel = classify(subTree[key], testVec, testLabels)    
                    else: classLabel = subTree[key]
            else:
                splitValue = float( key[1:] )
                if (key[0] == '>' and testValue > splitValue) or (key[0] == '<' and testValue < splitValue) :
                    if type(subTree[key]).__name__ == 'dict':
                        classLabel = classify(subTree[key], testVec, testLabels)    
                    else: classLabel = subTree[key]
        else:
            if testValue == key:
                if type(subTree[key]).__name__ == 'dict':
                    classLabel = classify(subTree[key], testVec, testLabels)
                else: classLabel = subTree[key]    
    return classLabel        

"""
use annotation of matplotlib.pyplot to plot nodes 
"""
import matplotlib.pyplot as plt
#add Chinese font
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

decisionNode = dict(boxstyle="sawtooth", fc = "0.8")
leafNode     = dict(boxstyle="round4",   fc = "0.8")
arrow_args   = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
                            xycoords = 'axes fraction', \
                            xytext= centerPt, textcoords = 'axes fraction',\
                            va = 'center', ha = 'center', bbox = nodeType, \
                            arrowprops=arrow_args)

"""
Test the plotNode subroutine
"""
#def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False)
#    plotNode(U'决策节点', (0.5, 1.0), (0.5, 1.0), decisionNode )        

"""
use iteration to count the number of leafs
"""
def getNumLeafs(myTree):
    numLeaf  = 0
    firstStr = list(myTree.keys())[0]
    subTree  = myTree[firstStr]
    for keys in subTree.keys():
        if type(subTree[keys]).__name__ == 'dict':
            numLeaf += getNumLeafs(subTree[keys])
        else:
            numLeaf += 1
    return numLeaf
    
"""
use iteration to return the depth of decistion tree
"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    subTree  = myTree[firstStr]
    for keys in subTree.keys():
        if type(subTree[keys]).__name__ == 'dict':
            thisDepth = getTreeDepth(subTree[keys]) + 1
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


"""
mark the context of node in the tree in the middle of the arrow
"""     
def plotMidTxt(cntrPt, parentPt, valueStr):
    xMid = ( cntrPt[0] + parentPt[0] ) * 0.5
    yMid = ( cntrPt[1] + parentPt[1] ) * 0.5
    # Toward Object "createPlot" to add text of "valueStr"
    # on the point (xMid, yMid)
    createPlot.ax1.text(xMid, yMid, valueStr)
    
"""
iteratively plot the tree structure
-global variables:
 (which are initallized at main subroutine "createPlot") 
    1. xOff --- to record the x axis of leaf nodes;
                initially is given at - 0.5/total Width
                and keep this point until the leaf nodes are 
                reached
                
    2. yOff --- to record the y axis of tree depth;
                initially is given at 1.0 and decrease with
                1/total Depth within a signle iteration
    3. tWidth --- the total Width for the tree 
                  and keep constant throughout
    4. tDepth --- the total Depth for the tree 
                  and keep constant throughout  
-initial input : 
    1. myTree 
    2. position of RootNode, 
    3. a blank string : '' for valueStr                 
                                
"""    
def plotTree(myTree, parentPt, valueStr):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    # evaluate the position for the centerPoint
    cntrPt_x = plotTree.xOff + 0.5/plotTree.tWidth * (1.0 + float(numLeafs))
    cntrPt_y = plotTree.yOff
    cntrPt   = (cntrPt_x, cntrPt_y)
    plotMidTxt(cntrPt, parentPt, valueStr)
    plotNode(firstStr,cntrPt, parentPt, decisionNode)
    # begin the iteration
    subTree = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.tDepth
    # begin to traverse the subtree
    for key in subTree.keys():
        # if the condition of iteration is acitived
        if type(subTree[key]).__name__ == 'dict':
            newParentPt = cntrPt
            plotTree(subTree[key], newParentPt, str(key) )
        else:
            # the stop condtion is actived: the leaf nodes are reached
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.tWidth
            newParentPt   = cntrPt
            newcntrPt     = (plotTree.xOff, plotTree.yOff)
            plotMidTxt(newcntrPt, newParentPt, str(key))
            plotNode(subTree[key], newcntrPt, newParentPt, leafNode)
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.tDepth
            
            
            
"""
To initialize the global variables for subroutine "plotTree"
and call plotTree
"""    
    
def createPlot(myTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # drop the xticks and yticks
    axprops = dict(xticks=[], yticks=[]) 
    # create a whiteboard
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # Toward Object "plotTree" to add feature 
    plotTree.tWidth = float( getNumLeafs(myTree) )
    plotTree.tDepth = float( getTreeDepth(myTree) )
    plotTree.xOff   = - 0.5/plotTree.tWidth
    plotTree.yOff   = 1.0
    RootNode        = (0.5, 1.0)
    plotTree(myTree, RootNode, '')
    print(plotTree.yOff)
    plt.show()
    

