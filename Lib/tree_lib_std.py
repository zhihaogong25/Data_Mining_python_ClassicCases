# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:31:45 2021

@author: Zhihao Gong


The code of decision tree is repeated. 

the input dataSet 's columns represent: feature1, feature2, ..., featureN, label
 

"""

import math
import operator


"""
use the Shannon Entropy formulation to calculate the entropy 
for input dataSet
"""
def calcEntropy(dataSet):
    num = len(dataSet)
    labelCts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCts.keys():
            labelCts[currentLabel] = 0
        labelCts[currentLabel] += 1
    Entropy = 0.0
    for key in labelCts:
        prob = float(labelCts[key])/num
        Entropy -=  prob * math.log(prob,2)
    return Entropy

"""
split dataSet with assigned axis and assigned value
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


"""
choosing the best feature to split the dataSet by maximum the Information Gain
"""
def chooseBestFeatToSplit(dataSet):
    # the total number of features
    numFeat = len(dataSet[0]) - 1
    # calculate the basic Entropy for the whole dataSet
    baseEntropy = calcEntropy(dataSet)
    # initialize the information Gain and corresponding best feature 
    bestInfoGain = 0.0
    bestFeature  = -1
    for i in range(numFeat):
        listSample = [dataVec[i] for dataVec in dataSet ]
        valueset = set(listSample)
        newEntropy = 0.0
        for value in valueset:
            # split the dataSet
            subDataSet = splitDataSet(dataSet, i, value)
            prob_sub   = float(len(subDataSet))/len(dataSet)
            newEntropy += prob_sub * calcEntropy(subDataSet)
        infoGain_i = baseEntropy - newEntropy
        if( infoGain_i > bestInfoGain ):
            bestInfoGain = infoGain_i
            bestFeature  = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount, \
                              key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0]
    


"""
Use iteration to build a decision tree from dataSet and labels
"""

def createTree(dataSet, labels):
    classList = [sample[-1]  for sample in dataSet ]
    # the first End Mark : all labels are same, like "no", which implies that 
    # a pure node is reached
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # the second End Mark : 
    if len(dataSet[0])  == 1 :
        return majorityCnt(classList) 
    # choose the best feature for the input dataSet
    bestFeat = chooseBestFeatToSplit(dataSet)
    # store the label of the best feature
    bestFeatLabel = labels[bestFeat]
    # use the dictionary to initialize the tree
    mytree = {bestFeatLabel:{}}
    #
    del(labels[bestFeat])
    featValue = [ sample[bestFeat] for sample in dataSet]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        sublabel = labels[:]
        mytree[bestFeatLabel][value] = \
            createTree(splitDataSet(dataSet, bestFeat, value), sublabel) 
    return mytree   


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
    
    
"""
use the created decision Tree to classify the test dataSet
"""    
def classify(myTree, testData, testLabel):
    firstStr = list(myTree.keys())[0]
    subTree = myTree[firstStr]
    labelIndx = testLabel.index(firstStr)
    testKey   = testData[labelIndx]
    for key in subTree.keys():
        if testKey == key :
            if type(subTree[key]).__name__ == 'dict':
                classfLabel = classify(subTree[key], testData,testLabel)
            else:
                classfLabel = subTree[key]
    return classfLabel            


"""
store and load the decision
"""

def storeTree(myTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(myTree, fw)
    fw.close()
    
def loadTree(filename):
    import pickle
    fr = open(filename, 'rb')
    inputTree = pickle.load(fr)
    return inputTree
