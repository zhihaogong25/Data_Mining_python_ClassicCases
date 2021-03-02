# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:54:52 2021

@author: Zhihao Gong



realize the PCA methods to reduce the 
dimension of Datas

"""



import numpy as np
from numpy import linalg as LA

"""
the row * col dataMat is input as : each column represents a feature, the total
number of features = col; the total numbers of sampling for each feature = row   

1. Eliminate mean value 
2. Calculate covariance matrix
3. Calculate eigenvalues and eigenvectors for covariance matrix
4. Sort the eigenvalues from largest to the smallest
5. Keep first N=topNfeat eigenvalues
6. Transform the data into the new space spanned by the N=topNfeat eigenvectors 


 the reduced row * topNfeat matrix & the reconstructed dataset in original space 
 are output

"""

def pca(dataMat, topNfeat = 9999999):
    meanVals          = np.mean(dataMat, axis = 0)
    meanRemoved       = dataMat - meanVals
    # use np.cov to calculate the covariance matrix, since the covaraince bewteen
    # columns is calculated, the parameter "rowvar=False" is used
    covMat            = np.cov(meanRemoved, rowvar=False)
    # use LA.eig to calculate eigensystems, here np.mat() is used since the input 
    # shall be interpreted as a matrix
    eigVals, eigVects = LA.eig(np.mat(covMat))
    # argsort is used to return the index of sorted array
    eigValInd         = np.argsort(eigVals)
    # :-(topNfeat + 1):-1 is used to 1. truncate the # of dimension, 2. inverse the sort 
    eigValInd         = eigValInd[:-(topNfeat + 1):-1]
    #  the eigen equation is given as 
    #  covMat*eigVects[:,n] = eigVals[n]*eigVects[:,n]
    redEigVects       = eigVects[:,eigValInd]
    # since for each sampling vectors, Y = U^T X , Thus we have X^T U = Y, 
    # pile all X^Ts gives the original dataMat
    # the output lowDDataMat reduce to row * topNfeat matrix
    lowDDataMat       = dataMat * redEigVects
    # in the original space, the reconstructed dataset is returned to be used to test
    reconMat          = ( lowDDataMat * np.transpose(redEigVects) ) + meanVals
    # the reduced row * topNfeat matrix & the reconstructed dataset in original space 
    # are output
    return lowDDataMat, reconMat
    
    
                         
                         
    
    