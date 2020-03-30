'''
Created on 27 de mar de 2019

@author: klaus
'''
from gssl.classifiers.classifier import GSSLClassifier
import numpy as np
import gssl.graph.gssl_utils as gutils

class GFHF(GSSLClassifier):
    """ Label Propagation based on Gaussian Fields and Harmonic Functions. See :cite:`Zhu_2003`. """

    @GSSLClassifier.autohooks
    def __GFHF_iter(self,W,Y,labeledIndexes,num_iter,  hook = None):
        W = W.todense()
        Y = np.copy(Y)
        
        Y[np.logical_not(labeledIndexes)] = 0
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        
        P  = gutils.deg_matrix(W,-1.0) @ W
        Yl = Y[labeledIndexes,:]
        for i in range(num_iter):
            Y = P@Y
            Y[labeledIndexes,:] = Yl
            
        return Y
        
        
    @GSSLClassifier.autohooks
    def __GFHF(self,W,Y,labeledIndexes, hook = None):
        W = W.todense()
        Y = np.copy(Y)
        Y[np.logical_not(labeledIndexes)] = 0
        
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        u = np.reshape(np.array(np.where(np.logical_not(labeledIndexes))),(-1))
        l = np.reshape(np.array(np.where(labeledIndexes)),(-1))
        
        d_inv = np.reciprocal(np.sum(W,axis=0))
        d_inv[np.logical_not(np.isfinite(d_inv))] = 1
        d_inv = np.diag(d_inv)
        
        P  = gutils.deg_matrix(W,-1.0) @ W
        
        I = np.identity(Y.shape[0] - sum(labeledIndexes))
        
        P_ul = P[u[:, None],l]
        P_uu = P[u[:, None],u]
        
        try:
            Y[u,:] = np.linalg.inv(I - P_uu) @ P_ul @ Y[l,:]
        except:
            Y[u,:] = np.linalg.pinv(I - P_uu) @ P_ul @ Y[l,:]
        

        return(Y)

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        if self.num_iter is None:
            return(self.__GFHF(W=W,Y=Y,labeledIndexes=labeledIndexes,hook=hook))
        else:
            return(self.__GFHF_iter(W=W,Y=Y,labeledIndexes=labeledIndexes,num_iter=self.num_iter,hook=hook))
    def __init__(self,num_iter=None):
        """ Constructor for GFHF classifier.
            
        Args:
            None
        """
        self.num_iter = num_iter