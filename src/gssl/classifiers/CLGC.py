'''
Created on 23 de jun de 2020

@author: klaus
'''
import numpy as np
import gssl.graph.gssl_utils as gutils
from gssl.classifiers.classifier import GSSLClassifier

import log.logger as LOG

class CLGCClassifier(GSSLClassifier):
    """ Constrained Local and Global Consistency Classifier. See :cite:`sousaconstrained`.
    """

    @property
    def alpha(self):
        """ :math:`\\alpha` such that :math:`\\mu = \\frac{1-\\alpha}{\\alpha}` is the regularization factor multiplying
            the label fitting criterion. It is required that :math:`0 < \\alpha < 1`.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value <= 0 or value >= 1:
            raise ValueError("alpha must be greater than 0, less than 1.Found {}".format(value))
        self._alpha = value

    @GSSLClassifier.autohooks
    def __LGC(self,X,W,Y,labeledIndexes, alpha = 0.1, useEstimatedFreq = None, hook=None):
        
        """ Init """
        import scipy.sparse
        if scipy.sparse.issparse(W):
            W = W.todense()
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        """ Estimate frequency of classes"""
        num_labeled = Y[labeledIndexes].shape[0]
        num_classes = Y.shape[1]
        if not useEstimatedFreq is None:
                if isinstance(useEstimatedFreq,bool):
                    estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
                else:
                    estimatedFreq = useEstimatedFreq
                    
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
        omega = estimatedFreq
        
        
        """  """
        mu = (1-alpha)/alpha
        n = Y.shape[0]
        c = Y.shape[1]
        
        I = np.identity(Y.shape[0])
        S = I - gutils.lap_matrix(W, is_normalized=True)
        
        """ stuff that has matrix multiplication with theta """
        theta = (1/mu)*np.asarray(np.linalg.inv(I - alpha*S))
        F_lgc = (theta@Y)*mu
        theta_1n = np.sum(theta,axis=1).flatten()
        theta_1n_ratio = (theta_1n/(np.sum(theta_1n)))[:,np.newaxis] #Shape: nx1
        print(theta_1n_ratio.shape)
        """ Intermediate calc """
        zeta = n*omega - np.sum(F_lgc,axis=0) #Shape: 1xc
        zeta = np.reshape(zeta,(1,c))
        
        ypsilon = np.ones(shape=(n,1)) - np.sum(F_lgc,axis=1)[:,np.newaxis] -\
             theta_1n_ratio * (n - np.sum(F_lgc.flatten())) #Shape: nx1
        
        F =  F_lgc 
        F += theta_1n_ratio @ zeta 
        F +=  (1/c)*(ypsilon@ np.ones((1,c)))
        
        log_args = [np.round(x,3) for x in [np.sum(F,axis=1)[0:10],  np.sum(F,axis=0), n*omega]]        
        LOG.info("F sum on rows: {} (expected 1,1,...,1); F sum col: {} (expected {})".format(*log_args) )
        
        return F
    

    
    @GSSLClassifier.autohooks
    def __LGC_iter_TF(self,X,W,Y,labeledIndexes, alpha = 0.1,useEstimatedFreq = True, num_iter = 1000, hook=None):
        from gssl.classifiers.LGC_tf import LGC_iter_TF
        """ Init """
        import scipy.sparse
        if not scipy.sparse.issparse(W):
            W = scipy.sparse.csr_matrix(W)
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        """ Estimate frequency of classes"""
        num_labeled = Y[labeledIndexes].shape[0]
        num_classes = Y.shape[1]
        if not useEstimatedFreq is None:
                if isinstance(useEstimatedFreq,bool):
                    estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
                else:
                    estimatedFreq = useEstimatedFreq
                    
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
        omega = estimatedFreq
        
        
        """  """
        mu = (1-alpha)/alpha
        n = Y.shape[0]
        c = Y.shape[1]
        print(np.concatenate([Y,np.ones((n,1))],axis=1))
        
        """ stuff that has matrix multiplication with theta """
        PY1 = LGC_iter_TF(X, W, np.concatenate([Y,np.ones((n,1))],axis=1), labeledIndexes, alpha, num_iter, hook)
        PY1 = np.asarray(PY1)
        F_lgc, theta_1n = (1/mu)*PY1[:,:-1] , (1/mu)*PY1[:,-1] 
        theta_1n_ratio = (theta_1n/(np.sum(theta_1n)))[:,np.newaxis] #Shape: nx1
        
        """ Intermediate calc """
        zeta = n*omega - np.sum(F_lgc,axis=0) #Shape: 1xc
        zeta = np.reshape(zeta,(1,c))
        
        ypsilon = np.ones(shape=(n,1)) - np.sum(F_lgc,axis=1)[:,np.newaxis] -\
             theta_1n_ratio * (n - np.sum(F_lgc.flatten())) #Shape: nx1
        
        F =  F_lgc 
        F +=  theta_1n_ratio @ zeta 
        F +=  (1/c)*(ypsilon@ np.ones((1,c)))
        import pandas as pd
        print(pd.Series(np.argmax(F,axis=1)).value_counts()/n)
        
        log_args = [np.round(x,3) for x in [np.sum(F,axis=1)[0:10],  np.sum(F,axis=0), n*omega]]        
        LOG.info("F sum on rows: {} (expected 1,1,...,1); F sum col: {} (expected {})".format(*log_args) )
        
        return F
    

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        if self.num_iter is None:
            return(self.__LGC(X,W, Y, labeledIndexes,  self.alpha, self.useEstimatedFreq, hook))
        else:
            return(self.__LGC_iter_TF(X,W, Y, labeledIndexes, self.alpha, self.useEstimatedFreq, self.num_iter, hook))
    
    def __init__(self, alpha = 0.1, num_iter = None, useEstimatedFreq=True):
        """ Constructor for the LGC classifier.
            
            Args:
                alpha (float): A value between 0 and 1 (not inclusive) for alpha.
                num_iter (float): Optional. If not ``Ǹone``, the number of iteration updates to perform
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.useEstimatedFreq = useEstimatedFreq
