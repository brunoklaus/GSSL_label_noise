'''
Created on 27 de mar de 2019

@author: klaus
'''
from gssl.classifiers.classifier import GSSLClassifier
import numpy as np
import gssl.graph.gssl_utils as gutils
import os
import datetime
import log.logger as LOG
import scipy.sparse
class GTAMClassifier(GSSLClassifier):
    """ Classifier using Graph Transduction Through Alternating Minimization (GTAM - see :cite:`Wang2008`).
    
    """

    @GSSLClassifier.autohooks
    def __GTAM(self,X,W,Y,labeledIndexes,mu = 99.0,useEstimatedFreq=True,num_iter = None,
             constant_prop=False,hook=None):
        '''BEGIN initialization'''
        Y = np.copy(Y)
        labeledIndexes = np.array(labeledIndexes)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        

        
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        
        """ Estimate frequency of classes"""
        if not useEstimatedFreq is None:
                if isinstance(useEstimatedFreq,bool):
                    estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
                else:
                    estimatedFreq = useEstimatedFreq
                    
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
        LOG.debug("Estimated frequency: {}".format(estimatedFreq),LOG.ll.CLASSIFIER)

        
        """ IMPORTANT! ERASES LABELS """        
        Y[np.logical_not(labeledIndexes),:] = 0
        
        D = gutils.deg_matrix(W, flat=True)
        #Identity matrix
        I = np.identity(W.shape[0])
        #Get graph laplacian
        L = gutils.lap_matrix(W, is_normalized=True)
        #Propagation matrix
        P = np.linalg.inv( I + L/mu )
        
        P_t = P.transpose()
        #Matrix A
        A = ((P_t @ L) @ P) + mu* ((P_t - I) @ (P - I))
        
        A = np.asarray(A)
        #A = A + A.transpose()
        
        W = scipy.sparse.coo_matrix(W)
        Z = []
        
        Q = None
        
        def divide_row_by_sum(e):
            e = gutils.scipy_to_np(e)
            e = e / np.sum(e + 1e-100,axis=1, keepdims=True)
            return e
        
        #Determine nontuning iter
        if num_iter is None:
            num_iter = num_unlabeled
        else:
            num_iter = min(num_iter,num_unlabeled)
            
        id_min_line, id_min_col = -1,-1
        '''END initialization'''
        #######################################################################################
        '''BEGIN iterations'''
        for i in np.arange(num_iter):

            '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                Then, we normalize each row so that row sums to its estimated influence
            '''
            ul = np.logical_not(labeledIndexes)
            
            Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq,weigh_by_degree=self.weigh_by_degree)
            if Q is None:
                #Compute graph gradient
                Q = np.matmul(A,Z)
                if not hook is None:
                    Q_pure = np.copy(Q)
                
                Q[labeledIndexes,:] = np.inf
                
            else:
                Q[id_min_line,:] = np.inf
                new_el_pct = Z[id_min_line,id_min_col] / np.sum(Z[:,id_min_col])
                Q[ul,id_min_col] =\
                 (1 - new_el_pct) * Q[ul,id_min_col] + Z[id_min_line,id_min_col] * A[ul,id_min_line]
            
            #Find minimum unlabeled index
            
            if constant_prop:
                    expectedNumLabels = estimatedFreq * sum(labeledIndexes)
                    actualNumLabels = np.sum(Y[labeledIndexes],axis=0)
                    class_to_label = np.argmax(expectedNumLabels-actualNumLabels)
                    id_min_col = class_to_label
                    id_min_line = np.argmin(Q[:,class_to_label])
                
                    
            else:
                id_min = np.argmin(Q)
                id_min_line = id_min // num_classes
                id_min_col = id_min % num_classes
            
                
            
            #Update Y and labeledIndexes
            labeledIndexes[id_min_line] = True
            Y[id_min_line,id_min_col] = 1
            
            
            
            #Maybe plot current iteration
            
            
            if not hook is None:
                hook._step(step=i,Y=Y,labeledIndexes=labeledIndexes,P=P,Z=Z,Q=Q_pure,
                           id_min_line=id_min_line,id_min_col=id_min_col)
        '''END iterations'''    
        ######################################################################################################
        
        return np.asarray(P@Z)
    
    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__GTAM(X,W,Y,labeledIndexes,
                           mu=self.mu,
                           useEstimatedFreq=self.useEstimatedFreq,
                           num_iter=self.num_iter,
                           constant_prop = self.constantProp,
                           hook = hook
                           ))


    def __init__(self, mu = 99.0,num_iter=None,useEstimatedFreq=True,constantProp=False,know_true_freq=True,weigh_by_degree=True):
        """" Constructor for GTAM classifier.
        
        Args:
            mu (float) :  a parameter determining the importance of the fitting term. Default is ``99.0``.
            num_iter (int) : Optional. The number of iterations to run. The default behaviour makes it N iterations given
                a NDArray[float].shape[N,D] input matrix.
            useEstimatedFreq (Union[bool,NDArray[C],None]) : If ``True``, then use estimated class freq. to balance the propagation.
                If it is a float array, it uses that as the frequency. If ``None``, assumes classes are equiprobable. Default is ``True``.
            useConstantProp (bool) : If ``True``, then use try to maintain a constant proportion of labels
                in all iterations.
                    
            
        """
        self.mu = mu
        self.num_iter = num_iter
        self.useEstimatedFreq = useEstimatedFreq
        self.constantProp = constantProp
        self.know_true_freq = know_true_freq
        self.weigh_by_degree = weigh_by_degree
        