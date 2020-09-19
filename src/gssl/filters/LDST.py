'''
Created on 1 de abr de 2019

@author: klaus
'''
from gssl.filters.filter import GSSLFilter
import numpy as np
import gssl.graph.gssl_utils as gutils
from docutils.nodes import Labeled
class LDST(GSSLFilter):

    '''
    classdocs
    '''
    @GSSLFilter.autohooks
    def LDST(self,X,W,Y,labeledIndexes,mu = 99.0,useEstimatedFreq=True,tuning_iter = 0,hook=None,
             constant_prop = False,useZ=False,weigh_by_degree=False):
        '''BEGIN initialization'''
        
        Y = np.copy(Y)
        #We make a deep copy of labeledindexes
        labeledIndexes = np.array(labeledIndexes)
        
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
           
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        D = np.sum(W,axis=0)
        if useEstimatedFreq:
            estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
            
            
        #Identity matrix
        I = np.identity(W.shape[0])
        #Get graph laplacian
        L = gutils.lap_matrix(W, is_normalized=True)
        #Propagation matrix
        P = np.linalg.inv( I + 0.5*(L + L.transpose())/mu )
        P_t = P.transpose()
        #Matrix A
        A = ((P_t @ L) @ P) + mu*((P_t - I) @ (P - I))
        A = A + A.transpose()
        
        Z = []
        
        #######################################################################################
        '''BEGIN iterations'''
        for i in np.arange(tuning_iter):

            
            '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                Then, we normalize each row so that row sums to its estimated influence
            '''
            if useZ:    
                Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq,weigh_by_degree=weigh_by_degree,reciprocal=False)
                Q = np.matmul(A,Z)
            else:
                Q = np.matmul(A,Y)
            
            #During label tuning, we'll also 'unlabel' the argmax
            
            unlabeledIndexes = np.logical_not(labeledIndexes)
            temp = Q[unlabeledIndexes,:]
            Q[unlabeledIndexes,:] = -np.inf
            id_max = np.argmax(Q)

            id_max_line = id_max // num_classes
            id_max_col = id_max % num_classes

            Q[unlabeledIndexes,:] = temp
                
            Q[labeledIndexes,:] = np.inf
            
            #Find minimum unlabeled index
            if constant_prop:
                id_min_line = np.argmin(Q[:,id_max_col])
                id_min_col = id_max_col
            else:
                id_min = np.argmin(Q)
                id_min_line = id_min // num_classes
                id_min_col = id_min % num_classes
            
            
            #Label OP
            labeledIndexes[id_min_line] = True
            Y[id_min_line,id_min_col] = 1
            
            #Unlabel OP
            labeledIndexes[id_max_line] = False
            Y[id_max_line,id_max_col] = 0
            
            
            
            
            if not hook is None:
                hook._step(step=i,X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,
                           l_i=id_max_line,l_j=id_max_col,ul_i=id_min_line,ul_j=id_min_col)
            
            
        '''END iterations'''    

        return Y, labeledIndexes

            
        
        

    def fit (self,X,Y,labeledIndexes,W = None,hook=None):
        if self.tuning_iter_as_pct:
            l = np.sum(labeledIndexes)
            tuning_iter = int(round(self.tuning_iter *l))            
        else:
            tuning_iter = self.tuning_iter
        
        return self.LDST(X, W, Y, labeledIndexes, self.mu, self.useEstimatedFreq, tuning_iter, hook, \
                         self.constantProp,self.useZ,self.weigh_by_degree)
    
    def __init__(self, tuning_iter,mu = 99.0, useEstimatedFreq=True,constantProp=False,useZ=True,
                 weigh_by_degree=False):
        """" Constructor for the LDST filter.
        
        Args:
            mu (float) :  a parameter determining the importance of the fitting term. Default is ``99.0``.
            tuning_iter (int) : The number of tuning iterations. 
            useEstimatedFreq (Union[bool,NDArray[C],None]) : If ``True``, then use estimated class freq. to balance the propagation.
                If it is a float array, it uses that as the frequency. If ``None``, assumes classes are equiprobable. Default is ``True``.
            constantProp (bool) : If  ``True``, whenever a label of a given class is removed, another label from the same
                class gets added. Default is `False`.
            useZ (bool) : If ``True``, then at each step update label matrix so that each class has total influence
               equal to the estimated frequency. Default is ``True``.
            weigh_by_degree (bool) : If ``True`` and ``useZ``` also ``True``, then vertices with higher degree will 
                have more confident labels. Default is ``False``.
            
               
            
            
            """
        self.mu = mu
        self.tuning_iter = tuning_iter
        self.useEstimatedFreq = useEstimatedFreq
        self.constantProp = constantProp
        self.useZ = useZ
        self.weigh_by_degree = weigh_by_degree
        