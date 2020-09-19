'''
Created on 1 de abr de 2019

@author: klaus
'''
from gssl.filters.filter import GSSLFilter
import numpy as np
import gssl.graph.gssl_utils as gutils
class LDSTRemover(GSSLFilter):
    
    
    
    '''
    classdocs
    '''
    @GSSLFilter.autohooks
    def LDST(self,X,W,Y,labeledIndexes,mu = 99.0,useEstimatedFreq=True,tuning_iter = 0,hook=None,
             constant_prop = False,useZ=True):
        '''BEGIN initialization'''
        Y = np.copy(Y)
        #We make a deep copy of labeledindexes
        labeledIndexes = np.array(labeledIndexes)
        
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
           
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")

        W = 0.5*(W + W.transpose())

        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        D = gutils.deg_matrix(W,flat=True)
        """ Estimate frequency of classes"""
        if not useEstimatedFreq is None:
                if isinstance(useEstimatedFreq,bool):
                    estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
                else:
                    estimatedFreq = useEstimatedFreq
                    
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
        
        #Identity matrix
        I = np.identity(W.shape[0])
        #Get graph laplacian
        L = gutils.lap_matrix(W, is_normalized=True)
        #Propagation matrix
        """ !!!!!! """
        P = np.linalg.inv( I + 0.5*(L + L.transpose())/mu )
        #P = np.zeros(W.shape)
        #P[np.ix_(labeledIndexes,labeledIndexes)] = np.linalg.inv( I + 0.5*(L + L.transpose())/mu )[np.ix_(labeledIndexes,labeledIndexes)] 
            
        P_t = P.transpose()
        #Matrix A
        A = ((P_t @ L) @ P) + mu*((P_t - I) @ (P - I))
        
        Z = []
        
        #######################################################################################
        '''BEGIN iterations'''
        for i_iter in np.arange(tuning_iter):
            
            
            if np.sum(labeledIndexes) > 0:
                
                '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                    Then, we normalize each row so that row sums to its estimated influence
                '''
                
                if useZ:
                    Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq,weigh_by_degree=self.weigh_by_degree)
                    #Compute graph gradient
                    Q = np.matmul(A,Z)
                    
                else:
                    Q = np.matmul(A,Y)
                    
                for i_labeled in np.where(labeledIndexes)[0]:
                    assigned_class  = np.argmax(Y[i_labeled,:])
                    other_classes = list(range(Y.shape[1]))
                    other_classes.remove(assigned_class)
                    
                    best_other = min([Q[i_labeled,j] for j in other_classes])
                    
                    for j in range(Y.shape[1]):
                        if self.gradient_fix:
                            Q[i_labeled,assigned_class] = -best_other
                        Q[i_labeled,other_classes] = -np.inf
                #During label tuning, we'll also 'unlabel' the argmax
                unlabeledIndexes = np.logical_not(labeledIndexes)
                Q[unlabeledIndexes,:] = -np.inf
                
                #Find minimum unlabeled index
                if constant_prop:
                    raise""
                    """expectedNumLabels = estimatedFreq * sum(labeledIndexes)
                    actualNumLabels = np.sum(Y[labeledIndexes],axis=0)
                    class_to_unlabel = np.argmax(actualNumLabels - expectedNumLabels)
                    
                    id_max_line = np.argmax(Q[:,class_to_unlabel])
                    id_max_col = class_to_unlabel
                    """
                    
                else:
                    id_max = np.argmax(Q)    
                    id_max_line = id_max // num_classes
                    id_max_col = id_max % num_classes
                    
                
                if not Y[id_max_line,id_max_col] == 1:
                    print(Y[id_max_line,:])
                    raise Exception("Tried to remove label from unlabeled instance")
                
                #Unlabel OP
                labeledIndexes[id_max_line] = False
                Y[id_max_line,id_max_col] = 0
                
            if not hook is None:
                hook._step(step=i_iter+1,X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)
            
        
        '''END iterations'''    
        return Y, labeledIndexes

            
        
        

    def fit (self,X,Y,labeledIndexes,W = None,hook=None):
        if self.tuning_iter_as_pct:
            l = np.sum(labeledIndexes)
            tuning_iter = int(round(self.tuning_iter *l))            
        else:
            tuning_iter = self.tuning_iter
            
        return self.LDST(X, W, Y, labeledIndexes, self.mu, self.useEstimatedFreq, tuning_iter,\
                          hook, self.constantProp,self.useZ)
    
    def __init__(self, tuning_iter,mu = 99.0, useEstimatedFreq=True,constantProp=False,useZ=True,
                 tuning_iter_as_pct=False,know_true_freq=False,weigh_by_degree=True,gradient_fix=True):
        """ Constructor for LDST-Removal Filter.
        
        Args:
            mu (float) :  a parameter determining the importance of the fitting term. Default is ``99.0``.
            tuning_iter (Union[int,float]) : The number of tuning iterations. 
            tuning_iter_as_pct (bool) :  If  ``True``, then  `tuning_iter` is to be interpreted as a percentage of the 
                number of labels.
            
            useEstimatedFreq (Union[bool,NDArray[C],None]) : If ``True``, then use estimated class freq. to balance the propagation.
                If it is a float array, it uses that as the frequency. If ``None``, assumes classes are equiprobable. Default is ``True``.
            constantProp (bool) : If  ``True``, whenever a label of a given class is removed, another label from the same
                class gets added. Default is `False`.
            
            useZ (bool) : If ``True``, then at each step update label matrix so that each class has total influence
                equal to the estimated frequency. Default is ``True``.
            weigh_by_degree (bool) : If ``True`` and `useZ` also ``True``, then vertice with higher degree will 
                have more confident labels. Default is ``False``.
            gradient_fix (bool) : If ``True``, changes the criteria when selecting the Q matrix to discourage picking the same 
            class. Default is ``True``, and should be kept that way for better performance.
                
            
              
            
        """
        self.mu = mu
        self.tuning_iter = tuning_iter
        self.useEstimatedFreq = useEstimatedFreq
        self.constantProp = constantProp
        self.useZ = useZ
        self.tuning_iter_as_pct = tuning_iter_as_pct
        self.know_true_freq = know_true_freq
        self.weigh_by_degree = weigh_by_degree
        self.gradient_fix = gradient_fix