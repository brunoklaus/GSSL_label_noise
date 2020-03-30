'''
Created on 13 de nov de 2019

@author: klaus
'''
from gssl.filters.filter import GSSLFilter
import numpy as np
import gssl.graph.gssl_utils as gutils
from gssl.classifiers.LGC_tf import LGC_iter_TF
from gssl.graph.gssl_utils import scipy_to_np as _to_np

import scipy.sparse
import itertools

class LGC_LVO_Filter(GSSLFilter):
    
    
    def get_prop_W(self,W,Y,mu):
        alpha = 1/(1+mu)
        #Get D^{-1/2}
        d_sqrt = np.reciprocal(np.sqrt(np.sum(W,axis=0)))
        d_sqrt[np.logical_not(np.isfinite(d_sqrt))] = 1
        d_sqrt = np.diag(d_sqrt)
        
        
        S = np.matmul(d_sqrt,np.matmul(W,d_sqrt))
        I = np.identity(Y.shape[0])
        
        return np.linalg.inv(I - alpha*S)
    
    
    '''
    classdocs
    '''
    @GSSLFilter.autohooks
    def LDST(self,X,W,Y,labeledIndexes,mu = 99.0,useEstimatedFreq=True,tuning_iter = 0,hook=None,
             constant_prop = False,useZ=True,normalize_rows=True):
        '''BEGIN initialization'''
        
        
        
        
        Y = np.copy(Y)
        #We make a deep copy of labeledindexes
        labeledIndexes = np.array(labeledIndexes)        
        lids = np.where(labeledIndexes)[0]
        
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
           
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")

        
        W = 0.5*(W + W.transpose())
        
        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        
        
        D = gutils.deg_matrix(W, flat=True)
        if not useEstimatedFreq is None:
                if isinstance(useEstimatedFreq,bool):
                    estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
                else:
                    estimatedFreq = useEstimatedFreq
                    
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
        
        import scipy.stats
        if scipy.sparse.issparse(W):
            l = np.sum(labeledIndexes)
            
            
            itertool_prod = [[i,j] for i in range(l) for j in range(l)]

            row = np.asarray([lids[i] for i in range(l)])
            col = np.asarray([i for i in range(l)])
            data = np.asarray([1.0]*l)
            temp_Y = _to_np(scipy.sparse.coo_matrix( (data,(row,col)),shape=(W.shape[0],l) ))
            
            PL = LGC_iter_TF(X,W,Y=temp_Y,labeledIndexes=labeledIndexes,alpha = 1/(1+mu),num_iter=1000)
            
            PL = PL[labeledIndexes,:]
            PL[range(PL.shape[0]),range(PL.shape[0])] = 0   #Set diagonal to 0
            
            PL =  PL
            
            
            del temp_Y
            
            row = np.asarray([lids[x[0]] for x in itertool_prod if x[0] != x[1] ])
            col = np.asarray([lids[x[1]] for x in itertool_prod if x[0] != x[1] ])
            data = [PL[x[0],x[1]] for x in itertool_prod if x[0] != x[1]]
            P = scipy.sparse.coo_matrix((data,(row,col)),shape=W.shape).tocsr()
            
            P = P 
        else:
            #Identity matrix
            I = np.identity(W.shape[0])
            #Get graph laplacian
            L = gutils.lap_matrix(W, is_normalized=True)
            #Propagation matrix
            P = np.zeros(W.shape)
            P[np.ix_(labeledIndexes,labeledIndexes)] = np.linalg.inv( I + 0.5*(L + L.transpose())/mu )[np.ix_(labeledIndexes,labeledIndexes)] 
            P[labeledIndexes,labeledIndexes] = 0
            P[np.ix_(labeledIndexes,labeledIndexes)] = P[np.ix_(labeledIndexes,labeledIndexes)]/np.sum(P[np.ix_(labeledIndexes,labeledIndexes)],axis=0,keepdims=False)
            
        
        W = scipy.sparse.csr_matrix(W)
        
        Z = []
        
        detected_noisylabels = []
        suggested_labels = []
        
        
        Y_flat = np.argmax(Y,axis=1)
        
        def divide_row_by_sum(e):
            
            e = _to_np(e)
            if normalize_rows:
                e = e / np.sum(e + 1e-100,axis=1, keepdims=True)
                return e
            else:
                return e
            
        def find_argmin(Q,class_to_unlabel):
            id_min_line = np.argmin(Q[:,class_to_unlabel])
            id_min_col = class_to_unlabel
            return id_min_line,id_min_col,Q[id_min_line,id_min_col]
        #######################################################################################
        '''BEGIN iterations'''
            
        
        Q = None
        cleanIndexes = np.copy(labeledIndexes)
        for i_iter in range(tuning_iter):
             
            found_noisy = True
            if np.sum(labeledIndexes) > 0 and found_noisy:
                
                '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                    Then, we normalize each row so that row sums to its estimated influence
                '''
                if (not self.use_baseline) or Q is None:
                    if useZ:
                        Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq, weigh_by_degree=False)
                        F = P@Z
                        if scipy.sparse.issparse(F):
                            F = np.asarray(F.toarray())
                
                        
                        #Compute graph gradient
                        Q = (divide_row_by_sum(F) - divide_row_by_sum(Z))
                    else:
                        F = P@Y
                        if scipy.sparse.issparse(F):
                            F = np.asarray(F.toarray())
                        Q = (divide_row_by_sum(F) - divide_row_by_sum(Y))
                #import scipy.stats
                
                #During label tuning, we'll also 'unlabel' the argmax
                unlabeledIndexes = np.logical_not(cleanIndexes)
                if self.early_stop:
                    Q[np.sum(F,axis=1) == 0.0,:] = 9999
                
                Q[unlabeledIndexes,:] = np.inf
                
                #Find minimum unlabeled index
                if constant_prop:
                    expectedNumLabels = estimatedFreq * np.sum(labeledIndexes)
                    actualNumLabels = np.sum(Y[labeledIndexes,:],axis=0)
                    temp = expectedNumLabels-actualNumLabels
                    class_priority = np.argsort(temp)
                    
                    found_noisy = False
                    for class_to_unlabel in class_priority:
                        id_min_line,id_min_col, val = find_argmin(Q, class_to_unlabel)
                        if val <  0:
                            #This means that the class would have a different label under the modified label prop    
                            found_noisy = True
                            break
                    
                        
                else:
                    id_min = np.argmin(Q)    
                    id_min_line = id_min // num_classes
                    id_min_col = id_min % num_classes   #The class previously assigned to instance X_{id_min_line}
                    found_noisy = Q[id_min_line,id_min_col] < 0

                if found_noisy:
                        
                    id_max_col = np.argmax(Q[id_min_line,:]) #The new, suggested class
                    
                    
                    detected_noisylabels.append(id_min_col)
                    suggested_labels.append(id_max_col)
                    
    
                    
                    
                    
                    #Unlabel OP
                    if labeledIndexes[id_min_line] == False:
                        raise Exception("Error: unlabeled instance was selected")
                    if not Y[id_min_line,id_min_col] == 1:
                        raise Exception("Error: picked wrong class to unlabel")
                    
                    labeledIndexes[id_min_line] = False
                    cleanIndexes[id_min_line] = False
                    
                    Y[id_min_line,id_min_col] = 0
                    if self.relabel:
                        labeledIndexes[id_min_line] = True
                        Y[id_min_line,:] = 0
                        Y[id_min_line,id_max_col] = 1
                    
                    

            if not hook is None:
                hook._step(step=(i_iter+1),X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)
            
            
            '''
            MATPLOTLIB stuff 
            '''
            """
                
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            lids = np.where(labeledIndexes)[0]
            
            PL = _to_np(P)[np.ix_(labeledIndexes,labeledIndexes)]
            PL = divide_row_by_sum(PL)
            
            Y_flat_l = Y_flat[labeledIndexes]
            
            from sklearn.preprocessing import scale
            QL = (Q[lids,:] - np.mean(Q[lids,:]))/(np.std(Q[lids,:])+1e-10)
            
            
            fig, ax = plt.subplots(1,2)
            
            ax[0].matshow(PL)
            ax[1].matshow(QL)
            
            for (i, j), z in np.ndenumerate(PL):
                if Y_flat_l[i] != Y_flat_l[j]:
                    z = -z
                ax[0].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
            for (i, j), z in np.ndenumerate(QL):
                ax[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
                
            
            print(PL.shape)
            
            plt.show()
            """    
            
        '''END iterations'''
        print("NUMBER OF DETECTED NOISY INSTANCES:{}".format(len(detected_noisylabels)))    

        
        
        
        return Y, labeledIndexes

            
        
        

    def fit (self,X,Y,labeledIndexes,W = None,hook=None):
        if self.tuning_iter_as_pct:
            tuning_iter = int(round(self.tuning_iter * X.shape[0]))
        else:
            tuning_iter = self.tuning_iter
        

        return self.LDST(X, W, Y, labeledIndexes, self.mu, self.useEstimatedFreq, tuning_iter,\
                          hook, self.constantProp,self.useZ,self.normalize_rows)
    
    def __init__(self, tuning_iter,mu = 99.0, useEstimatedFreq=True,constantProp=False,useZ=True,
                 tuning_iter_as_pct=False, normalize_rows = True, early_stop=False, use_baseline=False,
                 relabel=True):
        """ Constructor for LDST-Removal Filter.
        
        Args:
            mu (float) :  a parameter determining the importance of the fitting term. Default is ``99.0``.
            tuning_iter (int) : The number of tuning iterations. 
            useEstimatedFreq (bool) : If ``True``, then use estimated class freq. to balance the propagation.
                    Otherwise, assume classes are equiprobable. Default is ``True``.
            
        """
    
        self.mu = mu
        self.tuning_iter = tuning_iter
        self.useEstimatedFreq = useEstimatedFreq
        self.constantProp = constantProp
        self.useZ = useZ
        self.tuning_iter_as_pct = tuning_iter_as_pct
        self.normalize_rows = normalize_rows
        self.early_stop = early_stop
        self.relabel = relabel
        self.use_baseline = use_baseline