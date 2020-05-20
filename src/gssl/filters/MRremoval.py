'''
Created on 1 de abr de 2019

@author: klaus
'''
from gssl.filters.filter import GSSLFilter
import numpy as np
import scipy.linalg as sp
import log.logger as LOG
import gssl.graph.gssl_utils as gutils
from gssl.graph.gssl_utils import scipy_to_np
class MRRemover(GSSLFilter):
    
    @GSSLFilter.autohooks
    def __MR(self,X,W,Y,labeledIndexes,p,tuning_iter,hook=None):
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y[np.logical_not(labeledIndexes)] = 0
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        l = np.reshape(np.array(np.where(labeledIndexes)),(-1))
        num_lab = l.shape[0]
        
        
        if not isinstance(p, int):
            p = int(p * num_lab)
        if p > Y.shape[0]:
            p = Y.shape[0]
            LOG.warn("Warning: p greater than the number of labeled indexes",LOG.ll.FILTER)
        
        
        W = scipy_to_np(W)
        L = gutils.lap_matrix(W, is_normalized=False)
        D = gutils.deg_matrix(W)
        def check_symmetric(a, tol=1e-8):
            return np.allclose(a, a.T, atol=tol)    
        
        if check_symmetric(L):
            E = sp.eigh(L,D,eigvals=(1,p))[1]
        else:
            LOG.warn("Warning: Laplacian not symmetric",LOG.ll.FILTER)
            eigenValues, eigenVectors = sp.eig(L,D)
            idx = eigenValues.argsort() 
            eigenValues = eigenValues[idx]
            assert eigenValues[0] <= eigenValues[eigenValues.shape[0]-1]
            eigenVectors = eigenVectors[:,idx]
            E = eigenVectors[:,1:(p+1)]
        
        
        e_lab = E[labeledIndexes,:]
        """ TIKHONOV REGULARIZATION. Currently set to 0."""
        TIK = np.zeros(shape=e_lab.shape)
        try:
            A = np.linalg.inv(e_lab.T @ e_lab + TIK.T@TIK) @ e_lab.T        
        except:
            A = np.linalg.pinv(e_lab.T @ e_lab + TIK.T@TIK) @ e_lab.T        
        F = np.zeros(shape=Y.shape)
        
        y_m =  np.argmax(Y, axis=1)[labeledIndexes]
        

        
        
        for i in range(Y.shape[1]):
            c = np.ones(num_lab)
            c[y_m != i] = -1
            a = A @ np.transpose(c)
            LOG.debug(a,LOG.ll.FILTER)
            for j in np.arange(F.shape[0]):
                F[j,i] = np.dot(a,E[j,:])
        
        ERmat = -1*np.ones((Y.shape[0],))
        
        
        Y_amax = np.argmax(Y, axis=1)
        for i in np.where(labeledIndexes):
            ERmat[i] = np.square(Y[i,Y_amax[i]]-F[i,Y_amax[i]])
        
        removed_Lids = np.argsort(ERmat)
        removed_Lids = removed_Lids[::-1]
        
        
        
        labeledIndexes = np.array(labeledIndexes)
        Y = np.copy(Y)
        for i in range(tuning_iter):
            labeledIndexes[removed_Lids[i]] = False
            if not hook is None:
                hook._step(step=i,X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)
        
        return Y, labeledIndexes
        
        

    def fit (self,X,Y,labeledIndexes,W = None,hook=None):
        if self.tuning_iter_as_pct:
            tuning_iter = round(self.tuning_iter * X.shape[0])
        else:
            tuning_iter = self.tuning_iter
        return(self.__MR(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,p=self.p,
                         tuning_iter=tuning_iter,hook=hook))
    
    def __init__(self, p=0.2,tuning_iter=0, tuning_iter_as_pct=False):
        """" Constructor for Manifold Regularization Filter.            
        """
        self.p = p
        self.tuning_iter = tuning_iter
        self.tuning_iter_as_pct = tuning_iter_as_pct