'''
Created on 27 de mar de 2019

@author: klaus
'''
from gssl.classifiers.classifier import GSSLClassifier
import numpy as np
import gssl.graph.gssl_utils as gutils
import scipy.linalg as sp
import scipy.sparse
import log.logger as LOG

class LapEigLS(GSSLClassifier):
    """ Manifold Regularization with Laplacian Eigenmaps. 
        Minimizes the Least Squares in a semi-supervised way by using a linear combination of the first :math:`p` eigenfunctions. See :cite:`belkin2003`.
    """

    @GSSLClassifier.autohooks
    def __MR(self,X,W,Y,labeledIndexes,p,hook=None):
        Y = np.copy(Y)
        if Y.ndim == 1:
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
            LOG.warn("Warning: p greater than the number of labeled indexes",LOG.ll.CLASSIFIER)
        W = gutils.scipy_to_np(W)
        W =  0.5* (W + W.T)
        L = gutils.lap_matrix(W, is_normalized=False)
        D = gutils.deg_matrix(W)
        
        def check_symmetric(a, tol=1e-8):
            return np.allclose(a, a.T, atol=tol)
        def is_pos_sdef(x):
            return np.all(np.linalg.eigvals(x) >= -1e-06)
       
        
        if check_symmetric(L):
            eigenVectors, E = sp.eigh(L,D,eigvals=(1,p))
        else:
            LOG.warn("Warning: Laplacian not symmetric",LOG.ll.CLASSIFIER)
            eigenValues, eigenVectors = sp.eig(L,D)
            idx = eigenValues.argsort() 
            eigenValues = eigenValues[idx]
            assert eigenValues[0] <= eigenValues[eigenValues.shape[0]-1]
            eigenVectors = eigenVectors[:,idx]
            E = eigenVectors[:,1:(p+1)]
        
        
        
        
        
        e_lab = E[labeledIndexes,:]
        #TIK = np.ones(shape=e_lab.shape)
        TIK = np.zeros(shape=e_lab.shape)
        try:
            A = np.linalg.inv(e_lab.T @ e_lab + TIK.T@TIK) @ e_lab.T        
        except:
            A = np.linalg.pinv(e_lab.T @ e_lab + TIK.T@TIK) @ e_lab.T        
        F = np.zeros(shape=Y.shape)
        
        y_m =  np.argmax(Y, axis=1)[labeledIndexes]
        
        for i in range(p):
            if not hook is None:
                hook._step(step=i,X=X,W=W,Y=E[:,i])
        
        
        for i in range(Y.shape[1]):
            c = np.ones(num_lab)
            c[y_m != i] = -1
            a = A @ np.transpose(c)
            LOG.debug(a,LOG.ll.CLASSIFIER)
            for j in np.arange(F.shape[0]):
                F[j,i] = np.dot(a,E[j,:])
                F[j,i] = max(F[j,i],0)

        return (F)
        
        

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__MR(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,p=self.p,hook=hook))


    def __init__(self,p=0.2):
        """ Constructor for LapEigLS classifier.
            
        Args:
            p (Union[float,int]). The number of eigenfunctions. It is given as either the absolute value if integer, or a percentage of
                the labeled data. if float. Default is ``0.2``
        """
        self.p = p