import numpy as np
import gssl.graph.gssl_utils as gutils
from gssl.classifiers.classifier import GSSLClassifier


class LGCClassifier(GSSLClassifier):
    """ Local and Global Consistency Classifier. See :cite:`Zhou_etall_2004`.
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
    def __LGC(self,X,W,Y,labeledIndexes, alpha = 0.1, hook=None):
        import scipy.sparse
        if scipy.sparse.issparse(W):
            W = W.todense()
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        #Get D^{-1/2}
        d_sqrt = gutils.deg_matrix(W, pwr=-1/2)
        
        
        I = np.identity(Y.shape[0])
        S = I - gutils.lap_matrix(W, is_normalized=True)
        
        
        return(np.matmul(np.linalg.inv(I - alpha*S),Y))
    
    @GSSLClassifier.autohooks
    def __LGC_iter(self,X,W,Y,labeledIndexes, alpha = 0.1,num_iter = 1000, hook=None):
        from scipy import sparse
        from sklearn.preprocessing import normalize
        W = sparse.csr_matrix(W)
        
        Y = np.copy(Y)
        
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        Y = sparse.csr_matrix(Y)
        
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        #Get D^{-1/2}
        wsum = np.reshape(np.asarray(W.sum(axis=0)),(-1,) ) 
        d_sqrt = np.reciprocal(np.sqrt(wsum))
        d_sqrt[np.logical_not(np.isfinite(d_sqrt))] = 1
        d_sqrt = sparse.diags(d_sqrt).tocsr()
        
        
        
        F = sparse.csr_matrix.copy(Y)
        S = d_sqrt*W*d_sqrt

        for i in range(num_iter):
            F = self.alpha *(S @ F) + (1 - self.alpha)*Y
            if not hook is None:
                F_dense = np.asarray(F.todense())
                labeledIndexes = np.sum(F_dense,axis=1) > 0
                hook._step(step=i,X=X,W=W,Y=F_dense,labeledIndexes=labeledIndexes) 
            
                        

        F_dense = np.asarray(F.todense())
        return(F_dense)
    
    @GSSLClassifier.autohooks
    def __LGC_iter_TF(self,X,W,Y,labeledIndexes, alpha = 0.1,num_iter = 1000, hook=None):
        from gssl.classifiers.LGC_tf import LGC_iter_TF

        return LGC_iter_TF(X, W, Y, labeledIndexes, alpha, num_iter, hook)
    

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        if self.num_iter is None:
            return(self.__LGC(X,W, Y, labeledIndexes, self.alpha, hook))
        else:
            return(self.__LGC_iter_TF(X,W, Y, labeledIndexes, self.alpha, self.num_iter, hook))
    
    def __init__(self, alpha = 0.1, num_iter = None):
        """ Constructor for the LGC classifier.
            
            Args:
                alpha (float): A value between 0 and 1 (not inclusive) for alpha.
                num_iter (float): Optional. If not ``Ǹone``, the number of iteration updates to perform
        """
        self.alpha = alpha
        self.num_iter = num_iter
