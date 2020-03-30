
import numpy as np
import gssl.graph.gssl_utils as gutils
from gssl.classifiers.classifier import GSSLClassifier
from sklearn.ensemble import RandomForestClassifier
from gssl.graph.gssl_utils import init_matrix

class RFClassifier(GSSLClassifier):
    """ Supervised Random Forest classifier
    """
    @GSSLClassifier.autohooks
    def __RF(self,X,W,Y,labeledIndexes,n_estimators, hook=None):
        XL = X[labeledIndexes,:]
        print("RF")
        rf = RandomForestClassifier(n_estimators=n_estimators,verbose=2)
        rf.fit(X[labeledIndexes,:],np.argmax(Y[labeledIndexes,:],axis=1) )
        pred = rf.predict(X)
        
        return init_matrix(pred, np.ones(X.shape[0],).astype(np.bool))   
    

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__RF(X,W, Y, labeledIndexes, self.n_estimators, hook))
    
    
    def __init__(self,n_estimators=10):
        """ Constructor for the LGC classifier.
            
            Args:
                alpha (float): A value between 0 and 1 (not inclusive) for alpha.
        """
        self.n_estimators = n_estimators
