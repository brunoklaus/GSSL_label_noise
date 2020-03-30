
    
    
class GSSLClassifier(object):
    """ Skeleton class for GSSL Classifiers. """
    
    @classmethod
    def autohooks(cls, fun):
        """ Automatically calls the begin and end method of the hook. The classifier result is passed as 
        the 'Y' argument at the end."""
        
        def wrapper(self, *args, **kwargs):
            dct = dict(zip(fun.__code__.co_varnames[1:(len(args)+1)],args))
            slf = self
            print(dct)
            
            kwargs.update(dct)
            hook = kwargs["hook"]
            
            if not hook is None:
                hook._begin(**kwargs)         
            
            kwargs["self"] = slf
            F = fun(**kwargs)
            
            kwargs["Y"] = F
            if not hook is None:
                kwargs.pop("self")
                hook._end(**kwargs)   
            return F
        return wrapper
    
    
    def fit (self,X,W,Y,labeledIndexes, hook=None):
        """ Classifies the input data.
        
        Args:
            X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
            W (`NDArray[float].shape[N,N]`): The affinity matrix encoding the weighted edges.
            Y (`NDArray[float].shape[N,C]`): The initial belief matrix
            hook (GSSLHook): Optional. A hook to execute extra operations (e.g. plots) during the algorithm
        
        Returns:
            `NDArray[float].shape[N,C]`: An updated belief matrix.
        """
        if not hook is None:
            hook._begin(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)
            hook._end(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)         
        return Y



    

