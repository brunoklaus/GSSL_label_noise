'''
Created on 28 de mar de 2019

@author: klaus
'''
import input.noise.noise_utils as nutils

class LabelNoiseProcess(object):
    """ A noise process, which is used to corrupt the original, clean labels.
    
    Currently, the following parameters are expected:
    
    * type : Specifies the type of noise. Currently, this may be
        
            1. ``"NCAR"``: Noisy Completely at Random. The occurrence of error is independent of other random variables,
               including the true labels. The chance of a label being flipped to a given, different class is :math:``\\frac{p}{C-1}``.
               Requires specification of parameter ``noise_corruption``, which determines :math:`p`.
    
        
    * seed : Specifies the seed for reproducibility purposes.
    * deterministic : If `True`, the distribution of actual flips tries to reflect the scaled matrix of transition probabilities. 
      This means that, while the chosen labels might differ from run to run, the number of flips is essentially fixed,
      we have a fixed quantity of 'corrupted labels'.
    
    
    """
    
    def corrupt(self,Y, labeledIndexes,hook=None):
        """ Corrupts a set of clean labels, obtaining the corrupted labels.
            
            Args:
                Y (`NDArray[float].shape[N,C]`) : Target belief matrix, whose rows are one-hot selecting the correct label.
                labeledIndexes (`NDArray[bool].shape[N]`): A boolean array, indicating which instances are to be interpreted as labeled.
                hook (`GSSLHook`): Optional. A hook to be called before/during/after the noise process
            Returns:
                (`NDArray[float].shape[N,C]`) : Corrupted version of the target belief matrix
        
        """
        
        if not hook is None:
            print(hook)
            hook._begin(Y=Y,labeledIndexes=labeledIndexes)
            
        if self.args["type"] == "NCAR":
            A =  nutils.uniform_noise_transition_prob_mat(Y, self.args["corruption_level"])
            
        else:
            raise NotImplementedError()
        
        Y_noisy =  nutils.apply_noise(Y, labeledIndexes, A, self.args["seed"],deterministic= self.args["deterministic"])
            
        if not hook is None:
            hook._end(Y=Y_noisy,labeledIndexes=labeledIndexes,A=A)
            
        return Y_noisy
    

    def __init__(self, **kwargs):
        """ Constructor for LabelNoiseProcess.
        
        Args: 
            `**kwargs`: Key-value pairs with the configuration options.
       
        Raises:
            KeyError: If one of the required keys is not found.
        """
        for x in ["type","seed","deterministic"]:
            if not x in kwargs.keys():
                raise KeyError("Key " + x + " not found")
        
        if kwargs["type"] == "NCAR":  
            if not "corruption_level" in kwargs.keys():
                raise KeyError("Key noise_corruption not found")
        else:
            raise ValueError("Did not find noise process of type {}".format(kwargs["type"]))
        
        self.args = kwargs
