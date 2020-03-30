'''
Created on 27 de mar de 2019

@author: klaus
'''
from experiment.specification.specification_skeleton import EmptySpecification
import experiment.specification.specification_bits as spec
from experiment.specification.specification_bits import allPermutations as P,\
    AFFMAT_DEFAULT

import numpy as np
class ExpChapelle(EmptySpecification):
    '''
    classdocs
    '''
    ds = "Digit1"
    def get_spec_name(self):
        return "31Dez_LDST_mod_" + self.ds
    
    def generalConfig(self):
        s = spec.GENERAL_DEFAULT
        s["id"] = np.arange(20)
        return P(s)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        
        s["dataset"] = [self.ds]
        return P(s)

    def filterConfig(self):
        
        return  P(spec.FILTER_NOFILTER)
    
    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_SOME
        return P(s)
    
    def affmatConfig(self):
        s = spec.AFFMAT_CONSTANT
        return P(s)
    def algConfig(self):
        #s = spec.ALGORITHM_LGC_DEFAULT
        
        s = spec.ALGORITHM_SIIS_DEFAULT
        s["alpha"] = [10.0,100.0,1000.0]
        s["beta"] = [1.0]
        s["m"] = [200]
        s["max_iter"] = [200]
        s= spec.ALGORITHM_LGC_DEFAULT
        s = spec.ALGORITHM_MANIFOLDREG_DEFAULT
        s["p"] = [1,2,3,4,5,6,7,10,15,20,100]
        s = spec.ALGORITHM_GFHF_DEFAULT
        s = spec.ALGORITHM_LGC_DEFAULT
        
        s["alpha"] = 1/(1+99.0)
        return P(s)

    def __init__(self,ds):
        self.ds = ds

class ExpChapelle_2(EmptySpecification):
    '''
    classdocs
    '''

    def get_spec_name(self):
        return "29Ag_exp4_MRF_Digit1_MR"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        s["dataset"] = ["Digit1"]
        return P(s)

    def filterConfig(self):
        s = spec.FILTER_MR
        s["p"] = [4]
        return P(s)
    
    def noiseConfig(self):
        return P(spec.NOISE_UNIFORM_DET_SOME)
    
    def affmatConfig(self):
        return P(spec.AFFMAT_CONSTANT)
    def algConfig(self):
        s = spec.ALGORITHM_MANIFOLDREG_DEFAULT
        s["p"] = [4]
        return P(s) 
    

class ExpChapelle_3(EmptySpecification):
    '''
    classdocs
    '''

    def get_spec_name(self):
        return "29Ag_exp4_MRF_USPS_MR"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        s["dataset"] = ["USPS"]
        return P(s)

    def filterConfig(self):
        s = spec.FILTER_MR
        s["p"] = [15]
        return P(s)
    
    def noiseConfig(self):
        return P(spec.NOISE_UNIFORM_DET_SOME)
    
    def affmatConfig(self):
        return P(spec.AFFMAT_CONSTANT)
    def algConfig(self):
        s = spec.ALGORITHM_GFHF_DEFAULT
        s["p"] = [15]
        return P(s) 
    
    
    


class ExpChapelle_4(EmptySpecification):
    '''
    classdocs
    '''

    def get_spec_name(self):
        return "29Ag_exp4_MRF_g241c_MR"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        s["dataset"] = ["g241c"]
        return P(s)

    def filterConfig(self):
        s = spec.FILTER_MR
        s["p"] = [1]
        return P(s)
    
    def noiseConfig(self):
        return P(spec.NOISE_UNIFORM_DET_SOME)
    
    def affmatConfig(self):
        return P(spec.AFFMAT_CONSTANT)
    def algConfig(self):
        s = spec.ALGORITHM_MANIFOLDREG_DEFAULT
        s["p"] = [1]
        return P(s) 
    


class ExpChapelle_5(EmptySpecification):
    '''
    classdocs
    '''

    def get_spec_name(self):
        return "29Ag_exp4_MRF_g241n_MR"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        s["dataset"] = ["g241n"]
        return P(s)

    def filterConfig(self):
        s = spec.FILTER_MR
        s["p"] = [4]
        return P(s)
    
    def noiseConfig(self):
        return P(spec.NOISE_UNIFORM_DET_SOME)
    
    def affmatConfig(self):
        return P(spec.AFFMAT_CONSTANT)
    def algConfig(self):
        s = spec.ALGORITHM_MANIFOLDREG_DEFAULT
        s["p"] = [4]
        return P(s) 