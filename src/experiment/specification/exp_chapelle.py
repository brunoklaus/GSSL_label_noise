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
    ds = "g241c"
    def get_spec_name(self):
        return "31Dez_LDST_mod_" + self.ds
    
    def generalConfig(self):
        s = spec.GENERAL_DEFAULT
        s["id"] = [0]
        return P(s)
    
    def inputConfig(self):
        
        s = spec.INPUT_MNIST
        s['labeled_percent'] = [100/70000]
        #s['dataset'] = ['Digit1']
        #s['labeled_percent'] = [0.1]
        
        return P(s)

    def filterConfig(self):
        
        return  P(spec.FILTER_NOFILTER)
    
    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_SOME
        s["corruption_level"] = [0.0]
        return P(s)
    
    def affmatConfig(self):
        s = spec.AFFMAT_DEFAULT
        s['k'] = [15]
        return P(s)
    def algConfig(self):
        s = spec.ALGORITHM_LGC_DEFAULT
        s["alpha"] = [0.9]
        s["num_iter"] = [7000]
        #s = spec.ALGORITHM_CLGC_DEFAULT
        #s["useEstimatedFreq"] = [None]
        #s["num_iter"] = [1500]
        #s["alpha"] = [0.5]
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
        return P(spec.AFFMAT_DEFAULT)
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