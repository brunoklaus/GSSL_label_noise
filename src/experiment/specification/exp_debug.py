'''
Created on 27 de mar de 2019

@author: klaus
'''
from experiment.specification.specification_skeleton import EmptySpecification
import experiment.specification.specification_bits as spec
from experiment.specification.specification_bits import allPermutations as P

import numpy as np
class ExpDebug(EmptySpecification):
    '''
    classdocs
    '''

   
    def get_spec_name(self):
        return "exp_debug"
    
       
    def generalConfig(self):
        s = spec.GENERAL_DEFAULT
        s["id"] = [2]
        return P(s)
    
    def inputConfig(self):
        s = spec.INPUT_SPIRALS_DYNAMIC
        return P(s)


    def filterConfig(self):
        return  P(spec.FILTER_LGC_LVO)
        
    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_MODERATE
        s["corruption_level"]=[0.2]
        return P(s)
    
    def affmatConfig(self):
        s = spec.AFFMAT_DEFAULT
        return P(s)
    def algConfig(self):
        s = spec.ALGORITHM_LGC_DEFAULT
        

        return P(s)
