'''
Created on 22 de out de 2019

@author: klaus
'''

from experiment.specification.specification_skeleton import EmptySpecification
import experiment.specification.specification_bits as spec
from experiment.specification.specification_bits import allPermutations as P,\
    AFFMAT_DEFAULT

import numpy as np
class ExpCIFAR(EmptySpecification):
    '''
    classdocs
    '''

    def get_spec_name(self):
        return "cifar_nl=4000_supervised"
    
    def generalConfig(self):
        s = spec.GENERAL_DEFAULT
        s["id"] = np.arange(10)
        return P(s)

    def inputConfig(self):
        s = spec.INPUT_CIFAR_10
        s["labeled_percent"] = [4000/60000]
        return P(s)

    def filterConfig(self):
        s = spec.FILTER_NOFILTER
        #s["p"] = [30]
        return P(s)
    
    def noiseConfig(self):
        return P(spec.NOISE_UNIFORM_DET_MODERATE)
    
    def affmatConfig(self):
        s = spec.AFFMAT_CIFAR10_LOAD
        return P(s)
    def algConfig(self):
        s = spec.ALGORITHM_NN_CIFAR10
        s["NUM_EPOCHS"] = [150]
        s["BATCH_SIZE"] = [100]
        s["model_choice"] = ["conv_large"]
        #s["num_iter"] = [10]
        #s["alpha"] = [0.01]
        #s["mu"] = [99]
        #s["p"] = [2]
        return P(s)
    