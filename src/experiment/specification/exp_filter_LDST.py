'''
Created on 3 de abr de 2019

@author: klaus

'''

from experiment.specification.specification_skeleton import EmptySpecification
from experiment.experiments import Experiment, TIME_HOOKS
import experiment.specification.specification_bits as spec
from experiment.prefixes import FILTER_PREFIX

import numpy as np
from experiment.hooks.hook_skeleton import GSSLHook
from experiment.hooks.ldst_filterstats_hook import LDSTFilterStatsHook
from experiment.selector import Hook

from experiment.specification.specification_bits import allPermutations as P,\
    AFFMAT_DEFAULT
import log.logger as LOG

class FilterLDST(EmptySpecification):
    '''
    classdocs
    '''


    WRITE_FREQ = 100

    def get_spec_name(self):
        return "31Dez_LDST3_mod_chap"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        s["labeled_percent"] = [0.1,0.01]
        #s = spec.INPUT_SPIRALS_DYNAMIC
        #s["dataset"] = ["COIL2"]
        #s["labeled_percent"] = [0.1]
        #s = spec.INPUT_MNIST
        #s["labeled_percent"] = [50/70000]
        
        return P(s)


    def filterConfig(self):
        return  P(spec.FILTER_LDST) + P(spec.FILTER_LGC_LVO)
        
    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_MODERATE
        s["corruption_level"]=[0.35]
        return P(s)
    
    def affmatConfig(self):
        return P(spec.AFFMAT_DEFAULT)
    def algConfig(self):
        s = spec.ALGORITHM_NONE
        #s["num_iter"] = [10000]
        #s["alpha"] = [0.99]
        #s["mu"] = [99]
        #s["p"] = [2]
        return P(s)
    
    def run(self,cfg):
        res = Experiment(cfg).run(hook_list=[Hook.LDST_STATS_HOOK])
        res.update(cfg)
        
        stat_dict = res.pop("flt_stat_dict")
        L = []
        
        
        
        for k,v in stat_dict.items():
            dct = dict(res)
            dct[FILTER_PREFIX + "tuning_iter"] = int(k)
            dct.update(v)
            L.append(dct)
        return L
    

class MNIST(EmptySpecification):
    '''
    classdocs
    '''


    WRITE_FREQ = 100

    def get_spec_name(self):
        return "30Dez_LGCLVO_sens_mnist"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
    def inputConfig(self):
        #s = spec.INPUT_CHAPELLE_A
        #s = spec.INPUT_SPIRALS_DYNAMIC
        #s["dataset"] = ["Digit1","COIL2"]
        #s["labeled_percent"] = [0.1,0.01]
        s = spec.INPUT_MNIST
        s["labeled_percent"] = [100/70000]
        return P(s)


    def filterConfig(self):
        return    P({
        "filter": ["LGC_LVO"],
        "mu": [0.1111],
        "constantProp" : [False],
        "early_stop" : [True],
        "useZ" : [True],
        "normalize_rows" : [True],
        "use_baseline": [False],
        "tuning_iter": [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4],
        "tuning_iter_as_pct": [True],
        
        "relabel":[False],
        })
        
    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_SOME
        s["corruption_level"] = [0.3]
        return P(s)
    
    def affmatConfig(self):
        return P(spec.AFFMAT_CONSTANT)
    def algConfig(self):
        #s = spec.ALGORITHM_GTAM_DEFAULT
        #s["num_iter"] = [10000]
        #s["alpha"] = [0.9]
        #s["constantProp"] = [False]
        #s["mu"] = [0.1111]
        s = spec.ALGORITHM_SIIS_DEFAULT
        s["alpha"] = [1000.0]
        s["beta"] = [10.0]
        s["m"] = [200]
        s["max_iter"] = [500]
        s = spec.ALGORITHM_LGC_DEFAULT
        #s["num_iter"] = [1000]
        #s["alpha"] = [0.9]
        s["num_iter"] = [10000]
        #s["p"] = [2]
        return P(s)
    
    
class ISOLET(EmptySpecification):
    '''
    classdocs
    '''


    WRITE_FREQ = 100

    def get_spec_name(self):
        return "30Dez_SIIS_isolet"
    
    def generalConfig(self):
        return P(spec.GENERAL_DEFAULT)
    
   
    def inputConfig(self):
        return P(spec.INPUT_ISOLET)
   
    def filterConfig(self):
        return   P(spec.FILTER_NOFILTER)
        """"P({
        "filter": ["LGC_LVO"],
        "mu": [0.1111],
        "constantProp" : [False],
        "early_stop" : [True],
        "useZ" : [True],
        "normalize_rows" : [True],
        "use_baseline": [False],
        "tuning_iter": [1.2,1.4],
        "tuning_iter_as_pct": [True],
        
        "relabel":[False],
        })"""
        
    

    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_SOME
        s["corruption_level"] = [0.0,0.2,0.4,0.6]
        return P(s)
    
    def affmatConfig(self):
        s = spec.AFFMAT_ISOLET
        return P(s)
    def algConfig(self):
        #s = spec.ALGORITHM_GTAM_DEFAULT
        #s["num_iter"] = [10000]
        #s["alpha"] = [0.9]
        #s["constantProp"] = [False]
        #s["mu"] = [0.1111]
        s = spec.ALGORITHM_SIIS_DEFAULT
        s["alpha"] = [1000.0]
        s["beta"] = [10.0]
        s["m"] = [300]
        s["max_iter"] = [500]
        #s = spec.ALGORITHM_LGC_DEFAULT
        #s["num_iter"] = [1000]
        #s["alpha"] = [0.9]
        #s["num_iter"] = [10000]
        #s["p"] = [2]
        return P(s)
    
        