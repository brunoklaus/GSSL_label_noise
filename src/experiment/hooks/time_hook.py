'''
Created on 2 de abr de 2019

@author: klaus
'''
import experiment.hooks.hook_skeleton as hk
import time
from timeit import default_timer as timer
import numpy as np


class timeHook(hk.GSSLHook):
    """ Hook that times some part of the experiment execution.  """
        
    def _begin(self, **kwargs):
        self.experiment.out_dict[self.timer_name] = np.nan
        self.begin_time = timer()

    def _step(self,step, **kwargs):
        pass
    def _end(self, **kwargs):
        self.end_time = timer()
        self.experiment.out_dict[self.timer_name] = np.round(1000.0*(self.end_time - self.begin_time),decimals=3)
        
            
    def __init__(self,experiment,timer_name):
        """ Constructor for the timeHook.
        
        Args:
             experiment (:class:`experiment.experiments.Experiment`): The Experiment object.
                 The experiment's ``output_dict`` will be modified.
             timer_name (str) : The name of the timer. This name will be a key added to ``experiment.output_dict``. Initially the 
                 corresponding value will be ``np.nan``, but at the end it will be updated to be the time taken between `_begin` and
                 `_end` calls.
        """
        self.experiment = experiment
        self.timer_name = timer_name