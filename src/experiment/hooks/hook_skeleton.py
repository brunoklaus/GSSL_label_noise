import output.plots as plots
import os.path as path
import os
from inspect import signature, Parameter
from functools import partial
import numpy as np
import shutil
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import log.logger as LOG
class GSSLHook():
    """ Skeleton for GSSL hooks (for plotting, for example). """
    
    def get_step_rate(self):
        """ Specifies how many steps it takes to call the "step" procedure on this hook."""
        return 1
    
    def _begin(self, **kwargs):
        """ Procedure to be called just before GSSL algorithm is called. """
        raise NotImplementedError()
    def _step(self,step, **kwargs):
        """ Procedure to be called at every step for GSSL algorithms that work iteratively. """
        raise NotImplementedError()
    def _end(self, **kwargs):
        raise NotImplementedError()
    
    
    
class CompositeHook():
    """ A hook that calls other hooks. """
    
    def _begin(self, **kwargs):
        for x in self.hook_list:
            x._begin(**kwargs)

    def _step(self,step, **kwargs):
        for x in self.hook_list:
            x._step(step,**kwargs)
    def _end(self, **kwargs):
        for x in self.hook_list:
            x._end(**kwargs)
    
    def __init__(self, hook_list):
        """ Constructor for CompositeHook.
        
        Args:
            hook_list (List[GSSLHook]) : The list of hooks forming this CompositeHook.
            
        """
        self.hook_list = hook_list
        self.step_rate = 1

def _remove_excess_vars(f,kwargs):
    if isinstance(f, partial):
        f = f.func
    
    f_vars = f.__code__.co_varnames[0:f.__code__.co_argcount + f.__code__.co_kwonlyargcount:]
    #print("FUNCTION VARS:{}".format(f_vars))
    
    to_be_rm = []
    for k in kwargs.keys():
        if not k in f_vars:
            to_be_rm.append(k)
    for x in to_be_rm:
        kwargs.pop(x)
    #print("REMOVED VARS:{}".format(to_be_rm))
    #print("KEPT VARS:{}".format(kwargs))
    
    return kwargs

def _add_remaining_vars(f,kwargs,experiment):
    if isinstance(f, partial):
        f = f.func
    f_vars = [k for k,v in signature(f).parameters.items()]
    
    LOG.debug("FUNCTION NECESSARY VARS:{}".format(f_vars),LOG.ll.HOOK)
    
    for k in f_vars:
        if not k in kwargs.keys():
            kwargs[k] = getattr(experiment, k)
    return kwargs
    
        
            