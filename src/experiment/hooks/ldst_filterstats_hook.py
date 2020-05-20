'''
Created on 3 de abr de 2019

@author: klaus
'''
import experiment.hooks.hook_skeleton as hk
import gssl.filters.filter_utils as futils
import log.logger as LOG

class LDSTFilterStatsHook(hk.GSSLHook):
    """ Hook that saves stats  of the filter at each step."""
    
    best_f1 = -1.0
    def _collect_stats(self,step,**kwargs):
        A = futils.get_unlabeling_confmat(Y_true=self.xp.Y_true,
                                          Y_n = self.xp.Y_noisy,
                                          Y_f = kwargs["Y"],
                                          lb_n = self.xp.labeledIndexes,
                                          lb_f = kwargs["labeledIndexes"])
        self.xp.out_dict["flt_stat_dict"][str(step)] = futils.get_confmat_dict(A)
        self.best_f1 = max(self.xp.out_dict["flt_stat_dict"][str(step)]["out_filter_f1_score"],self.best_f1)
        
    def _begin(self, **kwargs):
        self.xp.out_dict["flt_stat_dict"] = {}
        
    def _step(self,step, **kwargs):
        if step % self.step_size != 0:
            return
        else:
            self.last_step = step
            self._collect_stats(step,**kwargs)
        
    def _end(self, **kwargs):
        LOG.info("Best F1 score: {}".format(self.best_f1),LOG.ll.HOOK)
        pass
            
    def __init__(self,experiment,step_size=5):
        """ Constructor for LDSTFilterStatsHook.
        
        Args:
             experiment (:class:`experiment.experiments.Experiment`): The Experiment object.
                 The experiment's ``output_dict`` will be modified.
             step_size (int) : Determines after how many iterations to save the filter stats.
        """
        self.xp = experiment
        self.step_size = step_size

        