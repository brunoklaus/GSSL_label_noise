'''
Created on 27 de mar de 2019

@author: klaus
'''
from experiment.experiments import AFFMAT_PREFIX,ALG_PREFIX,NOISE_PREFIX,INPUT_PREFIX,FILTER_PREFIX,\
    Experiment
from functools import reduce
import numpy as np
import pandas as pd
import os
import sys 
import experiment.specification.specification_bits as spec

from experiment.experiments import TIME_HOOKS
from output.folders import CSV_FOLDER
import progressbar
from output.aggregate_csv import aggregate_csv

import traceback
from functools import wraps
from multiprocessing import Process, Queue
def processify(func):
        '''
        From https://gist.github.com/schlamar/2311116
        Decorator to run a function as a process.
        Be sure that every argument and the return value
        is *pickable*.
        The created process is joined, so the code does not
        run in parallel.
        '''
    
        def process_func(q, *args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except Exception:
                ex_type, ex_value, tb = sys.exc_info()
                error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
                ret = None
            else:
                error = None
    
            q.put((ret, error))
    
        # register original function with different name
        # in sys.modules so it is pickable
        process_func.__name__ = func.__name__ + 'processify_func'
        setattr(sys.modules[__name__], process_func.__name__, process_func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            q = Queue()
            p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
            p.start()
            ret, error = q.get()
            p.join()
    
            if error:
                ex_type, ex_value, tb_str = error
                message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
                raise ex_type(message)
    
            return ret
        return wrapper
class EmptySpecification(object):
    """ EmptySpecification defines the methods expected from any class representing a 
    specification of experiments. By itself, it also specifies an empty set of experiments.
    """
    
    
    FORCE_GTAM_LDST_SAME_MU = True
    TUNING_ITER_AS_NOISE_PCT = False


    WRITE_FREQ = 10000
    DEBUG_MODE = True
    OVERWRITE = True
    
    def get_spec_name(self):
        """ Gets the name identifying the set of experiments that come out of this specification."""
        return ""
    
    def generalConfig(self):
        """General configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    def inputConfig(self):
        """Input configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    def noiseConfig(self):
        """Noise process configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    def filterConfig(self):
        """Filter configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    
    
    def affmatConfig(self):
        """Input configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    

    def algConfig(self):
        """Input configuration.
        
        Returns:
            `dict` A dictionary, such that each key maps to a list containing each possible value
            that the attribute might take.
        """
        return [{}]
    
    
    
    def get_all_configs(self):
        """Gets the configuration for every  experiment. The corresponding prefix is added for each stage.
            Returns:
            `List[dict]` A list of all possible configs. 
        """
        g = self.generalConfig()
        for x in g:
            x["spec_name"] = self.get_spec_name()
        Z = [(g,""),
             (self.inputConfig(),INPUT_PREFIX),
             (self.noiseConfig(),NOISE_PREFIX),
             (self.filterConfig(),FILTER_PREFIX),
             (self.affmatConfig(),AFFMAT_PREFIX),
             (self.algConfig(),ALG_PREFIX)\
             ]
        l = [[spec.add_key_prefix(y, elem) for elem in x] for x,y in Z]        
        res =  list(reduce(lambda x,y: spec.comb(x,y),l))
        
        if self.FORCE_GTAM_LDST_SAME_MU:
            old_len = len(res)
            res = [x for x in res if \
                   not (x[ALG_PREFIX+"algorithm"]=="GTAM" and \
                   x[FILTER_PREFIX+"filter"]=="LDST" and x[ALG_PREFIX+"mu"] != x[FILTER_PREFIX+"mu"])]
            
            res = [x for x in res if \
                   not (x[ALG_PREFIX+"algorithm"]=="LGC" and \
                   x[FILTER_PREFIX+"filter"]in["LDST","LGC_LVO"] and np.round((1-x[ALG_PREFIX+"alpha"])/x[ALG_PREFIX+"alpha"],4) != np.round(x[FILTER_PREFIX+"mu"],4))]
            
            print("REMOVED CONFIGS: {}".format(old_len-len(res)))
            
        if self.TUNING_ITER_AS_NOISE_PCT:
            for x in res:
                if not FILTER_PREFIX+ "tuning_iter" in x.keys():
                    pass
                else:
                    x[FILTER_PREFIX + "tuning_iter" ] = x[INPUT_PREFIX+"labeled_percent"] *\
                                         x[NOISE_PREFIX+"corruption_level"] *\
                                         x[FILTER_PREFIX+"tuning_iter"]
                    x[FILTER_PREFIX + "tuning_iter_as_pct"] = True
        return res
    

    
    @processify
    def run(self,cfg):
        exp = Experiment(cfg)
        res = exp.run(hook_list=TIME_HOOKS)
        del exp
        res.update(cfg)
        return [res]
       
    
    def _append_to_csv(self,output_dicts,result_path,f_mode,cfgs_keys):
        all_keys = set().union(*(d.keys() for d in output_dicts))
        
        for k in all_keys:
            cfgs_keys.add(k)
        
        df = pd.DataFrame(index=range(len(output_dicts)), columns=np.sort(list(cfgs_keys)))
        for i in range(len(output_dicts)):
            x = output_dicts[i]
            for k in x.keys():
                df[k].iloc[i] = x[k]

        with open(result_path, f_mode) as f:            
                is_header = (f_mode == "w")
                df.to_csv(f, header = is_header)
                
    def aggregate_csv(self):
         
        CSV_PATH = os.path.join(CSV_FOLDER, self.get_spec_name() + '.csv')
        JOINED_CSV_PATH  = os.path.join(CSV_FOLDER, self.get_spec_name() + '_joined.csv')
        
        aggregate_csv([CSV_PATH], JOINED_CSV_PATH)
    
    def run_all(self):
        
        CSV_PATH = os.path.join(CSV_FOLDER, self.get_spec_name() + '.csv')
        JOINED_CSV_PATH  = os.path.join(CSV_FOLDER, self.get_spec_name() + '_joined.csv')
        
        cfgs = self.get_all_configs()
        cfgs_keys = set()
        for x in cfgs:
            for k in x.keys():
                cfgs_keys.add(k)
        
        #List of produced output dicts
        output_dicts = list()
        
        cfgs_size = len(cfgs)
        
        has_written_already = False
        
        bar = progressbar.ProgressBar(max_value=cfgs_size)
        counter = 0
        bar.update(0)
        
        for i in range(cfgs_size):
            
            #Maybe suppress output
            nullwrite = open(os.devnull, 'w')   
            oldstdout = sys.stdout
            if not self.DEBUG_MODE:
                sys.stdout = nullwrite 
            
            output_dicts.extend(self.run(cfgs[i]))
            
            sys.stdout = oldstdout
            #Append to csv if conditions are met
            if i == cfgs_size-1 or i % self.WRITE_FREQ == 0:
                print("appending csv...")
                csv_exists = os.path.isfile(CSV_PATH)
                if self.OVERWRITE:
                    if csv_exists and has_written_already:
                        f_mode = 'a'
                    else:
                        f_mode = 'w'
                else:
                    if csv_exists:
                        f_mode = 'a'
                    else:
                        f_mode = 'w'
                print("f_mode={}".format(f_mode))
                self._append_to_csv(output_dicts,CSV_PATH,f_mode,cfgs_keys)
                has_written_already = True
                output_dicts.clear()
            
            bar.update(i+1)
        aggregate_csv([CSV_PATH], JOINED_CSV_PATH)


    
    
    
    
    
