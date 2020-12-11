'''
Created on 27 de mar de 2019

@author: klaus
'''
import numpy as np
from experiment.prefixes import *
from experiment.selector import select_input, select_affmat, select_classifier, select_noise,\
    select_filter
import gssl.graph.gssl_utils as gutils
from experiment.selector import Hook, select_and_add_hook
import log.logger as LOG
from gssl.filters import LGCLVO_NEW

## The hooks being utilized
PLOT_HOOKS = [Hook.INIT_LABELED,Hook.INIT_ALL,Hook.NOISE_AFTER,Hook.ALG_RESULT,Hook.ALG_ITER] \
                #+ [Hook.GTAM_Q,Hook.GTAM_F,Hook.GTAM_Y]
W_PLOT_HOOKS = [Hook.W_INIT_LABELED,Hook.W_INIT_ALL,Hook.W_NOISE_AFTER,Hook.FILTER_ITER,Hook.W_FILTER_AFTER,Hook.ALG_RESULT,Hook.ALG_ITER] \
                #+ [Hook.GTAM_Q,Hook.GTAM_F,Hook.GTAM_Y]

W_PLOT_HOOKS_NOITER = list(W_PLOT_HOOKS)
W_PLOT_HOOKS_NOITER.remove(Hook.ALG_ITER)
W_PLOT_HOOKS_NOITER.remove(Hook.FILTER_ITER)

PLOT_HOOKS_NOITER = list(PLOT_HOOKS)
PLOT_HOOKS_NOITER.remove(Hook.ALG_ITER)


TIME_HOOKS = [Hook.T_ALG,Hook.T_FILTER,Hook.T_NOISE,Hook.T_AFFMAT]                

_Z =[("AFFMAT",AFFMAT_PREFIX),
                ("INPUT",INPUT_PREFIX),
                ("FILTER",FILTER_PREFIX),
                ("NOISE",NOISE_PREFIX),
                ("ALG",ALG_PREFIX),
                ("GENERAL","")\
                ]


def keys_multiplex(args):

    mplex = {}
    for x,y in _Z:
        mplex[x] = {}
    
    
    for k,v in args.items():
        for x,y in _Z:
            if k.startswith(y):
                mplex[x][k[len(y):]] = v
                break
    return mplex



def postprocess(mplex):
    """ Performs some postprocessing on the multiplexed keys. """
    mplex = dict.copy(mplex)
    
    id = mplex["GENERAL"]["id"]
    for k in mplex.keys():
        if k == "ALG" or k == "FILTER":
            continue
        mplex[k]["seed"] = id
        
    if  "sigma" in mplex["AFFMAT"].keys() and mplex["AFFMAT"]["sigma"] == "mean":
        """ Just use the RBF's SIGMA directly instead of computing it manually. """
        
        if mplex["INPUT"]["dataset"] == "Digit1":
            mplex["AFFMAT"]["sigma"]  = 4.412518742145814
        elif mplex["INPUT"]["dataset"] == "COIL2":
            mplex["AFFMAT"]["sigma"]  =  2.4462853134304963
        elif mplex["INPUT"]["dataset"] == "COIL":
            mplex["AFFMAT"]["sigma"]  =  3.0904129359360937

        elif mplex["INPUT"]["dataset"] == "isolet":
            mplex["AFFMAT"]["sigma"]  =  2.7529673535028003
        elif mplex["INPUT"]["dataset"] == "g241c":
            mplex["AFFMAT"]["sigma"]  =  6.593260880190159
        elif mplex["INPUT"]["dataset"] == "g241n":
            mplex["AFFMAT"]["sigma"]  =  6.528820377801142
        elif mplex["INPUT"]["dataset"] == "USPS":
            mplex["AFFMAT"]["sigma"]  =  4.412518742145814
        elif mplex["INPUT"]["dataset"] == "cifar10":
            mplex["AFFMAT"]["sigma"]  =  903.6705243483848
        elif mplex["INPUT"]["dataset"] == "mnist":
            mplex["AFFMAT"]["sigma"]  =  423.5704955059233
        
        
    if "tuning_iter_as_pct" in mplex["FILTER"].keys() and mplex["FILTER"]["tuning_iter_as_pct"]:
        mplex["FILTER"]["tuning_iter"] = mplex["NOISE"]["corruption_level"] *\
                                         mplex["FILTER"]["tuning_iter"]
        
    
    return mplex

class Experiment():
    """ Encapsulates an experiment, composed of the following steps.
         
         1. Reading the input features and true labels.
         2. Apply some noise process to the true labels, obtaining the corrupted labels.
         3. Create the Affinity matrix from the input features (and, optionally, noisy labels).
         4. Apply some filter to the corrupted labels, obtaining filtered labels.
         5. Run an GSSL algorithm to obtain the classification
         6. Get performance measures from the classification and filtered labels.
         
         
     Attributes:
        X (NDArray[float].shape[N,D]): The calculated input matrix
        W (NDArray[float].shape[N,N]): The affinity matrix encoding the graph.
        Y (NDArray[float].shape[N,C]): Initial label matrix
    """
    
    

    
    def __init__(self,args):
        self.args = dict(args)
        self.X = None
        self.labeledIndexes = None
        self.labeledIndexes_filtered = None
        
        self.Y_true = None
        self.Y_noisy = None
        self.Y_filtered = None
        self.W = None
        self.F = None
        self.out_dict = {}

    def run(self,hook_list=PLOT_HOOKS):
        for k,v in self.args.items():
            LOG.debug("{}:{}".format(k,v),LOG.ll.EXPERIMENT)
        
        
        #Multiplex the arguments, allocating each to the correct step
        mplex = postprocess(keys_multiplex(self.args))
        
        
        #Get Hooks:
        hooks = select_and_add_hook(hook_list, mplex, self)
        
        
        LOG.info("Step 1: Read Dataset",LOG.ll.EXPERIMENT)
        
        #Select Input 
        self.X, self.Y_true, self.labeledIndexes = select_input(**mplex["INPUT"])

        
        
        
        if "know_estimated_freq" in mplex["ALG"].keys():
            mplex["ALG"]["useEstimatedFreq"] = np.sum(self.Y_true,axis=0) / self.Y_true.shape[0]
            mplex["ALG"].pop("know_estimated_freq")
        
        if "know_estimated_freq" in mplex["FILTER"].keys():
            mplex["FILTER"]["useEstimatedFreq"] = np.sum(self.Y_true,axis=0) / self.Y_true.shape[0]
            mplex["FILTER"].pop("know_estimated_freq")
            
        
        
        
        
        LOG.info("Step 2: Apply Noise",LOG.ll.EXPERIMENT)
        #Apply Noise
        self.Y_noisy = select_noise(**mplex["NOISE"]).corrupt(self.Y_true, self.labeledIndexes,hook=hooks["NOISE"])
        



        
        LOG.info("Step 3: Create Affinity Matrix",LOG.ll.EXPERIMENT)
        #Generate Affinity Matrix
        self.W = select_affmat(**mplex["AFFMAT"]).generateAffMat(self.X,hook=hooks["AFFMAT"])
        
        
        
        LOG.info("Step 4: Filtering",LOG.ll.EXPERIMENT)
        #Create Filter
        ft = select_filter(**mplex["FILTER"])
        self.ft = ft

        
        noisyIndexes = (np.argmax(self.Y_true,axis=1) != np.argmax(self.Y_noisy,axis=1))
        
        self.Y_filtered, self.labeledIndexes_filtered = ft.fit(self.X, self.Y_noisy, self.labeledIndexes, self.W, hook=hooks["FILTER"])


        LOG.info("Step 5: Classification",LOG.ll.EXPERIMENT)
        #Select Classifier 
        alg = select_classifier(**mplex["ALG"])
        #Get Classification
        self.F = alg.fit(self.X,self.W,self.Y_filtered,self.labeledIndexes_filtered,hook=hooks["ALG"])
        
        if isinstance(ft, LGCLVO_NEW.LGC_LVO_AUTO_Filter):
            if not ft.loss is None:
                self.F[self.labeledIndexes,:] = ft.Fl
        

        
        LOG.info("Step 6: Evaluation",LOG.ll.EXPERIMENT)
        LOG.debug("ALGORITHM settings:{}".format(mplex["ALG"]["algorithm"]),LOG.ll.EXPERIMENT)
        
        """ Accuracy. """
        acc = gutils.accuracy(gutils.get_pred(self.F), gutils.get_pred(self.Y_true))
        
        
        acc_unlabeled = gutils.accuracy(gutils.get_pred(self.F)[np.logical_not(self.labeledIndexes)],\
                                         gutils.get_pred(self.Y_true)[np.logical_not(self.labeledIndexes)])
        acc_labeled = gutils.accuracy(gutils.get_pred(self.F)[self.labeledIndexes],\
                                         gutils.get_pred(self.Y_true)[self.labeledIndexes])
        
        CMN_acc = gutils.accuracy(gutils.get_pred(gutils.class_mass_normalization(self.F,self.Y_filtered,self.labeledIndexes,normalize_rows=True)), gutils.get_pred(self.Y_true))
        
        """ Accuracy with Class Mass Normalization. """
        CMN_rownorm_pred =gutils.get_pred(gutils.class_mass_normalization(self.F,self.Y_filtered,self.labeledIndexes,normalize_rows=False))
        
        CMN_rownorm_acc = gutils.accuracy(CMN_rownorm_pred, gutils.get_pred(self.Y_true))
        
        
        CMN_rownorm_acc_unl = gutils.accuracy(CMN_rownorm_pred[np.logical_not(self.labeledIndexes)],
                                              gutils.get_pred(self.Y_true)[np.logical_not(self.labeledIndexes)])
        
        CMN_rownorm_acc_l = gutils.accuracy(CMN_rownorm_pred[self.labeledIndexes],
                                              gutils.get_pred(self.Y_true)[self.labeledIndexes])
        
        
        """
            Log accuracy results and update output dictionary
        """
        def _log(msg):
            LOG.info(msg,LOG.ll.EXPERIMENT)
            
        _log("Accuracy: {:.3%} | {:.3%}".format(acc,1-acc))
        _log("Accuracy (unlabeled): {:.3%} |{:.3%}".format(acc_unlabeled,1-acc_unlabeled))
        _log("Accuracy (labeled): {:.3%} | {:.3%}".format(acc_labeled,1-acc_labeled))    
        _log("Accuracy w/ CMN: {:.3%} | {:.3%}".format(CMN_acc,1-CMN_acc))
        _log("Accuracy w/ rownorm CMN: {:.3%} | {:.3%}".format(CMN_rownorm_acc, 1-CMN_rownorm_acc))
        _log("Accuracy w/ rownorm CMN(unlabeled): {:.3%} | {:.3%}".format(CMN_rownorm_acc_unl, 1-CMN_rownorm_acc_unl))
        
        self.out_dict.update({OUTPUT_PREFIX + "acc" :acc})
        self.out_dict.update({OUTPUT_PREFIX + "acc_unlabeled" :acc_unlabeled})
        self.out_dict.update({OUTPUT_PREFIX + "acc_labeled" :acc_labeled})
        self.out_dict.update({OUTPUT_PREFIX + "CMN_rownorm_acc" :CMN_rownorm_acc})
        self.out_dict.update({OUTPUT_PREFIX + "CMN_rownorm_acc_unl" :CMN_rownorm_acc_unl})
        self.out_dict.update({OUTPUT_PREFIX + "CMN_acc" :CMN_acc})
        
        return self.out_dict
    
    

def run_debug_example_one(hook_list=[]):
    import experiment.specification.exp_chapelle as exp
    
    opt = exp.ExpChapelle("Digit1").get_all_configs()[0]
    
    Experiment(opt).run(hook_list=hook_list)
    
    
def run_debug_example_all():
    import experiment.specification.exp_chapelle as exp
    exp.ExpChapelle("").run_all()
    
def main():
    run_debug_example_one()

    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()

    
    
    
