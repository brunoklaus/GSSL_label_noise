"""
settings.py
=======================
Auxiliary Module that stores the relevant folder paths.
@author klaus
"""
import os
import os.path as path
import numpy as np
from progressbar import ProgressBar,Percentage,Bar,ETA,Timer

def __calculate_root_path():
    root_path = os.path.abspath(__file__)
    i = 0
    while not root_path.endswith("src"):
        root_path = os.path.dirname(root_path)
        i += 1
        if len(root_path) < len("src") or i > 100:
            raise FileNotFoundError("Could not go up till a directory named EEF_2019 was found")
    return root_path

#: The path to the root of this python project
ROOT_FOLDER = __calculate_root_path()
INPUT_FOLDER = path.join(ROOT_FOLDER,"input")
CIFAR_MAT_FOLDER = path.join(INPUT_FOLDER,"dataset","cifar10_matrices")
CIFAR_FOLDER = path.join(INPUT_FOLDER,"dataset","cifar10_data")

import scipy.sparse

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    
def p_bar(max_value, title_text):
    widget_kwargs= {"max_value":max_value,"min_value":0}
    
    default_w = [title_text,' ', Percentage(), ' ', Bar(), ' ', ETA(),' ',Timer()]
            

    return ProgressBar(max_value=max_value,widgets= default_w,len_func=lambda x: len(x)*100)
