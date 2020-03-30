'''
filter_utils.py
====================================
Module containing utilities for GSSL filters.
'''

import numpy as np
from experiment.prefixes import OUTPUT_PREFIX
def get_unlabeling_confmat(Y_true,Y_n,Y_f,lb_n,lb_f):
    """
    Gets the confusion matrix related to the labels removed by the filter.
    
    More specifically, a matrix M is returned:
    
    ::

             
       M  =  [TN:#(clean labels NOT removed by filter)  FN:#(noisy labels NOT removed by filter)  ] 
             [FP:#(clean labels removed by filter)  TP:#(noisy labels removed by filter) ] 
    
    Args:
        Y_true (`NDArray[float].shape[N,C]`): A belief matrix encoding the true labels.
        Y_n (`NDArray[float].shape[N,C]`): A belief matrix encoding the noisy labels.
        Y_n (`NDArray[float].shape[N,C]`): A belief matrix encoding the filtered labels.
        lb_n(`NDArray[bool].shape[N]`)  : Indices of the noisy label matrix to be marked as labeled.
        lb_f(`NDArray[bool].shape[N]`)  : Indices of the filtered label matrix to be marked as labeled.
    
    Returns:
        `NDArray[float].shape[N,C]`: The matrix M.
    
    """
    is_noisy = np.argmax(Y_true,axis=1) != np.argmax(Y_n,axis=1)
    is_noisy[np.logical_not(lb_n)] = False
    
    is_clean = np.argmax(Y_true,axis=1) == np.argmax(Y_n,axis=1)
    is_clean[np.logical_not(lb_n)] = False
    
    
    b1 = np.logical_not(lb_f)
    b2 = np.logical_and(lb_f,np.argmax(Y_f,axis=1) != np.argmax(Y_n,axis=1))
    is_removed = np.logical_and(lb_n,np.logical_or(b1,b2))
    
    
    is_kept = np.logical_and(np.logical_and(lb_n,lb_f),np.argmax(Y_f,axis=1) == np.argmax(Y_n,axis=1))
    
    
    
    A = np.zeros((2,2))
    
    A[0,0] = np.sum(np.logical_and(is_clean,is_kept))
    A[0,1] = np.sum(np.logical_and(is_noisy,is_kept))
    A[1,0] = np.sum(np.logical_and(is_clean,is_removed))
    A[1,1] = np.sum(np.logical_and(is_noisy,is_removed))
    
    
    
    assert (np.sum(is_removed) + np.sum(is_kept) == np.sum(lb_n))
    assert (np.sum(is_clean) + np.sum(is_noisy) == np.sum(lb_n))
    
    
    return A
    
    
def get_confmat_TN(A):
    return A[0,0]
def get_confmat_FN(A):
    return A[0,1]
def get_confmat_FP(A):
    return A[1,0]
def get_confmat_TP(A):
    return A[1,1]
def get_confmat_recall(A):
    TP = get_confmat_TP(A)
    FN = get_confmat_FN(A)
    if TP + FN == 0:
        return np.nan

    return TP/(TP+FN)

def get_confmat_specificity(A):
    TN = get_confmat_TN(A)
    FP = get_confmat_FP(A)
    if TN + FP == 0:
        return np.nan

    
    return TN/(TN+FP)

def get_confmat_precision(A):
    TP = get_confmat_TP(A)
    FP = get_confmat_FP(A)
    if (TP + FP) == 0:
        return np.nan
    
    return TP/(TP + FP)

def get_confmat_npv(A):
    TN = get_confmat_TN(A)
    FN = get_confmat_FN(A)
    if TN + FN == 0:
        return np.nan
    
    return TN/(TN + FN)


def get_confmat_acc(A):
    TP = get_confmat_TP(A)
    TN = get_confmat_TN(A)
    FN = get_confmat_FN(A)
    FP = get_confmat_FP(A)
    return (TP+TN)/(TP+TN+FP+FN)

def get_confmat_f1_score(A):
    precision = get_confmat_precision(A)
    recall = get_confmat_recall(A)
    
    
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return 0
    
    
    return (2*precision*recall)/(precision+recall)
    


def get_confmat_dict(A):
    pairs = [
        ("TN",get_confmat_TN(A)),
        ("FN",get_confmat_FN(A)),
        ("FP",get_confmat_FP(A)),
        ("TP",get_confmat_TP(A)),
        ("recall",get_confmat_recall(A)),
        ("specificity",get_confmat_specificity(A)),
        ("precision",get_confmat_precision(A)),
        ("npv",get_confmat_npv(A)),
        ("acc",get_confmat_npv(A)),
        ("f1_score",get_confmat_f1_score(A)),
        ]
    dct = {}
    for k,v in pairs:
        dct[OUTPUT_PREFIX + "filter_" +  k] = v
    return dct



