"""
gssl_utils.py
====================================
Module containing utilities for GSSL algorithms.
"""

import numpy as np
import sklearn.model_selection as skmm
import sklearn.manifold as skmf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from plotly.api.v2.users import current
import warnings
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg


from matplotlib.pyplot import sci
def deg_matrix(W,pwr=1,flat=False, NA_replace_val = 1.0):
    """ Returns a diagonal matrix with the row-wise sums of a matrix W."""
    ws = W.sum(axis=0) if scipy.sparse.issparse(W) else np.sum(W,axis=0)
    D_flat = np.reshape(np.asarray(ws),(-1,))
    D_flat = np.power(D_flat,np.abs(pwr))
    is_zero = (D_flat == 0)
    if pwr < 0:
        D_flat[np.logical_not(is_zero)] = np.reciprocal(D_flat[np.logical_not(is_zero)])
        D_flat[is_zero] = NA_replace_val
    
    if scipy.sparse.issparse(W):
        

        if flat:
            return D_flat
        else:
            row  = np.asarray([i for i in range(W.shape[0])])
            col  = np.asarray([i for i in range(W.shape[0])])
            coo = scipy.sparse.coo_matrix((D_flat, (row, col)), shape=(W.shape[0], W.shape[0]))
            return coo.tocsr()
    else:
        if flat:
            return D_flat
        else:
            return(np.diag(D_flat))

def scipy_to_np(X):
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X)

def lap_matrix(W,is_normalized):
    """ Returns the graph Laplacian of some matrix W.
    
    Args:
        W (`NDArray[float].shape[N,N]`) : The given matrix.
        is_normalized (bool) : If ``True``, returns :math:`L = I - D^{-1/2} W D^{-1/2}`. Otherwise, returns :math:`L = D^{-1}W`.
    Returns:
        `NDArray[float].shape[N,N]` : The normalized or unnormalized graph Laplacian
    """
    import scipy.sparse
        
        
    if is_normalized:
            d_sqrt = deg_matrix(W,pwr=-1/2,NA_replace_val=1.0,flat=False)
            S = (d_sqrt @ W) @ d_sqrt 
            I = scipy.sparse.identity(W.shape[0]) if scipy.sparse.issparse(W) else np.identity(W.shape[0]) 
            return( I - S )
    else:
        return( deg_matrix(W) - W )



def init_matrix(Y,labeledIndexes):
    """ Creates a matrix containing the confidence for each class.
    
    Args:
        Y (`[NDArray[int].shape[N]`) : array of true labels.
        labeledIndexes (`NDArray[bool].shape[N]`) : determines which indices are to be considered as labeled.
    Returns:
        `[NDArray[float].shape[N,C]`: A matrix `init` such that `init[i,j]` has the confidence that the i-th instance has 
        the label corresponding to the j-th class.
    """
    Y = np.copy(Y)
    Y = np.array(Y) - np.min(Y)
    M = np.max(Y)   
    def one_hot(x):
        oh = np.zeros((M+1))
        oh[x] = 1
        return oh 
    Y_0 = np.zeros((Y.shape[0],M+1))
    Y_0[labeledIndexes,:] = [one_hot(x) for x in Y[labeledIndexes]]
    return(Y_0)

def init_matrix_argmax(Y):
    """ Returns the argmax of each row of a matrix. """
    return(np.argmax(Y,axis=1))


def split_indices(Y,split_p = 0.5,seed=None):
    """ Returns a percentage p of indices, using stratification.
    
    Args:
        Y (`NDArray.shape[N]`) : the vector from which to split with stratification w.r.t. each number that appears.
        split_p (float) : the percentage of stratified indexes to return
        seed (float) : Optional. Used to reproduce results. 
        
    Returns:
        `NDArray.shape[N*split_p]`: vector with `split_p` of indexes after stratified sampling.
    
    Raises:
        ValueError: if `split_p` is an invalid percentage.

    """
    
    if 0 > split_p or split_p > 1:
        raise ValueError("Invalid percentage")
    
    if split_p == 1.0:
        return np.ones((Y.shape[0])).astype(np.bool)
    
    index_train, _  = skmm.train_test_split(np.arange(Y.shape[0]),
                                                     stratify=Y,test_size=1-split_p,
                                                     random_state=seed)
    
    b = np.zeros((Y.shape[0]),dtype=np.bool)
    b[index_train] = True
    
    return b

def accuracy_unlabeled(Y_pred,Y_true, labeled_indexes):
    """ Calculates percentage of correct predictions on unlabeled indexes only."""
    unlabeled_indexes = np.logical_not(labeled_indexes)
    return(np.sum( (Y_pred[unlabeled_indexes] == Y_true[unlabeled_indexes]).\
                   astype(np.int32) )/Y_true[unlabeled_indexes].shape[0])
def accuracy(Y_pred,Y_true):
    """ Calculates percentage of correct predictions."""
    return(np.sum( (Y_pred== Y_true).astype(np.int32) )/Y_true.shape[0])

    
    
def get_pred(Y):
    """ Calculates predictions from a belief matrix.
    
    Args:
        Y (`NDArray[float].shape[N,C]`) : belief matrix.
    
    Returns:
        `NDArray[int].shape[N]` : prediction vector, each entry numbered from 0 to C-1.
     """
    return np.reshape(np.argmax(np.asarray(Y),axis=1),(-1,))
    
def class_mass_normalization(F,Y,labeledIndexes,normalize_rows=True):
    UlIndexes = np.logical_not(labeledIndexes)
    
    q = np.zeros((Y.shape[1]))
    
    for j in range(Y.shape[1]):
        q[j] = np.sum(Y[labeledIndexes,j])
    q = q / np.sum(q)
    
    for j in range(Y.shape[1]):
        q[j] = np.sum(Y[labeledIndexes,j])
    
    
    F = np.array(F)
    
    if normalize_rows:
        for i in range(F.shape[0]):
            if np.sum(F[i,:])>0:
                F[i,:] = F[i,:] / np.sum(F[i,:]) 
    
    
    for j in range(F.shape[1]):
        
        if np.sum(F[UlIndexes,j]) > 0:
            c = (q[j]/np.sum(F[UlIndexes,j]))
            F[:,j] = c * F[:,j]
        elif np.sum(F[UlIndexes,j]) == 0:
            warnings.warn("Warning: classification has sum of col = 0")
        else:
            raise Exception("Classification has sum of col < 0")
    
    return F

def get_Isomap(X,n_neighbors = 5):
    return(skmf.Isomap(n_neighbors).fit_transform(X))

def get_PCA(X):
    return(PCA.fit_transform(X))

def get_Standardized(X):
    return(StandardScaler().fit_transform(X))


def calc_Z(Y, labeledIndexes,D,estimatedFreq=None,weigh_by_degree=False):
    """ Calculates matrix Z used for GTAM/LDST label propagation.
    
    Args:
        Y (`[NDArray[int].shape[N,C]`) : confidence matrix
        labeledIndexes (`NDArray[bool].shape[N]`) : determines which indices are to be considered as labeled.
        D (`[NDArray[float].shape[N]`) : array of diagonal entries of degree matrix.
        estimatedFreq(´NDArray[float].shape[C]`) : Optional. The estimated class frequencies. If absent, it is assumed all 
            classes occur equally often.
        reciprocal (bool) :  If ``True``, use reciprocal of the degree instead. Default is ``False`` 
        
    Returns:
        `[NDArray[int].shape[N,C]` : Matrix Z, which normalizes Y by class frequencies and degree.
    """ 
    
    if estimatedFreq is None:
        estimatedFreq = np.repeat(1,Y.shape[0])
    if Y.ndim == 1:
        Y = init_matrix(Y,labeledIndexes)
        
    Z = np.array(Y)
    for i in np.where(labeledIndexes == True)[0]:
        Z[i,:] = 0
        if weigh_by_degree:
            Z[i,np.argmax(Y[i,:])] = D[i]
        else:
            Z[i,np.argmax(Y[i,:])] = 1

    for i in np.arange(Y.shape[1]):
        Z[:,i] = (Z[:,i] / np.sum(Z[:,i])) * estimatedFreq[i] * len(estimatedFreq)

    return(Z)




def extract_lap_eigvec(L,m,D=None,remove_first_eig=True):
    """ Extract  ``m`` eigenvectors and eigenvalues of the laplacian, in non-decreasing order. 
    
        Args:
            L (`[NDArray[float].shape[N,N]`) : laplacian matrix            
            m (int) : number of eigenvectors to extract
            D (`[NDArray[float].shape[N,N]`) : extra matrix for generalized eigenvalue problem

        
        Returns:
            Pair[NDArray[float].shape[M,N],NDArray[float].shape[M,M]] : matrix of eigenvectors, and diagonal matrix of eigenvalues
            
    """
    from gssl.classifiers.LGC_tf import convert_sparse_matrix_to_sparse_tensor
    
    def check_symmetric(a, tol=1e-10):
        return np.sum(a-a.T) < tol
    
    if remove_first_eig:
        m = min(m + 1,L.shape[0])
    
    
    if not check_symmetric(L):
        L = 0.5*(L+L.transpose())
    
    L = scipy.sparse.csc_matrix(L)
    
    if check_symmetric(L):
            S, U = scipy.sparse.linalg.eigsh(L,m,sigma=0, which='LM')
    else:
        raise ""
        S, U = scipy.sparse.linalg.eigen(L,m,sigma=0, which='LM')
    
    
    if remove_first_eig:
        U = U[:,1:]
        S = S[1:]    
    """ Make each eigenvector have norm equal to 1 """
    U = U / np.linalg.norm(U,axis=0)

    
    
    
    S = scipy.sparse.diags([S],[0])
    
    
    return U,S


def labels_indicator(labeledIndexes):
    """ Returns a Diagonal matrix J indicating whether the instance is labeled or not
        Args:
            labeledIndexes(`[NDArray[bool].shape[N]`) 
    """
    return scipy.sparse.diags([labeledIndexes.astype(np.float)],[0])
    
    
    
