"""
gssl_affmat.py
====================================
Module that handles the construction of affinity matrices.
"""

import numpy as np
import scipy.spatial.distance as scipydist
from sklearn.neighbors import NearestNeighbors
from functools import partial
import quadprog
import sys
import progressbar
from settings import load_sparse_csr,CIFAR_MAT_FOLDER
import scipy.sparse
import time
import gssl.graph.gssl_utils as  gutils
import faiss
def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    s_tuple = sorted(tuples, key=lambda x: (x[2], x[1]))
    
    row,col,data  = zip(*s_tuple)
    return scipy.sparse.coo_matrix((data, (row,col)), shape=m.shape)


class AffMatGenerator(object):
    """Constructs a dense affinity matrix from some specification.
    """
    
    
    def get_or_calc_Mask(self,X):
        """ Gets the previously computed mask for affinity matrix, or computes it."""
        if self.K is None:
            if self.mask_func == "load":
                self.K = load_sparse_csr(self.load_path)
            else:
                self.K = self.mask_func(X)
        return(self.K.astype(np.double))
    
    
    def handle_adaptive_sigma(self,K):
        #Take the mean from X to the third neighbor
        #if scipy.sparse.issparse(K):
        #    K = np.asarray(K.todense())
        
        if not scipy.sparse.issparse(K):
            M = K
            M[M==0] = np.infty 
            M = np.sort(M, axis=1)
            #print(M[0,:])
            self.sigma = np.mean(M[:,9])/3
            print("Adaptive sigma is {}".format(self.sigma))
            print(self.dist_func(10))
        else:
            self.sigma = np.mean([np.sort(K.getrow(i).data)[9]/3 for i in range(K.shape[0])])
            print("Adaptive sigma is {}".format(self.sigma))
        return partial(lambda d: np.exp(-(d*d)/(2*self.sigma*self.sigma)))
    
    def W_from_K(self,X,K):
        if not scipy.sparse.issparse(K):
            if self.dist_func_str == "LNP" or self.dist_func_str == "NLNP":
                W = self.dist_func(X,K)
            else:
                W =  np.reshape([0 if x == 0 else self.dist_func(x) for x in  np.reshape(K,(-1))],K.shape)
        else:
            if self.dist_func_str == "LNP" or self.dist_func_str == "NLNP":
                raise NotImplementedError("Did not implement LNP on sparse matrix yet")
            else:
                W = scipy.sparse.csr_matrix(K)
                W.data = np.asarray([self.dist_func(x) for x in W.data])
        return W  
    
    
    def generateAffMat(self,X,Y=None,labeledIndexes=None,hook=None):
        """ Generates the Affinity Matrix.
        
            Returns:
                `NDArray[float].shape[N,N]: A dense affinity matrix.
         """
        print("Creating Affinity Matrix...")
        
        if not hook is None:
            hook._begin(X=X,Y=Y,labeledIndexes=labeledIndexes,W=None)
        
        K = self.get_or_calc_Mask(X)
        
        if self.sigma == "mean":
            self.dist_func = self.handle_adaptive_sigma(K)
        

        if not K.shape[0] == X.shape[0]:
            raise ValueError("Shapes do not match for X,K")
            
        
        W = self.W_from_K(X,K)
        
        if self.row_normalize == True:
            W = gutils.deg_matrix(W, pwr=-1.0, NA_replace_val=1.0) @ W 
        
        del K
        print("Done!")
        assert(W.shape == (X.shape[0],X.shape[0]))
        if np.max(W)==0:
            raise Exception("Affinity matrix cannot have all entries equal to zero.")
        
        if not hook is None:
            hook._end(X=X,Y=Y,W=W)

        return(W.astype(np.float32))

    
    def __init__(self,dist_func, mask_func, metric="euclidean",load_path = None,num_anchors=None, **arg):
        """ Constructs the Affinity Matrix Generator.
        
        Args:
            X (`NDArray[float].shape[N,D]`): A matrix containing the vertex positions. 
            dist_func (str): specifies the distance function to be used. Supported values:
                {
                    * gaussian: ``np.exp(-(d*d)/(2*sigma*sigma))``, where d is the distance. Requires ``sigma`` on **kwargs.
                    * LNP: Linear neighborhood propagaton. Requires ``k`` on **kwargs. 
                    * NLNP: Normalized reciprocal of  Linear neighborhood propagation. Requires ``k`` on **kwargs.
                    * constant: Every weight is set to 1.
                    * inv_norm: ``1/d``, where d is the distance
                }
            mask_func (str): specifies the function used to determine the neighborhood. Supported Values:
                {
                    * epsilon: Epsilon-neighborhood. Requires ``eps`` on **args.
                    * knn: K-nearest neighbors Requires ``k`` on **args.
                    * load: loads CSR matrix specified by `load_path`
                }
            load_path(str): Path from where to load CSR Knn matrix with precalculated distances.
            metric (str): specifies the metric when computing the distance. Default is `euclidean`. See the documentation of
                `scipy.spatial.distance.cdist` for more details.
            **arg: Remaining arguments.

        """
        self.K = None
        self.sigma = None
        self.metric = metric
        self.dist_func_str = dist_func
        self.load_path = load_path
        self.num_anchors = num_anchors
        
        if "row_normalize" in arg and arg["row_normalize"] == True:
            self.row_normalize = True
        else:
            self.row_normalize = False
            
        
        
        if dist_func in ["LNP","NLNP"]:
            mask_func = dist_func
        if mask_func in ["LNP","NLNP"]:
            dist_func = mask_func
        
        
        
        if dist_func == "gaussian":
            if not "sigma" in arg:
                raise ValueError("Did not specify sigma for gaussian")
            
            self.sigma = arg["sigma"]
            self.dist_func = partial(lambda d: np.exp(-(d*d)/(2*self.sigma*self.sigma)))
        elif dist_func == "constant":
            self.dist_func = lambda d: 1
        elif dist_func == "inv_norm":
            self.dist_func = lambda d: 1/d
        
        if mask_func == "load":
            self.mask_func = "load"
        elif mask_func == "eps":
            if not "sigma" in arg:
                raise ValueError("Did not specify eps parameter for epsilon-neighborhood")
            self.mask_func = partial(lambda X,eps: epsilonMask(X, eps),eps=arg["eps"])
        elif mask_func == "knn":
            if not "k" in arg:
                raise ValueError("Did not specify k parameter for knn-neighborhood")
            self.mask_func = partial(lambda X,k: knnMask(X, k),k=arg["k"])
        elif mask_func == "LNP":
            if not "k" in arg:
                raise ValueError("Did not specify k for LNP")
            self.mask_func = partial(lambda X,K: LNP(X, K))
        elif mask_func == "NLNP":
            if not "k" in arg:
                raise ValueError("Did not specify k for NLNP")
            self.mask_func = partial(lambda X,K: NLNP(X,K))
        
        

def epsilonMask(X,eps,metric="euclidean"):
    """
    Calculates the distances only in the epsilon-neighborhood.
    
    Args:
        X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
        eps (float) : A parameter such that K[i,j] = 0 if dist(X_i,X_j) >= eps.
    Returns:
        `NDArray[int].shape[N,N]` : a dense matrix ´K´ of shape `[N,N]` whose nonzero 
        **[i,j]** entries correspond to distances between neighbors **X[i,:],X[j,:]** .
    
    """
    print(type(X))
    assert isinstance(X, np.ndarray)
    
    K = scipydist.cdist(X,X,metric=metric)
    rows,cols = np.where(K > eps)
    K[rows,cols] = 0
    return(K)
   
def knnMask(X,k,symm = True,metric="euclidean"):
    """
    Calculates the distances only in the knn-neighborhood.
    
    Args:
        X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
        k (int) :  A parameter such that ´K[i,j] = 1´ iff X_i is one of the k-nearest neighbors of X_j
        symm (bool) : if True, then ``K[i,j] = max(K[i,j],K[j,i])``. Default is ``True``
    Returns:
        `NDArray[int].shape[N,N]` : a dense matrix ´K´ of shape `[N,N]` whose nonzero 
        **[i,j]** entries correspond to distances between neighbors **X[i,:],X[j,:]** .
    
    """

    if X.shape[0] > 2000:
        K =  _faiss_knn(X, k, symm=symm)
        return K
        
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree',metric=metric).fit(X)
    K = np.zeros((X.shape[0],X.shape[0]))
    for i in np.arange(X.shape[0]):
        distances, indices = nbrs.kneighbors([X[i,]])
        
        for dist, index in zip(distances,indices):
            K[i,index] = np.array(dist)
            if symm:
                K[index,i] = np.array(dist)
    return scipy.sparse.csr_matrix(K)

def _faiss_knn(X,k, symm= True, inner_prod = False):
    # kNN search for the graph
    X  = np.ascontiguousarray(X)
    
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    if inner_prod:
        faiss.normalize_L2(X)
        index =  faiss.GpuIndexFlatIP(res,d,flat_config)         
    else:
        index = faiss.GpuIndexFlatL2(res,d,flat_config)   # build the index
    #normalize_L2(X)
    index.add(X) 
    N = X.shape[0]
    Nidx = index.ntotal

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)



    # Create the graph
    D = np.sqrt(D[:,1:])
    
    
    
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    
    
    
    if symm:
        W = W.minimum(W.T)
    return W

def __quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]   

def LNP(X,K, symm = True):
    """ Computes the edge weights through Linear Neighborhood Propagation.
    
    Args:
         X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
         K (`NDArray[float].shape[N,N]`) : Dense Affinity mask, whose positive entries correspond to neighbors.
    Returns:
        `NDArray[float].shape[N,N]` : A dense affinity matrix whose weights minimize the linear reconstruction of each instance.
        
    """
    W = np.zeros((X.shape[0],X.shape[0]))

    if K.shape[0] !=  X.shape[0]:
        raise ValueError("Incompatible shapes for X,K")
    
    
    if K.shape[0] == 0:
        return K
    
    P = {}
    q = {}
    G = {}
    h = {}
    A = {}
    b = {}
    
    num_nbors = np.zeros((K.shape[0]))
    all_indices = [None] * X.shape[0]
    
    for i in range(K.shape[0]):
        all_indices[i] = (np.where(W[i,] > 0))
        num_nbors[i] = str(all_indices[i].shape[0])
        k = num_nbors[i]
        if not k in P.keys():
            P = np.zeros([k,k])
        if not k in q.keys():
            q[k] = np.zeros((k))
        if not k in G.keys():
            G[k] = -np.identity(k)
        if not k in h.keys():
            h[k] = np.zeros((k))
        if not k in A.keys():
            A[k] = np.ones((1,k))
        if not k in b.keys():
            b[k] = np.ones((1))
        
    
    for i in np.arange(X.shape[0]):
        k = num_nbors[i]
        indices = all_indices[i]
        for m in range(k):
            for n in range(k):
                P[m,n] = np.dot((X[i,]-X[indices[m],]),(X[i,]-X[indices[n],]).T)        
        for m in range(k):
            P[m,m] += 1e-03
          
        W_lnp = __quadprog_solve_qp(P[k], q[k], G[k], h[k], A[k], b[k])
        
        for m in range(k):
            W[i,indices[m]] = W_lnp[m]
    if symm:
        W = 0.5*(W + W.T)
    
    return(W)

def NLNP(X,K, symm = True):
    """ Computes the normalized reciprocals of the edge weights through Linear Neighborhood Propagation.
    
    Args:
         X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
         K (`NDArray[float].shape[N,N]`) : Dense Affinity mask, whose positive entries correspond to neighbors.
    Returns:
        `NDArray[float].shape[N,N]` : A dense affinity matrix whose weights are the normalized reciprocals of the ones given by Linear Neighborhood Propagation.
            
    """
    W = LNP(X,K,symm)
    for i in range(W.shape[0]):
        nonz = (np.where(W[i,] > 0))
        W[i,nonz] = np.reciprocal(W[i,nonz])
        W[i,nonz] = W[i,nonz] / np.linalg.norm(W[i,nonz])
    return(W)


 
