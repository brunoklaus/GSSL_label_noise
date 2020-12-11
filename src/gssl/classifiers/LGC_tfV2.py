'''
Created on 23 de nov de 2019

@author: klaus
'''
import tensorflow.compat.v1 as tf
import numpy as np
from gssl.classifiers.classifier import GSSLClassifier

import time
from scipy import sparse
import log.logger as LOG 



from gssl.graph.gssl_utils import scipy_to_np as _to_np
import scipy.sparse

def get_S_fromtensor(W):
    wsum = tf.sparse.reduce_sum(W,axis=1)
    wsum = tf.reshape(wsum,(-1,))
    d_sqrt = tf.reciprocal(tf.sqrt(wsum))
    d_sqrt = tf.where(tf.is_finite(d_sqrt),d_sqrt,tf.ones(shape=tf.shape(d_sqrt)))
    
    d_sqrt_i = tf.gather(d_sqrt,W.indices[:,0])
    d_sqrt_j = tf.gather(d_sqrt,W.indices[:,1])
    
    
    S = tf.sparse.SparseTensor(indices=W.indices,
                                    values=W.values * d_sqrt_i * d_sqrt_j,
                                    dense_shape=W._dense_shape)
    
    return S

def update_F_2(TOTAL_ITER,SIGMA,LAP,F_0):
    MINUS_SIGMA_INV = -1.0*tf.reciprocal(SIGMA)[:,tf.newaxis]
    #SIGMA_INV = SIGMA_INV[:,tf.newaxis]
    LAP = LAP* MINUS_SIGMA_INV
    i = tf.constant(0)
    c = lambda i,F: tf.less(i, TOTAL_ITER)
    b = lambda i,F: (tf.add(i, 1), tf.sparse.matmul(LAP ,F) + F_0)
    r = tf.while_loop(c, b, [i,F_0])
    return r 


def update_F(TOTAL_ITER,SIGMA,S,F_0):
    ALPHA =  tf.reciprocal(1+SIGMA)
    i = tf.constant(0)
    c = lambda i,F: tf.less(i, TOTAL_ITER)
    b = lambda i,F: (tf.add(i, 1),F_0*((1 - ALPHA)[:,np.newaxis]) + tf.sparse.matmul(S,F)*(ALPHA[:,np.newaxis]) )
    r = tf.while_loop(c, b, [i,F_0])
    return r
            
def get_P(TOTAL_ITER,ALPHA,S,F_0):
    i = tf.constant(0)
    c = lambda i,F: tf.less(i, TOTAL_ITER)
    b = lambda i,F: (tf.add(i, 1),(1 - ALPHA)*F_0 + ALPHA*tf.sparse.matmul(S,F))
    r = tf.while_loop(c, b, [i,F_0])
    return r



""" UTIL FUNCTIONS BEGIN """
def gather(x,F):
    with tf.name_scope("gather"):
        return tf.gather(F,tf.reshape(x,(-1,)))  
    

def repeat(x,n):
    with tf.name_scope('repeat'):
            x = tf.expand_dims(x, axis=-1)
            x = tf.tile(x,[1,n])
            return x
def row_normalize(x):
    with tf.name_scope('row_norm'):
        x = tf.clip_by_value(x,0.,1.)
        s= tf.cast(tf.shape(x)[1],tf.float32)
        vec = tf.reduce_sum(x,axis=1)
        x = x/repeat(vec,s)
        x = tf.where(tf.is_finite(x),x,tf.ones(shape=tf.shape(x))/s)
        return x
""" UTIL FUNTIONS END """

def convert_sparse_matrix_to_sparse_tensor(X,var_values=False):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    indices = np.reshape(np.asarray(indices),(-1,2))
    
    return tf.SparseTensor(indices, np.reshape(np.asarray(coo.data).astype(np.float32),(-1,)), coo.shape)
    
def LGC_iter_TF(X,W,Y,labeledIndexes, alpha = 0.1,num_iter = 1000, hook=None):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        
        """ Set W to sparse if necessary, make copy of Y """
        W = sparse.csr_matrix(W)        
        Y = np.copy(Y)
        """ Zero OUT unlabeled """
        l = np.sum(labeledIndexes)
        
        
        """ Create Indicator Y """
        lids = np.where(labeledIndexes)[0]
        itertool_prod = [[i,j] for i in range(l) for j in range(l)]
        row = np.asarray([lids[i] for i in range(l)])
        col = np.asarray([i for i in range(l)])
        data = np.asarray([1.0]*l)
        temp_Y = _to_np(scipy.sparse.coo_matrix( (data,(row,col)),shape=(W.shape[0],l) ))
    
        
        from gssl.graph.gssl_utils import lap_matrix
        LAP = convert_sparse_matrix_to_sparse_tensor(lap_matrix(W, is_normalized=True))
        
        """ Convert W to tensor """
        W = convert_sparse_matrix_to_sparse_tensor(W)
        LOG.debug(W,LOG.ll.CLASSIFIER)
        
        """ Get degree Matrix """
        D =  tf.sparse.reduce_sum(W,axis=1)
        
        
        """ F_0 is a copy of the label matrix, but we erase the information on labeled Indexes """
        F_0 = np.copy(Y).astype(np.float32) 
        F_0[np.logical_not(labeledIndexes),:] = 0.0
        
        
        
        SIGMA = tf.Variable(tf.cast(np.asarray([(1-alpha)/alpha]*Y.shape[0]),tf.float32 ))
        SIGMA_INV = tf.reciprocal(SIGMA)
        SIGMA_INV = SIGMA_INV[:,tf.newaxis]
        """
            Run variable initializers 
        """
        global_var_init=tf.global_variables_initializer()
        local_var_init = tf.local_variables_initializer()
        sess.run([global_var_init,local_var_init])    
        
        
        
        """
            CREATE S - Needed for LGC propagation
        """
        S =  get_S_fromtensor(W)
        
        MINUS_S = tf.sparse.SparseTensor(S.indices,-1*S.values,S._dense_shape)
        I = tf.sparse.eye(tf.shape(S)[0]) * 1.01 * np.ones((Y.shape[0],))
        #LAP = tf.sparse.add(I,MINUS_S)
        

        """
        CREATE F variable
        """
        F = tf.Variable(np.copy(F_0).astype(np.float32),name="F")
        F_0 = tf.Variable(F_0)
        TOTAL_ITER = tf.constant(int(num_iter))
        
        def _to_dense(sparse_tensor):
            return tf.sparse_add(tf.zeros(sparse_tensor._dense_shape), sparse_tensor)
        calc_F = update_F_2(TOTAL_ITER, SIGMA,LAP, F_0)
        #calc_F = update_F(TOTAL_ITER, SIGMA,S, F_0)
        
        assign_to_F = tf.assign(F,calc_F[1]* SIGMA_INV)
        
        MINUS_SIGMA_INV = -1.0*tf.reciprocal(SIGMA)
        
        """
            Run variable initializers 
        """
        global_var_init=tf.global_variables_initializer()
        local_var_init = tf.local_variables_initializer()
        sess.run([global_var_init,local_var_init])    
        
        

        
        c = time.time()
        sess.run(assign_to_F)
        elapsed = time.time() - c
        LOG.info('Label Prop (excluding initialization) done in {:.2} seconds'.format(elapsed),
                 LOG.ll.CLASSIFIER)
        
        print(sess.run(tf.reduce_min(F)))
        raise ""
        
        result  = F.eval(sess) 
        sess.close()
        return result


