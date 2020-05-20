'''
Created on 27 de mar de 2019

@author: klaus
'''
from gssl.classifiers.classifier import GSSLClassifier
import numpy as np
import gssl.graph.gssl_utils as gutils
import scipy.linalg as sp
import scipy.sparse
import scipy.linalg
import log.logger as LOG
from gssl.graph.gssl_utils import scipy_to_np as _to_np

class SIISClassifier(GSSLClassifier):
    """ Label Propagation based on Semi-Supervised Learning under Inadequate and Incorrect Supervision (SIIS). """

    
    @staticmethod
    def edge_mat(W):
            W = W.tocoo()
            
            nz = W.data.shape[0]
            row  = np.zeros((2*nz,))
            col  = np.zeros((2*nz,))
            data = np.zeros((2*nz,))
            """
            import scipy.stats
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use("TkAgg")
            plt.hist(W.data)
            plt.show()
            plt.close()
            LOG.debug(scipy.stats.describe(W.data),LOG.ll.CLASSIFIER)
            raise ""
            """
            for k in range(nz):
                row[2*k] = k
                col[2*k] = W.row[k]
                data[2*k] = W.data[k]
    
                row[2*k + 1] = k
                col[2*k + 1] = W.col[k]
                data[2*k + 1] = -W.data[k]
            
            coo = scipy.sparse.coo_matrix((data, (row, col)), shape=(nz, W.shape[0]))
            
            assert abs(coo.sum()) < 1e-08 
            
            return coo.tocsr()

    @GSSLClassifier.autohooks
    def __SIIS(self,X,W,Y,labeledIndexes,m,alpha,beta,rho,max_iter,hook=None):
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        if m is None:
            m = W.shape[0]
        
        
        
        c = Y.shape[1]
        
        W = scipy.sparse.csr_matrix(W) / np.mean(W.data)

        
        
        D = gutils.deg_matrix(W, pwr=1.0)

        
        L = gutils.lap_matrix(W, is_normalized=True)
        
        U, SIGMA = gutils.extract_lap_eigvec(L,m,remove_first_eig=True)
        
        U = scipy.sparse.csr_matrix(U)
        SIGMA = _to_np(SIGMA)
        
    

        
        
        J = gutils.labels_indicator(labeledIndexes)
        
        """ !!! """
        P = SIISClassifier.edge_mat(W) 
        
        
        
        """ Initialize params """
        LAMB_1 = np.ones((P.shape[0],c))
        LAMB_2 = np.ones((Y.shape[0],c))
        mu = 1.0
        mu_max = 10000000.0
        eps = 1/(10000)
        
        """ Reusable matrices """
        JU = _to_np(J@U)
        PU = _to_np(P@U)
        PU_T = PU.transpose()
        JU_T = JU.transpose()
        
        
        
        A = np.zeros((m,c))
        Q = None
        B = None
        
        improvement  = 1
        iter = 0
        
        """ TODO: Tensorflow version 
            import tensorflow as tf
            with tf.Session() as sess:
                A = tf.Variable(1e-06*tf.ones((m,c),dtype=tf.float64))
                sess.run(tf.global_variables_initializer())
                
                C = tf.reduce_sum(tf.linalg.norm(tf.matmul(PU,A),axis=1)) +\
                 alpha*tf.reduce_sum(tf.linalg.norm(tf.matmul(_to_np(U)[labeledIndexes,:],A)-Y[labeledIndexes,:],axis=1)) +\
                 beta* tf.trace(tf.matmul(tf.matmul(tf.transpose(A),SIGMA),A))
                opt = tf.train.AdamOptimizer(learning_rate=0.5*1e-02)
                opt_min = opt.minimize(C)
                sess.run(tf.global_variables_initializer())
                for i in range(2000):
                    sess.run(opt_min)
                    LOG.debug(sess.run(C),LOG.ll.CLASSIFIER)
                LOG.debug(sess.run(C),LOG.ll.CLASSIFIER)    
                F = _to_np(U)@sess.run(A)
                
                LOG.debug(F.shape,LOG.ll.CLASSIFIER)
            
        
        """
        A = np.zeros((m,c))
        while  iter <= max_iter and improvement > eps:
            
            """ Update Q """
            N = PU@A - (1/mu)*LAMB_1
            N_norm = np.linalg.norm(N, axis=1)
            
            
            to_zero = N_norm <= (1/mu)
            mult = ((N_norm - (1/mu))/N_norm)
            N = N * mult[:,np.newaxis]
            
            
            N[to_zero,:] = 0.0
            Q = N 
            
            """ Update B """
            M = JU@A - Y - (1/mu)*LAMB_2
            M_norm = np.linalg.norm(M,axis=1)
            to_zero = M_norm <= (alpha/mu)
            mult = ((M_norm - (alpha/mu))/M_norm)
            M = M * mult[:,np.newaxis]
            M[to_zero,:] = 0.0 
            B = M
            
            
            old_A = A
            """ Update A """
            
            A_inv_term = 2*beta*SIGMA + mu*PU_T@PU + mu*JU_T@JU
            A_inv_term = np.linalg.inv(A_inv_term) 
            A = A_inv_term @ \
                (PU_T@ LAMB_1 + JU_T@LAMB_2 +\
                  mu * PU_T@Q + mu* JU_T @ (B + Y) )
        
            """ Update Lagrangian coeffs """
            LAMB_1 = LAMB_1 + mu* (Q - PU@A)
            LAMB_2 = LAMB_2 + mu*(B- JU@A + Y)
            """ Update penalty coeffficients """
            mu = min(rho*mu,mu_max)
        
        
            
            if not old_A is None:
                improvement = (np.max(np.abs(A-old_A)))/np.max(np.abs(old_A))
                
            
            
            LOG.debug("Iter {}".format(iter),LOG.ll.CLASSIFIER)
            iter += 1
        
        C = np.sum(np.linalg.norm(PU@A,axis=1)) + alpha*np.sum(np.linalg.norm(JU@A - Y,axis=1)) +\
             beta*np.trace(A.T@SIGMA@A)
        LOG.debug("Iter {} - Cost {}".format(iter,C),LOG.ll.CLASSIFIER)
            
        
        F = U@A
            
            
        for i in range(F.shape[0]):
            mx = np.argmax(F[i,:])
            F[i,:] = 0.0
            F[i,mx] = 1.0
        
        
        return F
        
        

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__SIIS(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,m=self.m,alpha=self.alpha,
                           beta=self.beta,rho=self.rho,max_iter=self.max_iter,hook=hook))


    def __init__(self,m=100,beta=10,alpha=100,rho=1.2,max_iter=100):
        """ Constructor for SIIS classifier.
            
        Args:
            m (Union[float,int]): The number of eigenvectors. It is given as either the absolute value (int), or a percentage of
                the labeled data (float). Default is ``0.2``
        """
        self.m = m
        self.beta = beta
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        
        
if __name__ == "__main__":
    pass
    
    
    
