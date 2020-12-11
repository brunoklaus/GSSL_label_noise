'''
Created on 27 de mar de 2019

@author: klaus
'''
from gssl.classifiers.classifier import GSSLClassifier
import numpy as np
import gssl.graph.gssl_utils as gutils
import scipy.linalg as sp
import scipy.sparse.linalg as ssp
import log.logger as LOG
from caffe2.python.operator_test.square_root_divide_op_test import grad
from output.folders import RESULTS_FOLDER

class LapEigLS(GSSLClassifier):
    """ Manifold Regularization with Laplacian Eigenmaps. 
        Minimizes the Least Squares in a semi-supervised way by using a linear combination of the first :math:`p` eigenfunctions. See :cite:`belkin2003`.
    """

    @GSSLClassifier.autohooks
    def __MR(self,X,W,Y,labeledIndexes,p,hook=None):
        ORACLE_Y = Y.copy()
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        l = np.reshape(np.array(np.where(labeledIndexes)),(-1))
        num_lab = l.shape[0]
        
        
        if not isinstance(p, int):
            p = int(p * num_lab)
    
        if p > Y.shape[0]:
            p = Y.shape[0]
            LOG.warn("Warning: p greater than the number of labeled indexes",LOG.ll.CLASSIFIER)
        #W = gutils.scipy_to_np(W)
        #W =  0.5* (W + W.T)
        L = gutils.lap_matrix(W, is_normalized=True)
        D = gutils.deg_matrix(W,flat=False)
        L = 0.5*(L+L.T)
        
        def check_symmetric(a, tol=1e-8):
            return np.allclose(a, a.T, atol=tol)
        def is_pos_sdef(x):
            return np.all(np.linalg.eigvals(x) >= -1e-06)
        import scipy.sparse
        sym_err = L - L.T
        sym_check_res = np.all(np.abs(sym_err.data) < 1e-7)  # tune this value
        assert sym_check_res
        
        """
            EigenFunction Calculation
        """
        import time
        start_time = time.time()
        
        import os.path as osp
        from settings import INPUT_FOLDER

        
        cache_eigvec = osp.join(INPUT_FOLDER,'eigenVectors.npy')
        cache_eigval = osp.join(INPUT_FOLDER,'eigenValues.npy')
        
        if osp.isfile(cache_eigval) and osp.isfile(cache_eigvec):
            print("Loading eigenvectors/eigenvalues...")
            eigenValues, eigenVectors = np.load(cache_eigval), np.load(cache_eigvec)
        else:
            print("Creating  eigenvectors/eigenvalues...")            
            eigenValues, eigenVectors = ssp.eigsh(L, k=p, M=ssp.aslinearoperator(scipy.sparse.eye(L.shape[0])),sigma=-0.75, which='LM',mode='cayley',tol=1e-05)
            time_elapsed = time.time() - start_time
            LOG.info("Took {} seconds to calculate eigenvectors".format(int(time_elapsed)))
            idx = eigenValues.argsort() 
            eigenValues = eigenValues[idx]
            LOG.debug(eigenValues)
            assert eigenValues[0] <= eigenValues[eigenValues.shape[0]-1]
            eigenVectors = eigenVectors[:,idx]
            np.save(cache_eigval,arr=eigenValues)
            np.save(cache_eigvec,arr=eigenVectors)
        
        import tensorflow as tf
        
        U = eigenVectors
        #Y =  U @ (U.T @ Y)
        
        Y# = Y - (np.min(Y,axis=-1)[:,None])
        
        U, X, Y = [tf.constant(x.astype(np.float32)) for x in [U,X,Y]]
        
        _U_times_U = tf.multiply(U,U)
        
        N = X.shape[0]
        
        def my_generator():
            while True:
                yield (X, Y)
        
                

        
        
        
        
        def to_sp_diag(x):
            n = tf.cast(x.shape[0],tf.int64)
            indices = tf.concat([tf.range(n,dtype=tf.int64)[None,:],
                                 tf.range(n,dtype=tf.int64)[None,:]],axis=0)
            return tf.sparse.SparseTensor(indices=tf.transpose(indices),values=x,dense_shape=[n,n])
                
        @tf.function
        def smooth_labels(labels, factor=0.001):
            # smooth the labels
            labels = tf.cast(labels,tf.float32)
            labels *= (1 - factor)
            labels += (factor / tf.cast(tf.shape(labels)[0],tf.float32))
            # returned the smoothed labels
            return labels
        @tf.function
        def divide_by_row(x,eps=1e-07):
            x = tf.abs(x)
            x = x + eps # [N,C]    [N,1]
            #tf.print(tf.reduce_sum(x,axis=-1))
            y = x / (tf.reduce_sum(x,axis=-1)[:,None])
            #tf.print(tf.reduce_sum(y,axis=-1))
            #tf.print("===============")
            #tf.assert_equal(1,0)
            
            return x / (tf.reduce_sum(x,axis=-1)[:,None])
        
        def spd_matmul(x,y):
            return tf.sparse.sparse_dense_matmul(x,y)
        
        def mult_each_row_by(X,by):
            #[N,C]  [N,1]
            return X * by[None,:]
        
        def mult_each_col_by(X,by):
            #[N,C]  [1,C]
            return X * by[:,None]
        
        
        @tf.function
        def accuracy(y_true,y_pred):
            acc = tf.cast(tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32)
            acc = tf.cast(acc,tf.float32)
            return tf.reduce_mean(acc)
        
        
        
        """
        
            DEFINE VARS
        """
        
        MU = tf.Variable(1.0,name="MU")
        LAMBDA = tf.Variable(eigenValues.astype(np.float32),name="LAMBDA")        
        PI = tf.Variable(tf.ones(shape=(tf.shape(Y)[0],),dtype=tf.float32),name="PI")
        
        @tf.function
        def forward(Y,U,PI, mode='train',remove_diag=True):
            mu = tf.abs(MU)
            if mode == 'train':
                U = tf.gather(U,indices=np.where(labeledIndexes)[0],axis=0)
                Y = tf.gather(Y,indices=np.where(labeledIndexes)[0],axis=0)
                PI = tf.gather(PI,indices=np.where(labeledIndexes)[0],axis=0)
                
            
            pi_Y = mult_each_col_by(Y, by=PI)
            #pi_Y = U@(tf.transpose(U)@pi_Y)
            
            alpha = tf.pow(2.0,-tf.math.reciprocal(tf.abs(mu)))
            lambda_tilde = tf.math.reciprocal(1-alpha + alpha*LAMBDA)
            _self_infl = mult_each_row_by(tf.square(U),by=lambda_tilde) #Square each element of U, then dot product of each row with lambda_tilde
            _self_infl = tf.reduce_sum(_self_infl,axis=1)
            
            #sparse_lambda_tilde = to_sp_diag(lambda_tilde)
            
            _P_op = U @ (mult_each_col_by(  (tf.transpose(U) @  pi_Y) ,by=lambda_tilde )  )
            if not remove_diag:
                _diag_P_op = tf.zeros_like(mult_each_col_by(pi_Y,by=_self_infl))
            else:
                _diag_P_op = mult_each_col_by(pi_Y,by=_self_infl)
            #tf.print(tf.reduce_min(_P_op),tf.reduce_min(_P_op-_diag_P_op),tf.reduce_min(pi_Y))
            #tf.print(tf.reduce_max(_P_op),tf.reduce_max(_P_op-_diag_P_op),tf.reduce_max(pi_Y))
            
            return divide_by_row( tf.clip_by_value(_P_op-_diag_P_op,0,999999))
        

        xent = lambda y_, y: tf.reduce_mean(-tf.reduce_sum(y * tf.cast(tf.math.log(smooth_labels(y_,factor=0.01)),tf.float32),axis=[1]))
        sq_loss = lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.square(y_-y),axis=[1]))
        abs_loss = lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.abs(y_-y),axis=[1]))
            
        NUM_ITER = 3000
        opt = tf.keras.optimizers.RMSprop(learning_rate=1e-01)
        
        Y_l = tf.gather(Y,indices=np.where(labeledIndexes)[0],axis=0)
        
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('tkagg')
        import pandas as pd
        
        L = []
        df = pd.DataFrame()
        
        for i in range(-20,NUM_ITER):
            MU.assign(i)
            with tf.GradientTape() as t:
                # no need to watch a variable:
                # trainable variables are always watched
                F_l = forward(Y,U,PI,mode='train')
                loss_sq = sq_loss(F_l,Y_l)
                loss = abs_loss(F_l,Y_l)
                loss_xent = xent(F_l,Y_l)
                
                mx = tf.reduce_min(F_l)
            acc = accuracy(Y_l,F_l)
            acc_true = accuracy(ORACLE_Y,forward(Y,U,PI,mode='eval'))
            
            L.append(np.array([i,loss_sq,loss,loss_xent,acc,acc_true])[None,:])
            
            if i > 50:
                df = pd.DataFrame(np.concatenate(L,axis=0),
                                  columns=["iter",'l2_loss','l1_loss','xent_loss','acc','acc_true'],
                                  index=range(len(L)))
                for c in df.columns[1:]:
                    plt.plot(df['iter'].values,df[c].values,linewidth=2)
                plt.grid(which='both')
                plt.legend(df.columns[1:], loc='upper right')
                plt.ylim((0,1))
                import os.path as osp
                plt.savefig(osp.join(RESULTS_FOLDER,"with_diag_extraction.png"))
                plt.show()
                raise ""
            #### Option 1
            
            # Is the tape that computes the gradients!
            #trainable_variables = [MU]
            #gradients = t.gradient(loss, trainable_variables)
            # The optimize applies the update, using the variables
            # and the optimizer update rule
            #opt.apply_gradients(zip(gradients, trainable_variables))
            #print(gradients)
            MU.assign(tf.abs(MU)+1e-08)
            PI.assign(tf.clip_by_value(PI,0,9999))

            if i % 10 == 0:
                alpha = tf.pow(2.0,-tf.math.reciprocal(tf.abs(MU)))
                print(f"Acc: {acc.numpy():.3f} Loss: {loss.numpy():.3f}; alpha = {alpha.numpy():.3f}; PI min = {tf.reduce_min(PI).numpy():.3f} ")
            
        
        return tf.clip_by_value(forward(Y,U,PI,mode='eval'),0,999999).numpy()
        """
            
        data, row,col = np.reciprocal((1/mu)*eigenValues+1), np.arange(eigenVectors.shape[1]) , np.arange(eigenVectors.shape[1]) 

        
        
        M_inv = scipy.sparse.coo_matrix( (data,(row,col)), shape=[row.shape[0],col.shape[0]])
        
        A = M_inv @ E.T @ Y
        print(A)
        print(A.shape)
        return np.clip(E@A,0,None)
        
                
        e_lab = E[labeledIndexes,:]
        #TIK = np.ones(shape=e_lab.shape)
        mu = 0.01
        TIK = (1/mu)*np.sqrt(scipy.sparse.diags([(eigenValues) ],[0]))
        try:
            A = np.linalg.inv(e_lab.T @ e_lab + TIK.T@TIK) @ e_lab.T        
        except:
            print("WARNING: Could not invert")
            A = np.linalg.pinv(e_lab.T @ e_lab + TIK.T@TIK) @ e_lab.T        
        F = np.zeros(shape=Y.shape)
        
        y_m =  np.argmax(Y, axis=1)[labeledIndexes]
        
        for i in range(p):
            if not hook is None:
                hook._step(step=i,X=X,W=W,Y=E[:,i])
        
        
        for i in range(Y.shape[1]):
            c = np.ones(num_lab)
            c[y_m != i] = -1
            a = A @ np.transpose(c)
            LOG.debug(a,LOG.ll.CLASSIFIER)
            for j in np.arange(F.shape[0]):
                F[j,i] = np.dot(a,E[j,:])
                F[j,i] = max(F[j,i],0)

        return (F)
        """
        

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__MR(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,p=self.p,hook=hook))


    def __init__(self,p=0.2):
        """ Constructor for LapEigLS classifier.
            
        Args:
            p (Union[float,int]). The number of eigenfunctions. It is given as either the absolute value if integer, or a percentage of
                the labeled data. if float. Default is ``0.2``
        """
        self.p = p