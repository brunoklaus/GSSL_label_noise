'''
Created on 13 de nov de 2019

@author: klaus
'''
from gssl.filters.filter import GSSLFilter
import numpy as np
import gssl.graph.gssl_utils as gutils
from gssl.classifiers.LGC_tf import LGC_iter_TF
from gssl.graph.gssl_utils import scipy_to_np as _to_np
import log.logger as LOG
import scipy.sparse
import itertools
from docutils.nodes import Labeled
from log.logger import LogLocation
from gssl.graph import gssl_utils

    


class LGC_LVO_AUTO_Filter(GSSLFilter):
    """"" Novel, automated LGC_LVO_F. Still under development """
        
    Fl = None
    def getLabeledClassification(self):
        if self.Fl is None:
            raise ValueError("Cannot get classification of labeled instances before calling 'fit'")
        return self.Fl
    
    '''
    classdocs
    '''
    @GSSLFilter.autohooks
    def LGCLVO(self,X,W,Y,labeledIndexes,mu = 99.0,lgc_iter = 10000,hook=None,which_loss="xent"):
        if which_loss is None:
            return Y, labeledIndexes
        
        Y = np.copy(Y).astype(np.float32)
        #We make a deep copy of labeledindexes
        labeledIndexes = np.array(labeledIndexes)        
        lids = np.where(labeledIndexes)[0]
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
           
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")

        """ Ensure that it is symmetric """
        W = 0.5*(W + W.transpose())
        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        if scipy.sparse.issparse(W):
            l = np.sum(labeledIndexes)
            
            
            itertool_prod = [[i,j] for i in range(l) for j in range(l)]

            row = np.asarray([lids[i] for i in range(l)])
            col = np.asarray([i for i in range(l)])
            data = np.asarray([1.0]*l)
            temp_Y = _to_np(scipy.sparse.coo_matrix( (data,(row,col)),shape=(W.shape[0],l) ))
            
            
            PL = LGC_iter_TF(X,W,Y=temp_Y,labeledIndexes=labeledIndexes,alpha = 1/(1+mu),num_iter=lgc_iter).astype(np.float32)
            
            PL = PL[labeledIndexes,:]
            PL[range(PL.shape[0]),range(PL.shape[0])] = 0   #Set diagonal to 0
            
            PL =  PL
            
            
            del temp_Y
            
            row = np.asarray([lids[x[0]] for x in itertool_prod if x[0] != x[1] ])
            col = np.asarray([lids[x[1]] for x in itertool_prod if x[0] != x[1] ])
            data = [PL[x[0],x[1]] for x in itertool_prod if x[0] != x[1]]
            P = scipy.sparse.coo_matrix((data,(row,col)),shape=W.shape).tocsr()
            P = P 
        else:
            #Identity matrix
            I = np.identity(W.shape[0])
            #Get graph laplacian
            L = gutils.lap_matrix(W, is_normalized=True)
            L = 0.5*(L + L.transpose())
            #Propagation matrix
            P = np.zeros(W.shape).astype(np.float32)
            P[np.ix_(labeledIndexes,labeledIndexes)] = np.linalg.inv( I - (1/1+mu)*(I-L) )[np.ix_(labeledIndexes,labeledIndexes)] 
            P[labeledIndexes,labeledIndexes] = 0
            P[np.ix_(labeledIndexes,labeledIndexes)] = P[np.ix_(labeledIndexes,labeledIndexes)]/np.sum(P[np.ix_(labeledIndexes,labeledIndexes)],axis=0,keepdims=False)
            PL = P[np.ix_(labeledIndexes,labeledIndexes)]
            
        W = scipy.sparse.csr_matrix(W)

                
        def divide_row_by_sum(e):
            e = _to_np(e)
            e = e / np.sum(e + 1e-100,axis=1, keepdims=True)
            return e
        
        PL = divide_row_by_sum(PL)
        
        import tensorflow as tf
        A = PL
        B = Y[labeledIndexes,:]
        PTP = np.transpose(A)@A
        PT = np.transpose(A)
        SIGMA = lambda:tf.linalg.tensor_diag(tf.clip_by_value(_SIGMA,0.0,tf.float32.max))
        C = lambda:tf.linalg.tensor_diag(_C)

        _SIGMA = tf.Variable(np.ones((PL.shape[0],), dtype=np.float32))
        
        _C = tf.Variable(_SIGMA)
        
        to_prob = lambda x: tf.nn.softmax(x,axis=1)
        xent = lambda y_, y: tf.reduce_mean(-tf.reduce_sum(y_ * tf.cast(tf.math.log(y+1e-06),tf.float32),axis=[1]))
        
        sq_loss = lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.square(y_-y),axis=[1]))
        norm_s = lambda: _SIGMA*tf.gather(tf.math.reciprocal_no_nan(tf.reduce_sum(to_prob(A@SIGMA()@B),axis=0)),tf.argmax(B,axis=1) )
        

        if which_loss ==  "xent":
            loss = lambda: xent(to_prob(A@SIGMA()@B),B)   + 10*tf.reduce_sum(tf.square( tf.reduce_mean(to_prob(A@SIGMA()@B),axis=0) - tf.reduce_mean(B,axis=0)))
        elif which_loss == "mse":
            loss = lambda: sq_loss(to_prob(A@SIGMA()@B),B)  #+ 1*tf.reduce_sum(tf.square( tf.reduce_mean(to_prob(A@SIGMA()@B),axis=0) - tf.reduce_mean(B,axis=0)))
            
            
        acc = lambda: 100.0*tf.math.reduce_mean(tf.cast(  tf.equal(tf.argmax(to_prob(A@SIGMA()@B),axis=1), tf.argmax(B,axis=1)), tf.float32) )
        

        
        #0.99 - 0.07466477900743484
        #0.9 - 0.0856625959277153
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.7)
        
        #for i in range(2000):
        #    opt.minimize(loss, [_C])
        #    print(loss().numpy())
        
        #for i in range(200):
        #    opt.minimize(loss, [_C])
        #    print(loss().numpy())

        np.set_printoptions(precision = 3)
        #raise ""
        
        #0.99 - 0.06267
        #0.9 - 0.06164

        for i in range(1000):
            opt.minimize(loss, [_SIGMA])
            #_SIGMA.assign(norm_s())
            print(loss().numpy())

        self.Fl = (lambda: to_prob(A@SIGMA()@B))().numpy()

        Y[labeledIndexes,:] = self.Fl

        
        return Y, labeledIndexes
        """
        
        Yl = Y[labeledIndexes,:]
        it_counter = 0
        loss  = np.inf
        LR = 0.1
        for i in range(1000000):
            grad_SIGMA = 2*np.transpose(C@A)@((C@A@SIGMA@B)-B)@np.transpose(B)
            grad_C = 2*(C@A@SIGMA@B-B)@(np.transpose(B)@np.transpose(SIGMA)@np.transpose(A))
            
            SIGMA -= LR*(np.diag(np.diagonal(grad_SIGMA)))
            C -= LR*(np.diag(np.diagonal(grad_C)))
            
            SIGMA =  np.maximum(SIGMA,np.zeros_like(SIGMA))
            new_loss = np.sum(np.square((C@A)@SIGMA@B - B))
            if new_loss > loss:
                LR *= 0.5
                it_counter += 1
                if it_counter == 10:
                    break
            else:
                it_counter = 0
                loss = new_loss
                print(new_loss)
        """
        
            
        return Y, labeledIndexes
        for _ in range(10):
            Yl = Y[labeledIndexes,:]
            PL_masked  = PL*(Yl@np.transpose(Yl) )
            
            
            labeled_ids = np.where(labeledIndexes)[0]
            for i, l_id in enumerate(labeled_ids):
                den = np.square(np.max(Y[l_id,:])) * np.sum(np.square(PL[:,i]))
                den += 1e-30
                num = np.sum(PL_masked[:,i])
                Y[l_id,:] *= (num/den)  
            
        
       
        return Y, labeledIndexes

            
        
        

    def fit (self,X,Y,labeledIndexes,W = None,hook=None):

        return self.LGCLVO(X, W, Y, labeledIndexes,lgc_iter=self.LGC_iter,\
                          hook=hook, mu=self.mu, which_loss=self.loss)
    
    def __init__(self, mu = 99.0, LGC_iter = 10000, loss="xent"):
        """ Constructor for the Automatic LGCLVO filter.
        
        Args:
            mu (float) :  a parameter determining the importance of the fitting term. Default is ``99.0``.
            LGC_iter (int) :  Number of steps when approximating the propagation matrix. Default is ``10000``. 
        """
        self.LGC_iter = LGC_iter
        self.mu = mu
        self.loss = loss