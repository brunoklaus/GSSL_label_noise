"""

A Neural Network approach I was previously using for some experiments. Ignore this for now.

"""
from gssl.classifiers.classifier import GSSLClassifier
import numpy as np
import gssl.graph.gssl_utils as gutils
import scipy.linalg as sp

import scipy.sparse
from settings import p_bar

import matplotlib.pyplot as plt
from input.dataset.cifar10 import INPUT_FOLDER

import time
import faiss
from faiss import normalize_L2
from output import plot_core, plots
import log.logger as LOG
import tensorflow as tf

def debug(msg):
    LOG.debug(msg,LOG.ll.CLASSIFIER)

class Accumulator():
    
    def __init__(self,tensor,name,init_val=0.0):
        self.var_accumulator = tf.Variable(init_val,name=name+"_accumulator")
        self.name = tensor.name
        self.var_examples = tf.Variable(0,name=name+"_examples",dtype=tf.int32)
        self.avg=  self.var_accumulator/tf.cast(self.var_examples,tf.float32)
        self.reset_op = tf.group(tf.assign(self.var_accumulator,init_val),
                                 tf.assign(self.var_examples,0))
                                 
        
        with tf.control_dependencies([tensor]):
            
                sh = 1 if len(tensor.shape) == 0 else tf.shape(tensor)[0]
                self.op = tf.group(tf.assign(self.var_accumulator,self.var_accumulator + tf.reduce_sum(tensor) ),
                                        tf.assign(self.var_examples,self.var_examples +sh ))
            
        self.summary = tf.summary.scalar(name, self.avg)
            

def cos_decay(init_val, EPOCH_VAR, rampdown_length):
    EPOCH_VAR = tf.where(tf.greater(EPOCH_VAR,rampdown_length),rampdown_length,EPOCH_VAR)
    return init_val * tf.cast(.5 * (tf.cos(np.pi * EPOCH_VAR / rampdown_length) + 1),tf.float32)


def get_S(W):
    from scipy import sparse
    from sklearn.preprocessing import normalize
    wsum = np.reshape(np.asarray(W.sum(axis=0)),(-1,) ) 
    d_sqrt = np.reciprocal(np.sqrt(wsum))
    d_sqrt[np.logical_not(np.isfinite(d_sqrt))] = 1
    d_sqrt = sparse.diags(d_sqrt).tocsr()
    debug(d_sqrt.shape)
    debug(W.shape)
    
    S = d_sqrt*W*d_sqrt
    return S


def get_S_fromtensor(W):
    wsum = tf.sparse.reduce_sum(W,axis=1)
    LOG.debug(wsum,LOG.ll.CLASSIFIER)
    wsum = tf.reshape(wsum,(-1,))
    d_sqrt = tf.reciprocal(tf.sqrt(wsum))
    d_sqrt = tf.where(tf.is_finite(d_sqrt),d_sqrt,tf.ones(shape=tf.shape(d_sqrt)))
    
    d_sqrt_i = tf.gather(d_sqrt,W.indices[:,0])
    d_sqrt_j = tf.gather(d_sqrt,W.indices[:,1])
    
    
    S = tf.sparse.SparseTensor(indices=W.indices,
                                    values=W.values * d_sqrt_i * d_sqrt_j,
                                    dense_shape=W._dense_shape)
    
    return S


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

def convert_sparse_matrix_to_sparse_tensor(X,var_values=False):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    indices = np.reshape(np.asarray(indices),(-1,2))
    
    return tf.SparseTensorValue(indices, np.reshape(np.asarray(coo.data).astype(np.float32),(-1,)), coo.shape)
    
    if not var_values:
        return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    else:
        W_values = tf.Variable(coo.data.astype(np.float32))
        return tf.SparseTensor(indices, tf.identity(W_values), coo.shape), W_values


    
def kl_divergence(self,p, q): 
    return tf.reduce_sum(p * tf.log(p/q))

def ent(Y):
    Y = row_normalize(tf.clip_by_value(Y,0.,1.))
    Y = Y + tf.random.uniform((),minval = 1e-08, maxval= 2*1e-08)
    Y = tf.clip_by_value(Y,0.,1.)
    return -tf.reduce_sum(Y*tf.log(Y),axis=1)/tf.log(tf.cast(tf.shape(Y)[1],tf.float32))

def xent(distr_1,distr_2):
        distr_2 = tf.clip_by_value(distr_2,1e-06,1.)
        xent = -tf.reduce_sum(distr_1 * tf.log(distr_2), 1)
        return xent  

class NNClassifier(GSSLClassifier):
    
    """ A NN classifier that was intended to optimize a labeled and unlabeled objective at the same time. Please ignore this for the time being
    
     """
        
    def _model(self,input_shape,out_size):
        from gssl.classifiers.nn import models
        with tf.name_scope("MODEL"): 
            if self.model_choice == "conv_large":
                return models.conv_large(input_shape, out_size)
            elif self.model_choice == "conv_small":
                return models.conv_small(input_shape, out_size)
            else:
                return models.linear(input_shape, out_size)
        
    ALPHA = 0.1
    LAMBDA = 0.5
    
    USE_UNLABELED = True
    RECALC_W = False
    
    
    SIGMA = tf.constant(0.04)

    def _sim_func(self,X1,X2,Y1= None,Y2 = None):
        x_shape = tf.shape(X1)
        x_shape = [x_shape[0],tf.reduce_prod(x_shape[1:])]
        X1 = tf.reshape(X1,x_shape)
        X2 = tf.reshape(X2,x_shape)
        
        
        with tf.name_scope('similarity_func'):
            #X1 = tf.nn.l2_normalize(X1,1)
            #X2 = tf.nn.l2_normalize(X2,1)
             
            #simm = tf.clip_by_value(tf.reduce_sum(X1*X2,axis=1),0.0,1.0)
            #simm = tf.pow(simm,3.0)
            D_x = tf.reduce_sum(tf.square(X1-X2),axis=1)
            simm =  tf.exp(-D_x/(2.0*tf.pow(self.SIGMA,2.0))) 
            
            if Y1 is None:
                return simm
            else:
                ent_simm = ent(tf.stop_gradient(Y2))*simm
                return ent_simm
    
  
    def _entropy_loss(self,Y1):
        
        #Y1 = row_normalize(Y1)
        #Y2 = row_normalize(Y2)
        n_cols = tf.cast(tf.shape(Y1)[1],tf.float32)
 
        e1 = tf.nn.softmax_cross_entropy_with_logits(logits=Y1,labels=tf.ones(tf.shape(Y1))/n_cols)
        

        e1 = ent(Y1)
        #e2 = xent(row_normalize(Y2),tf.ones(tf.shape(Y2))/n_cols)
        
        s = tf.reshape(tf.reduce_sum(Y1,axis=0),(-1,1))
        e3 = xent(row_normalize(s), tf.ones(tf.shape(Y1)[1])/n_cols)


        return  2*e3, e1
     
     

    def _similarity_loss(self,X1,X2,Y1,Y2,D1,D2,normalize_w = False):
       
        #Y2 = tf.stop_gradient(Y2)
        #D_y = tf.reduce_sum(tf.square(Y1/tf.sqrt(repeat(D1,tf.shape(Y1)[1]))-\
        #                              Y2/tf.sqrt(repeat(D2,tf.shape(Y2)[1]))) ,axis=1)
        #D_y = tf.square(norm(Y1/tf.sqrt(repeat(D1,tf.shape(Y1)[1]))-\
        #                              Y2/tf.sqrt(repeat(D2,tf.shape(Y2)[1]))
        #                             ))
        #D_y = tf.nn.softmax_cross_entropy_with_logits(logits=Y1,labels=tf.nn.softmax(Y2))

            
        D_y = tf.reduce_sum(tf.square(Y1-Y2),axis=1)  
        
        Y2 = tf.stop_gradient(Y2)
        
        SIMs = (1-ent(Y2))*self._sim_func(X1, X2)
        if normalize_w:
            SIMs /= tf.reduce_sum(SIMs)
            SIMs *= tf.cast(tf.shape(SIMs)[0],tf.float32)
        
        #SIMs = tf.ones_like(SIMs)
        UNIQUE_EXAMPLES = tf.cast(tf.shape(X1)[0],tf.float32)/ self.AVG_NEIGHBORS
        
        sim_loss = D_y * SIMs 
        
        
        sim_loss = sim_loss #+ self.ENTROPY_REG * tf.reduce_sum(self._entropy_loss(Y1,Y2),axis=1)
        
        
        return sim_loss
    def labeled_gen(self):
        
        where_labeled = np.where(self.labeledIndexes)[0].astype(np.int32)
        perm = np.random.permutation(where_labeled.shape[0])
        
        
        i = 0
        while i < where_labeled.shape[0]:
            nxt_i = min(i+self.BATCH_SIZE,where_labeled.shape[0]) 
            ids = where_labeled[perm[i:nxt_i]]
            i = nxt_i
            yield self.X[ids,:], self.Y[ids,:], ids
            
    def pred_gen(self):
        i = 0
        while i < self.X.shape[0]:
            nxt_i = min(i+self.BATCH_SIZE,self.X.shape[0]) 
            ids = np.arange(i,nxt_i)
            i = nxt_i

            yield self.X[ids,:], self.Y[ids,:],  ids
   
                
    def unlabeled_gen(self):
            is_cifar_test = [False]*50000 + [True]*10000
            #is_cifar_test = [True] * self.labeledIndexes.shape[0]
            where_unlabeled = np.where(np.logical_and(is_cifar_test,np.logical_not(self.labeledIndexes)))[0].astype(np.int32)
            perm = np.random.permutation(where_unlabeled.shape[0])
            
            i = 0
            while i < where_unlabeled.shape[0]:
                nxt_i = min(i+self.BATCH_SIZE,where_unlabeled.shape[0]) 
                ids = where_unlabeled[perm[i:nxt_i]]
                i = nxt_i
                assert np.max(self.Y[ids,:]) > 0
                yield self.X[ids,:], self.Y_true[ids,:], ids
                
                
    
    def random_gen(self):
            
            
            M= np.max(self.X) 
            m = np.min(self.X) 
             
            while True:
                X_random = np.random.uniform(low=m,high=M,size=(self.BATCH_SIZE,self.X.shape[1]))
                yield (X_random,)
                
    eval_get_data = None
    def evaluate_simfunc(self,W_sparse_vals):
        if self.eval_get_data is None:
            self.eval_ids_i = tf.placeholder(tf.int32,shape=[None])
            self.eval_ids_j = tf.placeholder(tf.int32,shape=[None])
            self.eval_X_descr = tf.placeholder(tf.float32)
            
            
            self.eval_get_data = self._sim_func(X1=tf.gather(self.eval_X_descr,self.eval_ids_i) ,
                                            X2=tf.gather(self.eval_X_descr,self.eval_ids_j))
 
            
        ids_i = np.reshape(np.asarray(W_sparse_vals.indices[:,0]),(-1,))
        ids_j = np.reshape(np.asarray(W_sparse_vals.indices[:,1]),(-1,))
        

        
        data = np.zeros(ids_i.shape)
        i = 0
        
        while i < ids_i.shape[0]:
            nxt_i = min(i+self.X.shape[0],ids_i.shape[0])
            data[i:nxt_i] = self.sess.run(self.eval_get_data,
                                          feed_dict={self.W.indices:W_sparse_vals.indices,
                                                     self.W.values:W_sparse_vals.values,
                                                     self.eval_ids_i:ids_i[i:nxt_i],
                                                     self.eval_ids_j:ids_j[i:nxt_i],
                                                     self.eval_X_descr:self.X_descr})
            i = nxt_i
        
        
        updated_W = tf.SparseTensorValue(W_sparse_vals.indices,data,W_sparse_vals.dense_shape)
        return updated_W
        
    def unlabeled_pairs_gen(self):
            W = self.eval_W
            D = self.eval_D
            
            W_row = W.row
            W_col = W.col
            W_data = W.data
            NUM_PAIRS = W_row.shape[0]
            

            
            r_perm = np.random.rand(W.shape[0])
            
            tup_list = []
            for i in range(NUM_PAIRS):
                tup_list.append((r_perm[W_row[i]],W_col[i],i))
            tup_list = sorted(tup_list)
            
            perm = [x[2] for x in tup_list]
            
            
            
            i = 0
            while i < NUM_PAIRS:
                nxt_i = min(i+self.BATCH_SIZE,NUM_PAIRS) 
                ids = perm[i:nxt_i]
                i = nxt_i
                
                yield self.X[W_row[ids],:], self.X[W_col[ids],:], D[W_row[ids]], D[W_col[ids]],\
                    W_row[ids],W_col[ids]
  
    
    def build_graph(self,X,k):
        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res,d,flat_config)   # build the index

        #normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        LOG.debug('kNN Search done in %d seconds'.format(elapsed),LOG.ll.CLASSIFIER)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T
        return W
    
    @GSSLClassifier.autohooks
    def __NN(self,X,W,Y,labeledIndexes,hook=None):
        if self.image_shape is None:
            input_shape = X.shape[1:]
            self.X = X.astype(np.float32)
        else:
            input_shape = self.image_shape
            self.X = np.reshape(X.astype(np.float32),tuple([-1] + self.image_shape))
        out_size = Y.shape[1]


        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        

        
        """ Get number of labeled and unlabeled instances, and avg number of neighbors """
        N_l = np.sum(labeledIndexes)
        N_u = np.sum(np.logical_not(labeledIndexes))
        M = len(W.data)        
        self.AVG_NEIGHBORS = len(W.data)/W.shape[0]
        debug("Number of  labeled examples : {}".format(N_l))
        
        
        """ Convert W to coo. Makes it easy to set the value of the weights """
        self.W  = tf.sparse_placeholder(tf.float32,shape=W.shape)
        calc_D = tf.sparse.reduce_sum_sparse(self.W,axis=1).values
        
        
        
        del W
        
        self.Y = Y.astype(np.float32)
        
        
        
        
        self.labeledIndexes = labeledIndexes
        
        

        
        """ Get degree Matrix """
        self.D =  tf.sparse.reduce_sum(self.W,axis=1)

      
        """ F_0 is a copy of the label matrix, but we erase the information on labeled Indexes """
        self.F_0 = np.copy(Y).astype(np.float32) 
        self.F_0[np.logical_not(labeledIndexes),:] = 0.0
        
        
        
        
        
        
        

        """
            CREATE S - Needed for LGC propagation
        """
        self.S =  get_S_fromtensor(self.W)#convert_sparse_matrix_to_sparse_tensor(get_S(self.W))
    

        """
        CREATE F variable
        """
        self.F = tf.Variable(np.copy(self.F_0).astype(np.float32),name="F")
        self.F_0 = tf.Variable(self.F_0)
        
        
        """
        CREATE w_i vector: holds (1 - entropy) for each F row
        """
        tmp = np.ones((self.X.shape[0],))
        tmp[labeledIndexes] = 0.0
        
        self.F_ent = tf.Variable(np.copy(tmp).astype(np.float32),name="F_ent")
        debug(ent(self.F).shape)
        update_F_ent = tf.assign(self.F_ent,ent(self.F)) #run this op to update entr
        
        """
        CREATE class_freq vector: holds frequency of each class in argmax
        """
        self.class_freq = tf.Variable(np.asarray([self.X.shape[0]/out_size]*out_size).astype(np.float32),name="class_freq")
        calc_class_freq = \
        tf.map_fn(lambda i: tf.reduce_sum(
                                tf.cast(\
                                    tf.equal(tf.argmax(self.F,axis=1),i),
                                tf.int64)),
                            tf.range(out_size,dtype=tf.int64))
        debug(calc_class_freq.shape)
        update_class_freq = tf.assign(self.class_freq,tf.cast(calc_class_freq,tf.float32) )
        
        
        
        
        
        
            
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        """
            Possibly augment image.
        """ 
        if self.AUGMENT:

            CROPPED_SIZE = int(np.round(input_shape[0] * 30/32))
            def augment(x,noise=False):
                degrees = tf.random.uniform((),-7.5,7.5)
                x = tf.reshape(x,input_shape)
                
                if noise:
                    x = x + tf.random.normal(x.shape,stddev=0.15)
                x = tf.contrib.image.rotate(x, degrees * np.math.pi / 180, interpolation='BILINEAR')
                x = tf.image.random_flip_left_right(tf.image.random_crop(x,(CROPPED_SIZE,CROPPED_SIZE,3)))
                
                return x
            maybe_cropped_input_shape = (CROPPED_SIZE,CROPPED_SIZE,3)
        else:
            maybe_cropped_input_shape = input_shape
            def augment(x,noise=False):
                return x
        
        """ 
            Define Iterators and datasets.
        """   
        # Iterator has to have same output types across all Datasets to be used
        

        with tf.name_scope('iterator'):
            iterator_l = tf.data.Iterator.from_structure((tf.float32,tf.float32, tf.int32),\
                                                       (tuple([None] + list(maybe_cropped_input_shape)),(None,self.Y.shape[1]),
                                                        tuple([None,1])
                                                        )
        
                                            )
            iterator_ul = tf.data.Iterator.from_structure(tuple([tf.float32]*4 + [tf.int32]*2),\
                                                       (tuple([None] + list(maybe_cropped_input_shape)),
                                                        tuple([None] + list(maybe_cropped_input_shape)),
                                                        tuple([None]),tuple([None]),
                                                        tuple([None,1]),tuple([None,1])
                                                        )
                                                       )
            iterator_R = tf.data.Iterator.from_structure(output_types=(tf.float32,),
                                                         output_shapes=(tuple([None] + list(maybe_cropped_input_shape)),) )
                                                       
            
        nxt_X, nxt_Y, nxt_F = iterator_l.get_next()
        nxt_X1, nxt_X2, nxt_D1, nxt_D2, nxt_F1, nxt_F2 = iterator_ul.get_next()
        
        nxt_XR =  iterator_R.get_next()[0]
        

        with tf.name_scope('dataset'):
            ds_l_train = tf.data.Dataset.from_generator(self.labeled_gen, output_types=(tf.float32,tf.float32,tf.int32))\
                        .map(lambda x,y,f: (tf.map_fn(lambda img: augment(img,noise=True),x),y,f)).repeat((self.W.shape[0])//N_l)
            ds_ul_pairs = tf.data.Dataset.from_generator(self.unlabeled_pairs_gen,
                                                         output_types=tuple([tf.float32]*4 + [tf.int32]*2))\
                        .map(lambda x1,x2,d1,d2,f1,f2: (tf.map_fn(lambda img: augment(img,noise=True),x1),
                                              tf.map_fn(lambda img: augment(img,noise=True),x2),d1,d2,f1,f2))
            ds_R = tf.data.Dataset.from_generator(self.random_gen,
                                                         output_types=(tf.float32,))\
                        .map(lambda x1: (tf.map_fn(lambda img: augment(img,noise=True),x1),))
                        
            ds_l_val = tf.data.Dataset.from_generator(self.unlabeled_gen, output_types=(tf.float32,tf.float32,tf.int32))\
                        .map(lambda x,y,f: (tf.map_fn(lambda img: augment(img, noise=False),x),y,f))
                    
            ds_pred = tf.data.Dataset.from_generator(self.pred_gen, output_types=(tf.float32,tf.float32,tf.int32))\
                        .map(lambda x,y,f: (tf.map_fn(lambda img: augment(img,noise=False),x),y,f))
        
        
        """ Define the tensors corresponding to model predictions. """
        model = self._model(maybe_cropped_input_shape,out_size)
        
        m_temp = model(nxt_X)
        
        model_l =  m_temp[0]
        descr_l =  m_temp[1:] + [model_l]
        descr_l[:-2] = [tf.reduce_mean(x, reduction_indices=[1, 2]) for x in descr_l[:-2]]
        model_ul_1 =  model(nxt_X1)[0]
        model_ul_2 =  model(nxt_X2)[0]
        
        model_ul_1 = tf.nn.sigmoid(model_ul_1)
        model_ul_2 = tf.nn.sigmoid(model_ul_2)
        
        
        #w = repeat(tf.random_uniform(shape=(tf.shape(nxt_X2)[0],),minval=0.0,maxval=1.0),tf.shape(nxt_X2)[1])
        model_ul_R =  tf.nn.sigmoid(model(nxt_XR)[0])
        model_ul_1_F = tf.clip_by_value(gather(nxt_F1,self.F),0.,1.)
        model_ul_2_F =  tf.clip_by_value(gather(nxt_F2,self.F),0.,1.)
        model_l_F =  tf.clip_by_value(gather(nxt_F,self.F),0.,1.)
        model_pred = model_l
        """ As well as row_normalized versions """  
        norm_pred = tf.nn.softmax(model_pred)
        F_norm = row_normalize(self.F)
        

        EPOCH = tf.Variable(0)
        
        
        
        
        
        
        """ 
            Define loss ops,summaries, and optimizer.
        """
        with tf.name_scope('loss'):
            with tf.name_scope('l_loss_F'):
                l_loss_F = tf.reduce_sum(tf.square(model_l_F-nxt_Y),axis=1) 
                mean_l_loss_F = tf.reduce_mean(l_loss_F) 
                adj_l_loss_F = N_l * l_loss_F
                
            
            with tf.name_scope('l_loss'):
                l_loss = tf.losses.softmax_cross_entropy(logits=model_l,onehot_labels=nxt_Y)
                mean_l_loss = tf.reduce_mean(l_loss) 
            
                
               
                
            
            with tf.name_scope('ul_loss_F'):
                ul_loss_F = self._similarity_loss(nxt_X1, nxt_X2, model_ul_1_F,model_ul_2_F,nxt_D1,nxt_D2)
                mean_ul_loss_F = tf.reduce_mean(ul_loss_F)
                adj_ul_loss_F = M * mean_ul_loss_F
                
            with tf.name_scope('ul_loss'):            
                
                
                cf = tf.gather(tf.reshape(self.class_freq,(-1,1)),tf.argmax(model_ul_2_F,axis=1))
                fent = tf.gather(tf.reshape(self.F_ent,(-1,1)),nxt_F2,axis=0)
                expected_cf = self.X.shape[0] / out_size
                
                example_w = (1.0-fent)*tf.reciprocal(cf/expected_cf)
                example_w = tf.reshape(example_w,(-1,))
                example_w = tf.stop_gradient(example_w)
                
                
                ul_loss = tf.reduce_sum( repeat(example_w,out_size)*\
                                        tf.square(tf.one_hot(tf.argmax(model_ul_2_F,1),depth=out_size) - model_ul_2),axis=1)
                mean_ul_loss = tf.reduce_mean(ul_loss) 
                
                
            
            with tf.name_scope('pull_down'):
                is_unlabeled = 1.0 - tf.cast(tf.gather(tf.constant(self.labeledIndexes),nxt_F1),tf.float32)
                is_unlabeled = tf.reshape(is_unlabeled,(-1,))
                pull_down_F =  tf.reduce_sum(tf.square(model_ul_2_F),axis=1) * is_unlabeled 
                mean_pull_down_F = pull_down_F / tf.reduce_sum(is_unlabeled)
                adj_pull_down_F = mean_pull_down_F * N_u
                
                
            with tf.name_scope('total_loss_F'):                
                total_loss_F = (mean_l_loss_F + mean_pull_down_F* (N_u/N_l)) * (1 - self.ALPHA) + mean_ul_loss_F *  (M/N_l) *  self.ALPHA 
            with tf.name_scope('total_loss_NN'):
                FINAL_DECAY = tf.constant(1/100.0,dtype=tf.float32)
                decay = tf.train.exponential_decay(learning_rate=1.0,
                                   global_step = EPOCH,
                                   decay_steps=30,
                                   decay_rate=FINAL_DECAY,
                                   staircase=False)
                decay = tf.where(tf.less(decay,FINAL_DECAY),FINAL_DECAY,decay)
                
                
                #total_loss_NN = \
                #    10.0*(mean_l_loss - tf.reduce_mean(entr_loss_A)*decay + \
                #            FINAL_DECAY/decay  * self.ENTROPY_REG * mean_entr_loss)* (1 - self.LAMBDA)
                if self.USE_UNLABELED:
                    total_loss_NN = mean_l_loss * (1 - self.LAMBDA)  +  tf.where(EPOCH < 11, 0.0,
                                                                                  mean_ul_loss)  * self.LAMBDA              
                else:
                    total_loss_NN = mean_l_loss
        
            with tf.name_scope('accuracy'):
                l_acc_op  = tf.cast(tf.equal(tf.argmax(model_l,1),tf.argmax(nxt_Y,1)),tf.float32)
            with tf.name_scope('accuracy_F'):
                l_acc_F_op  = tf.cast(tf.equal(tf.argmax(model_l_F,1),tf.argmax(nxt_Y,1)),tf.float32)
        
        
        """
            Set up ACCUMULATORS
        """
        self.accumulators_train = []
        self.accumulators_val = []
        self.accumulators = []
        
        
        from os import  path  as path
        if self.USE_UNLABELED:
            with tf.name_scope("Accumulator_F"):
                for tensor in [l_acc_F_op, total_loss_F]:#,l_loss_F,adj_pull_down_F,adj_ul_loss_F]:
                    acm = Accumulator(tensor=tensor,name=path.dirname(tensor.name))
                    self.accumulators.append(acm)
                    self.accumulators_train.append(acm)
                    if tensor in [l_acc_F_op]:
                        self.accumulators_val.append(acm)
                
                        
        with tf.name_scope("Accumulator"):
            var_list = [l_loss,ul_loss,l_acc_op,total_loss_NN] if self.USE_UNLABELED else [l_loss,l_acc_op]
            for tensor in var_list:
                acm = Accumulator(tensor=tensor,name=path.dirname(tensor.name))
                self.accumulators.append(acm)
                self.accumulators_train.append(acm)
                self.accumulators_val.append(acm)
                
        
        
            
        
                
        acm_summaries = {"train": tf.summary.merge([x.summary for x in self.accumulators_train]),
                       "val":tf.summary.merge([x.summary for x in self.accumulators_val])}
        
        acm_reset = tf.group(*[x.reset_op for x in self.accumulators])
        acm_ops = tf.group(*[x.op for x in self.accumulators])
        
        decayed_lr = tf.train.exponential_decay(learning_rate=3*1e-04,
                                   global_step = EPOCH,
                                   decay_steps=20,
                                   decay_rate=0.1,
                                   staircase=False)
        decayed_lr = tf.where(tf.less(decayed_lr,1e-04),1e-04,decayed_lr)
                
        
        optim_F = tf.train.AdadeltaOptimizer(learning_rate=0.5)
        optim = tf.train.AdamOptimizer(learning_rate= 3*1e-04)
        if True:
            grads,vars= zip(*optim.compute_gradients(total_loss_NN))
            get_g_NN = [(g,v) for g,v in zip(grads,vars) if v.name and v.name.startswith("MODEL")]
            debug(get_g_NN)
            
            get_g_F = optim_F.compute_gradients(total_loss_F,[self.F])
            #opt_2 = optim_F.apply_gradients(get_g_F, global_step)
            UPDATE_COUNTER = tf.Variable(0)
            TOTAL_ITER = 200
            def update_F():
                i = tf.constant(0)
                c = lambda i,F: tf.less(i, TOTAL_ITER)
                b = lambda i,F: (tf.add(i, 1),(1 - self.ALPHA)*self.F_0 + self.ALPHA*tf.sparse.matmul(self.S,F))
                r = tf.while_loop(c, b, [i,self.F_0])
                return r
            
            def get_P():
                i = tf.constant(0)
                c = lambda i,F: tf.less(i, TOTAL_ITER)
                b = lambda i,F: (tf.add(i, 1),(1 - self.ALPHA)*self.F_0 + self.ALPHA*tf.sparse.matmul(self.S,F))
                r = tf.while_loop(c, b, [i,self.F_0])
            
            
                return r
            def _to_dense(sparse_tensor):
                return tf.sparse_add(tf.zeros(sparse_tensor._dense_shape), sparse_tensor)
            
            def F_cost(F):
                F = (1 - self.ALPHA) * F
                lb = tf.constant(np.where(self.labeledIndexes)[0])
                minus_S = tf.sparse.SparseTensor(indices=self.S.indices,
                                    values=-1.0 * self.S.values,
                                    dense_shape=self.S._dense_shape)
                temp  = tf.sparse.add(tf.sparse.eye(num_rows=self.X.shape[0],dtype=tf.float32),
                                      minus_S)
                                      
                smoothness_criterion = tf.trace(tf.matmul(tf.transpose(F),tf.sparse.matmul(temp,F)))
                fit_criterion = tf.square(self.F_0 - F) 
                return  (1-self.ALPHA)/self.ALPHA * fit_criterion + smoothness_criterion 
            """
            def no_update_F():
                return tf.identity(self.F)
            
            def maybe_update_F():
                update_counter = tf.assign(UPDATE_COUNTER, UPDATE_COUNTER + 1)
                with tf.control_dependencies([update_counter]):
                    cnd = tf.cond(tf.equal(tf.mod(UPDATE_COUNTER,100),0),no_update_F,no_update_F)
                
                return cnd 
            """
            opt_2 = optim.apply_gradients(get_g_NN, global_step)
            calc_F = update_F()
            opt_F = tf.assign(self.F,calc_F[1])
            
            cost = F_cost(calc_F[1])
           
            grad_F0 = tf.gradients(cost, self.F_0)
            reset_F = tf.assign(self.F,self.F_0)

        
        if False:
            grads,vars= zip(*optim.compute_gradients(total_loss_NN))
            
            grads,vars = [(g,v) for g,v in zip(grads,vars) if v.name and v.name== "F:0"][0]
            
            F_prime_update = self.F_prime.scatter_update(grads)
        
        
        
        """
            Run variable initializers 
        """
        global_var_init=tf.global_variables_initializer()
        local_var_init = tf.local_variables_initializer()
        
        
        




        l_iterator_train = iterator_l.make_initializer(ds_l_train)
        ul_iterator_train = iterator_ul.make_initializer(ds_ul_pairs)
        debug(ds_R)
        r_iterator_train = iterator_R.make_initializer(ds_R)
        
        l_iterator_val = iterator_l.make_initializer(ds_l_val)
        iterator_pred = iterator_l.make_initializer(ds_pred)
        
        


        l_shape_op = tf.shape(nxt_X)
        ul_shape_op = tf.shape(nxt_X1)
        
        no_op = tf.no_op()
        
        """ 
        Begin Session...
        """
        with sess.as_default():
            sess.run([global_var_init,local_var_init])
            
        
            
            """ Initialize writers"""
            from os import path
            MODEL_PATH = path.join(INPUT_FOLDER,"NN_models","model_l")
            
            import shutil
            if path.isdir(MODEL_PATH):
                shutil.rmtree(MODEL_PATH)
            
            debug(path.join(MODEL_PATH,"labeled"))
            
            train_Writer = tf.summary.FileWriter(path.join(MODEL_PATH,"train"), sess.graph)
            val_Writer = tf.summary.FileWriter(path.join(MODEL_PATH,"val"), sess.graph)
            
            
            self.F_loss = None
            
            temp_F = tf.placeholder(tf.float32)
            temp_argmax = tf.argmax(temp_F,axis=1)
            temp_acc = \
            tf.reduce_mean(tf.cast(tf.equal(temp_argmax[50000:],tf.argmax(self.Y_true[50000:,:],axis=1)),tf.float32))
            
            def train_loop(i,mode='train'):
                
                EPOCH.load(i)
                """ 
                OBTAIN DESCRIPTORS
                """
                self.ALL_DESCR = list()
                
                for descr_id,descr in enumerate(descr_l):     
                    sess.run(iterator_pred)
                    sh = [self.X.shape[0]]+list(sess.run(tf.shape(descr_l[descr_id])[1:]))
                    debug(sh)
                    self.ALL_DESCR.append(np.zeros(shape=sh, dtype=np.float32))                
                    sess.run(iterator_pred) #It runs again here, as getting the shape seems to consume the first batch
                with p_bar(self.X.shape[0],"Training - EPOCH {} -mode={}...".format(i,"DESCRIPTORS")) as p:
                    example_counter = 0
                    while True:
                        try:
                                
                                desc, ids = \
                                    sess.run([descr_l, nxt_F])
                                
                                
                                example_counter += ids.shape[0]
                                for descr_id,descr in enumerate(descr_l):
                                    self.ALL_DESCR[descr_id][ids,:] = desc[descr_id]
                                p.update(example_counter)
                                
                                    
                        except tf.errors.OutOfRangeError:
                            break
                descr_acc = []
                ensemble_pred =[]
                
                if i % 10 == 0:
                    descr_id_list = [len(descr_l)-2]
                else:
                    descr_id_list = [len(descr_l)-2]#[len(descr_l)-2]
                for descr_id in descr_id_list:
                    self.X_descr = self.ALL_DESCR[descr_id]
                    """ 
                    OBTAIN AFFINITY MATRIX
                    """
                    debug("Building graph...")
                    self.sess = sess
                    W_sparse_tensor_values = self.evaluate_simfunc(\
                                               convert_sparse_matrix_to_sparse_tensor(self.build_graph(self.X_descr, k=10))
                                               )
                    W_feed_dict = {self.W.values : W_sparse_tensor_values.values,
                                   self.W.indices : W_sparse_tensor_values.indices
                                   }
                    
                    
                    
                    
                    
                    
                    """
                        Label PROP
                    """
                    debug("Running Label Prop...")
                    _,gf0 = sess.run([opt_F,tf.convert_to_tensor(grad_F0[0])],feed_dict=W_feed_dict)
                    
                    gf0[np.logical_not(self.labeledIndexes),:] = np.inf
                    gf0_ids = np.argsort(gf0,axis=None)
                    noisy_labels = np.logical_and(labeledIndexes,
                                                           np.argmax(self.Y_noisy,axis=1) != np.argmax(self.Y_true,axis=1))
                    
                    import scipy.stats
                    debug(scipy.stats.describe(gf0) )
                    debug(np.round(gf0[labeledIndexes,:],3) )
                    debug(np.round(gf0[noisy_labels,:],3) )
                    
                    num_noisy = np.sum(noisy_labels)
                    detected_labels = np.asarray([False] * self.X.shape[0])                    
                    num_detected = 0
                    ptr = 0
                    debug(gf0_ids)
                    gf0_ids = gf0_ids.tolist()
                    gf0_ids_i = [x // out_size for x in gf0_ids]
                    while num_detected < num_noisy:
                        id = gf0_ids_i.pop(0)
                        if detected_labels[id] == False:
                            detected_labels[id] = True
                            num_detected += 1
                    assert np.sum(noisy_labels) == np.sum(detected_labels)
                    
                    recall = np.sum(np.logical_and(noisy_labels,detected_labels))/np.sum(noisy_labels)
                    
                    
                    debug("%Recall:{}".format(recall))
                    
                    """ 
                    VISUALIZE TSNE
                    """
                    if mode == 'train':
                        from sklearn.preprocessing import normalize
                        from MulticoreTSNE import MulticoreTSNE as TSNE
                        from sklearn.decomposition import PCA
                        lb = np.copy(labeledIndexes)
                        lb[0:10000] = True
                        embeddings = TSNE(n_jobs=16,random_state=123456).fit_transform(normalize(self.X_descr)[np.where(lb)[0],:])
                        debug("Creating TSNE...Done!")
                        F = sess.run(self.F)
                        self.eval_W = scipy.sparse.coo_matrix((W_sparse_tensor_values.values,
                                       (W_sparse_tensor_values.indices[:,0],
                                        W_sparse_tensor_values.indices[:,1])), shape = W_sparse_tensor_values.dense_shape)

                        W = self.eval_W
                        noisy_labels = np.logical_and(labeledIndexes,
                                                               np.argmax(self.Y_noisy,axis=1) != np.argmax(self.Y_true,axis=1))
                        
                        plots.plot_all_indexes(embeddings, self.Y_noisy[np.where(lb)[0],:], noisy_labels[np.where(lb)[0]], W=None,
                                                   title="EPOCH_"+str(sess.run(EPOCH)),
                                                   plot_filepath="./TSNE_NOISY_DESCR_{}_EP_{}".format(descr_id,i)+".png",
                                                                               mode="discrete")
                        plots.plot_all_indexes(embeddings, self.Y_true[np.where(lb)[0],:], noisy_labels[np.where(lb)[0]], W=None,
                                                   title="EPOCH_"+str(sess.run(EPOCH)),
                                                   plot_filepath="./TSNE_TRUE_DESCR_{}_EP_{}".format(descr_id,i)+".png",
                                                                               mode="discrete")
                        """
                        plots.plot_all_indexes(self.X, F, labeledIndexes, W=W,
                                                                       title="EPOCH_"+str(sess.run(EPOCH)),
                                                                       plot_filepath="./TSNE_TRUE_DESCR_{}_EP_{}".format(descr_id,i)+".png",
                                                                                                   mode="discrete")
                        plots.plot_all_indexes(self.X,F, noisy_labels, W=W,
                                                                       title="EPOCH_"+str(sess.run(EPOCH)),
                                                                       plot_filepath="./TSNE_NOISY_DESCR_{}_EP_{}".format(descr_id,i)+".png",
                                                                                                   mode="discrete")
                        plots.plot_all_indexes(self.X, F, detected_labels, W=W,
                                                                       title="EPOCH_"+str(sess.run(EPOCH)),
                                                                       plot_filepath="./TSNE_DETECTED_DESCR_{}_EP_{}".format(descr_id,i)+".png",
                                                                                                mode="discrete")
                        """
                    
                    if i == 5:
                        raise ""
                    
                    #debug(gf0)
                    
                    debug("Running Label Prop...Done!")
                    
                    
                    
                    
                    #debug(stats.describe(sess.run(self.F_ent)) )
                    sess.run(update_class_freq)
                    sess.run(update_F_ent)
                    #debug(stats.describe(sess.run(self.F_ent)) )
                    
                    F_acc = sess.run(temp_acc,feed_dict={temp_F:sess.run(self.F)})
                    debug("Label Prop acc : {}".format(F_acc))
                    descr_acc.append(F_acc)
                    
                    #Adicionamos pred 
                    F = sess.run(tf.one_hot(indices=temp_argmax,depth=out_size),feed_dict={temp_F:sess.run(self.F)})
                    debug(F)
                    ensemble_pred.append(F)
                    
                    
                    
                    
                    
                
                def prediction_ensemble(pred_list):
                    assert len(pred_list) > 0
                    F = pred_list[0]
                    for i in range(1,len(pred_list)):
                        F += pred_list[i]
                    return F.astype(np.float32)
                    
                
                ensemble_acc = []
                for pred_id in range(len(ensemble_pred)):
                    p_ens = prediction_ensemble(ensemble_pred[pred_id:])
                    ensemble_acc.append(sess.run(temp_acc,feed_dict={temp_F:p_ens}))

                    
                debug(descr_acc)
                debug(ensemble_acc)
                """ Get W as scipy matrix """
                self.eval_D = sess.run(calc_D,feed_dict=W_feed_dict)
                self.eval_W = scipy.sparse.coo_matrix((W_sparse_tensor_values.values,
                                                       (W_sparse_tensor_values.indices[:,0],
                                                        W_sparse_tensor_values.indices[:,1])), shape = W_sparse_tensor_values.dense_shape)
                del W_sparse_tensor_values
                debug("Building graph...Done!")
                
                
                """ 
                debug STATS
                """
                from scipy import stats
                debug("Stats about W values:")
                debug(stats.describe(self.eval_W.data))
                
                
                
                """
                    TRAIN 
                """
                writer = train_Writer if mode == 'train' else val_Writer
                if mode == 'train':
                    sess.run(l_iterator_train)
                else:
                    sess.run(l_iterator_val)
                sess.run(ul_iterator_train)
                sess.run(r_iterator_train)
                
                del self.ALL_DESCR
                """ RESET ACCUMULATORS """
                sess.run(acm_reset)
                
                
                    
                
                sz = (self.Y.shape[0]//N_l) * N_l\
                if mode == 'train'  else np.sum(np.logical_not(self.labeledIndexes.astype(np.int8)))
                with p_bar(sz,"Training - EPOCH {} -mode={}...".format(i,mode)) as p:
                    
                    
                    example_counter = 0
                    F_loss = np.zeros(self.X.shape[0])
                    
            
                    
                    while True:
                        
                        try:
                            
                            p.update(example_counter)
                            _, optt, l_shape = \
                                sess.run([acm_ops,
                                          opt_2 if mode == 'train' else nxt_Y,
                                          l_shape_op])
                            
                            
                            example_counter += l_shape[0]
                            
                                
                        except tf.errors.OutOfRangeError:
                            break
                            """for acm in self.accumulators:
                                debug("AVG {}:{}".format(acm.name,acm.avg))
                            break
                            """

                    if mode == 'val':
                        for acm in self.accumulators:
                            debug("AVG {}:{}".format(acm.name,sess.run(acm.avg) ))
                            
                        
                    writer.add_summary(sess.run(acm_summaries[mode]),sess.run(global_step))
                    if i % 10 == 0:
                        writer.flush()
                    self.F_loss = F_loss
            
                   
            for i in range(self.NUM_EPOCHS):
                train_loop(i, 'train')    
                train_loop(i, 'val')
                debug(i)
                if self.X.shape[0] > 10000:
                    continue
                if (not hook is None and i % 25 == 0) or i == self.NUM_EPOCHS-1:                        
                    """
                        PRED 
                    """
                    #hook._step(plot_id=1,step=i,X=self.X,W=None,Y=np.copy(self.F_loss),labeledIndexes=self.labeledIndexes )
                    
                    """ ADD dummy grid points """
                    old_X = self.X
                    x = np.linspace(np.min(self.X[:,0]), np.max(self.X[:,0]), 100)
                    y = np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), 100)
                    grid = np.zeros((x.shape[0]**2,2))
                    for i_1 in range(100):
                        for j_1 in range(100):
                            grid[i_1*100+j_1,0] = x[i_1]
                            grid[i_1*100+j_1,1] = y[j_1]
                        
                    old_Y = self.Y
                    self.X = np.concatenate((grid,old_X),axis=0)
                    self.Y = np.zeros((grid.shape[0] + old_X.shape[0],self.Y.shape[1]))
                    
                    
                    F = -1.0*np.ones((self.X.shape[0],self.Y.shape[1]))
                    with p_bar(F.shape[0],"Training - EPOCH {} -mode={}...".format(i,"pred")) as p:
                        sess.run(iterator_pred) 
                        example_counter = 0
                        while example_counter != F.shape[0]:
                                p.update(example_counter)
                                pred, x_shape, ids = sess.run([norm_pred,l_shape_op,nxt_F])
                                F[ids,:] = pred
                                example_counter += x_shape[0]
                    """ Restore / Cleanup """
                    F[0,0] = 0.0
                    F[9999,0] = 1.0
                    if not hook is None:
                        hook._step(plot_id=0,step=i,X=self.X,W=None,Y=np.copy(F[:,0]),labeledIndexes=[False]*10000 + list(self.labeledIndexes) )
                    self.X = old_X
                    self.Y = old_Y
                    
                    
                    """ Get value of F """
                    F = np.copy(sess.run(F_norm))
                    F[np.argmin(F[:,0]),0] = 0.0
                    F[np.argmax(F[:,0]),0] = 1.0
                    if not hook is None:
                        hook._step(plot_id=1,step=i,X=self.X,W=self.eval_W,Y=np.copy(F[:,0]),labeledIndexes=self.labeledIndexes) 
                        
            
            """ Final Prediction """            
            F = -1.0*np.ones((self.X.shape[0],self.Y.shape[1]))
            with p_bar(F.shape[0],"Training - EPOCH {} -mode={}...".format(i,"pred")) as p:
                sess.run(iterator_pred) 
                example_counter = 0
                while example_counter != F.shape[0]:
                        p.update(example_counter)
                        pred, x_shape, ids = sess.run([norm_pred,l_shape_op,nxt_F])
                        F[ids,:] = pred
                        example_counter += x_shape[0]           
            return F
        
        

    def fit (self,X,W,Y,labeledIndexes, hook=None, Y_true=None):
        self.Y_true = Y_true
        self.Y_noisy = Y
        Y = self.Y_noisy
        return(self.__NN(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,hook=hook))

    def __init__(self, image_shape = None,OUT_SIZE=10,NUM_EPOCHS=2000,BATCH_SIZE=50,AUGMENT=False,model_choice="simple",
                 LEARNING_RATE=1e-04):
        """ Constructor for NN classifier.
            
        Args:
            
        """
        self.image_shape = image_shape
        self.BATCH_SIZE = BATCH_SIZE
        self.OUT_SIZE = OUT_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.AUGMENT = AUGMENT
        self.model_choice = model_choice
        