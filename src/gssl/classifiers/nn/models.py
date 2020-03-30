'''
Created on 4 de out de 2019

@author: klaus
'''

import numpy as np
import gssl.graph.gssl_utils as gutils
import scipy.linalg as sp
import tensorflow as tf
import tensorflow.keras as keras
import scipy.sparse
from settings import p_bar
import tensorflow.keras as K

layers = tf.keras.layers
regularizers = tf.keras.regularizers

def linear(input_shape, output_shape):
       
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape,name="descriptor"))
        model.add(layers.Dense(20,kernel_regularizer=keras.regularizers.l2(0.0)))
        model.add(layers.Activation('tanh'))
        model.add(layers.Dense(30,kernel_regularizer=keras.regularizers.l2(0.0)))
        model.add(layers.Activation('tanh'))

        
        model.add(layers.Dense(output_shape))
        print(model.summary())
        model = keras.models.Model(model.input,[model.output,model.input])
        return model

def simple(input_shape, output_shape):
    model = keras.Sequential()
    model.add(layers.Dense(10,input_shape=input_shape))
    model.add(layers.ReLU())
    model.add(layers.Dense(output_shape))   
    
    return model


def conv_large(input_shape, output_shape):
    #weight_decay = 1e-02
    model = keras.Sequential()
    
    model.add(layers.Conv2D(128, (3,3), padding='same',\
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    
    i = 0
    for f_out, max_p, ksize,pad in zip([128,128,256,256,256,512,256,128],
                                   [False,True,False,False,True,False,False,False],
                                   [3,3,3,3,3,3,1,1],
                                   ['same','same','same','same','same','valid','same','same']):
        
    
        model.add(layers.Conv2D(f_out, (ksize,ksize), padding=pad))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.1))
        
        if max_p:
            model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Dropout(0.5))
        i += 1
    
    model.add(layers.GlobalAveragePooling2D(name="global_pool"))
    

    model.add(layers.Lambda(lambda t: K.backend.l2_normalize(t,axis=1),name="global_pool_unit"))

        
    model.add(layers.Dense(output_shape))
    
   
    descr = model.get_layer("global_pool_unit").output 
    #descr = model.input
    model = keras.models.Model(model.input,[model.output] \
                               + [model.get_layer("leaky_re_lu_{}".format(i)).output for i in range(1,8)]\
                               + [descr])
    print(model.summary())
    return model



def conv_small(input_shape, output_shape):
    #weight_decay = 1e-02
    model = keras.Sequential()
    
    model.add(layers.Conv2D(96, (3,3), padding='same',\
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    
    i = 0
    for f_out, max_p, ksize,pad in zip([96,96,192,192,192,192,192,192],
                                   [False,True,False,False,True,False,False,False],
                                   [3,3,3,3,3,3,1,1],
                                   ['same','same','same','same','same','valid','same','same']):
        
    
        model.add(layers.Conv2D(f_out, (ksize,ksize), padding=pad))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.1))
        
        if max_p:
            model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Dropout(0.5))
        i += 1
    
    model.add(layers.GlobalAveragePooling2D(name="global_pool"))
    

    model.add(layers.Lambda(lambda t: K.backend.l2_normalize(t,axis=1),name="global_pool_unit"))

        
    model.add(layers.Dense(output_shape))
    
    descr = model.get_layer("global_pool_unit").output 
    #descr = model.input
    model = keras.models.Model(model.input,[model.output] \
                               + [model.get_layer("leaky_re_lu_{}".format(i)).output for i in range(1,8)]\
                               + [descr])
    print(model.summary())

    return model