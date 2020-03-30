'''
Created on 22 de Nov de 2019

@author: klaus
'''
import os
import pandas as pd
import pickle
INPUT_FOLDER = os.path.dirname(os.path.realpath(__file__))
MNIST_PATH = os.path.join(INPUT_FOLDER,"mnist_data")
import numpy as np

def get_mnist():
    X = pd.concat([pd.DataFrame.from_csv(os.path.join(MNIST_PATH,'mnist_train.csv'),header=None,index_col=-1),
                   pd.DataFrame.from_csv(os.path.join(MNIST_PATH,'mnist_test.csv'),header=None,index_col=-1)],axis=0).values
    Y = np.reshape(X[:,0],(-1,))
    X = X[:,1:]
                                         
                              
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    
    return {"X":X,"Y": Y}







if __name__ == "__main__":
    get_mnist()


