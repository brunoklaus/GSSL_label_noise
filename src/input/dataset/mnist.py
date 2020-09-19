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
    X = pd.concat([pd.read_csv(os.path.join(MNIST_PATH,'mnist_train.csv'),header=None),
                   pd.read_csv(os.path.join(MNIST_PATH,'mnist_test.csv'),header=None)],axis=0).values
    Y = np.reshape(X[:,0],(-1,))
    X = X[:,1:]
                                         
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    """
    for x in [8940, 63344, 48865]:
        plt.imshow(np.reshape(X[x,:],(28,28)),cmap='gray')
        plt.show()
    raise ""        
    """
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    
    return {"X":X,"Y": Y}







if __name__ == "__main__":
    get_mnist()


