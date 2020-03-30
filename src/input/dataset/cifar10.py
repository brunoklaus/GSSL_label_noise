'''
Created on 13 de set de 2019

@author: klaus
'''
import os
import pickle
INPUT_FOLDER = os.path.dirname(os.path.realpath(__file__))
CIFAR10_PATH = os.path.join(INPUT_FOLDER,"cifar10_data")
import numpy as np
def load_cifar10_batch(batch_id):
    with open(CIFAR10_PATH + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def load_cifar10_test():
    
    # load the test dataset
    with open(CIFAR10_PATH+ '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']
    return test_features, test_labels
def load_cifar10_labelnames():
    # load the test dataset
    with open(CIFAR10_PATH+ '/batches.meta', mode='rb') as file:
        dct = pickle.load(file, encoding='latin1')
    return dct['label_names']



def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = np.linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return whiten

def get_cifar10(flattened=False):
    if flattened:
        X = np.zeros((60000,32*32*3))
    else:
        X = np.zeros((60000,32,32,3))
    Y = np.zeros((60000,))
    i = 0
    
    for b_id in range(1,7):
        if b_id == 6:
            features, labels = load_cifar10_test()
            print(features)
        else:
            features, labels = load_cifar10_batch(b_id)
        if flattened:
            features = np.reshape(features,(features.shape[0],-1,))
        
        
        X[i:(i+features.shape[0]),:] = features
        Y[i:(i+features.shape[0])] = labels
        i += features.shape[0]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = ZCA(X)
    Y = Y  -  np.min(Y)
    
    return {"X":X,"Y": Y}







if __name__ == "__main__":
    get_cifar10()


