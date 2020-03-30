import numpy as np
import pandas as pd
import sklearn.model_selection
import os
import gssl.graph.gssl_utils as gutils

TOY_DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),"toy_data")
print("Toy dataset path:{}".format(TOY_DATASET_PATH))

def getDataframe(ds_name):
    path_X = TOY_DATASET_PATH + "/" + ds_name + "_X.csv"
    path_Y = TOY_DATASET_PATH + "/" + ds_name + "_Y.csv"
    
    X = pd.read_csv(path_X,sep=",",index_col=0,header=0)
    Y = pd.read_csv(path_Y,sep=",",index_col=0,header=0)
    return {"X":X.values,"Y":np.reshape(Y.values,(-1)) - 1}

def getTFDataset(ds_name, num_labels,label_seed=1):
    
    import tensorflow as tf
    df = getDataframe(ds_name)
    perm = np.random.RandomState(seed=label_seed).permutation(np.arange(df["X"].shape[0]))
    is_labeled = gutils.split_indices(df["Y"], num_labels/df["X"].shape[0])
    is_unlabeled = np.logical_not(is_labeled)
    df_x_l = df["X"][is_labeled,:]
    df_y_l = df["Y"][is_labeled]
    df_id_l = np.reshape(np.where(is_labeled),[-1,1])
    df_x_ul = df["X"][is_unlabeled,:]
    df_y_ul = df["Y"][is_unlabeled]
    df_id_ul = np.reshape(np.where(is_unlabeled),[-1,1])
    
    
    
    
    with tf.Session() as sess:
        init_confidence_values = tf.one_hot(tf.cast(df["Y"],dtype=tf.int32),
                                       1 + np.max(df["Y"]))
        init_confidence_values = init_confidence_values.eval()
        init_confidence_values[perm[num_labels:],:] *= 0
    
        df_y_l = tf.one_hot(tf.cast(df_y_l,dtype=tf.int32),
                                       1+np.max(df["Y"])).eval()
        df_y_ul = tf.one_hot(tf.cast(df_y_ul,dtype=tf.int32),
                                       1+np.max(df["Y"])).eval()
        df_y = tf.one_hot(tf.cast(df["Y"],dtype=tf.int32),
                                       1+np.max(df["Y"])).eval()
    
    labeled_ds = tf.data.Dataset.from_tensor_slices({
                    "ID":df_id_l,
                    "X":df_x_l,
                    "Y":df_y_l})
    unlabeled_ds = tf.data.Dataset.from_tensor_slices({
                    "ID": df_id_ul,
                    "X":df_x_ul,
                    "Y":df_y_ul})
    all_ds = tf.data.Dataset.from_tensor_slices({
                    "ID":np.arange(df["X"].shape[0]),
                    "X":df["X"],
                    "Y":df_y})
    
    return{"labeled":labeled_ds,"unlabeled":unlabeled_ds,"df_x":df["X"],"df_y":df_y,
           "df_x_ul":df_x_ul,"df_x_l":df_x_l,"df_y_l": df_y_l,"INIT":init_confidence_values}

    


