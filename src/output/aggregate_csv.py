"""
aggregate_csv.py
====================================
Module encapsulating the functionality of merging runs that differ only in id.
"""
import pandas as pd
import os
import sys
import numpy as np
from experiment.prefixes import OUTPUT_PREFIX
from matplotlib.sphinxext.plot_directive import out_of_date



def calculate_statistics(df):
    """ Obtains a dataframe which 'merges' runs that share the same configuration but different ``'id'``.
        For each output variable (as in, one that uses the prefix :const:`experiment.prefixes.OUTPUT_PREFIX`), a few summary statistics are calculated:
         
         * A string with the comma-separated values.
         * The mean of the attribute.
         * The standard deviation of the attribute.
         * The median of the attribute.
         * The minimum value of the attribute.
         * The maximum value of the attribute.
         
    Args:
        df (pd.Dataframe): The original dataframe
    Returns:
        pd.Dataframe: A dataframe with statistics about runs that share same configuration.
    
    """
    
    '''
        out_dict (dict) : A dictionary with k,v pairs:
            
            k (str): a string uniquely identifying the runs that differ only in id
            v (dict): A dictionary with k1, v1 pairs:
            
                k1 (str): some output attribute
                v1: The value of the output attribute 
        
    '''
    out_dict = {}
    freq_dict = {}
    index_dict = {}
    
    rel_cols = [x  for x in df.columns if not (x.startswith(OUTPUT_PREFIX) or x == "id" or x == "index") ]
    out_cols = [x  for x in df.columns if x.startswith(OUTPUT_PREFIX) ]
    
    
    print("rows:{}".format(df.shape[0]))
    
    for i in range(df.shape[0]):
        print(i)
        k = str(df.loc[df.index[i],rel_cols].values)

        #Initialize with empty dicts
        if not k in out_dict.keys():
            out_dict[k] = {}
            freq_dict[k] = 0
            
        
        if index_dict.get(k,None) is None:
            index_dict[k] = i
        
        freq_dict[k] += 1
        for k1 in out_cols:            
            v1 = df.loc[df.index[i],k1]
            L = (out_dict[k]).get(k1,[])    
            L.append(v1)
            (out_dict[k])[k1] = L
    

    agg_cols =  [[x + "_mean", x + "_sd", x + "_values",\
                  x + "_min", x + "_max", x + "_median"] for x in out_cols]
    agg_cols = [item for sublist in  agg_cols for item in sublist]
    agg_cols += ["out_num_experiments"] + rel_cols
    
    key_list = list(index_dict.keys())
    
    new_df = pd.DataFrame(index=range(len(key_list)),columns=agg_cols)


    
    print("Num keys:{}".format(len(key_list)))
    for i in range(len(key_list)):
        print(i)
        k = key_list[i]
        new_df.loc[df.index[i],"out_num_experiments"] = freq_dict[k]   
       
        new_df.loc[df.index[i],rel_cols] = df.loc[df.index[index_dict[k]],rel_cols]
        for k1 in out_cols:
            vals = out_dict[k][k1]
            new_df[k1+ "_mean"].iloc[i] = np.mean(vals)
            new_df[k1+ "_sd"].iloc[i] = np.std(vals)
            new_df[k1 + "_values"].iloc[i] = ','.join([str(x) for x in vals])
            new_df[k1 + "_min"].iloc[i] = min(vals)
            new_df[k1 + "_max"].iloc[i] = max(vals)
            new_df[k1 + "_median"].iloc[i] = np.median(vals)
        
        
    new_df = new_df.loc[:,[x not in ['acc','id','index'] for x in new_df.columns]]
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    new_df = new_df.sort_values(by = list(new_df.columns))
    return(new_df)
    
def aggregate_csv(files_to_join,output_path):
    """ Joins multiple CSVs and produces the aggregate dataframe according to :meth:`calculate_statistics`. 
    It is assumed that each csv corresponds to a set of runs. Each CSV must have the same column structure.
    
    Args:
        files_to_join (List[str]) : The list of files to join.
        output_path (str) : path to the produced aggregate dataframe, including filename and '.csv' extension.
        
    Returns:
        None
    """


    joined_df = None

    for i in range(len(files_to_join)):
        some_f = files_to_join[i]
        if not os.path.isfile(some_f):
            raise FileNotFoundError("Did not find " + str(some_f))

        new_df = pd.read_csv(some_f,delimiter=",",header=0)
            

        if i == 0:
            joined_df = new_df
        else:
            joined_df = joined_df.append(new_df)
    joined_df = joined_df.reset_index()
    joined_df = joined_df.loc[:,[x not in ['index','Unnamed: 0'] for x in joined_df.columns]]
    
    
    summarized_df = calculate_statistics(joined_df)
    print(summarized_df.iloc[0:10,])
    print(summarized_df.shape)
    
    summarized_df.to_csv(output_path)  
    
if __name__ == "__main__":
    aggregate_csv(["/home/klaus/eclipse-workspace/NoisyGSSL/results/csvs/Nov28/29Nov_filter_LGCLVOvsBaseline_chap.csv"],
                  "/home/klaus/eclipse-workspace/NoisyGSSL/results/csvs/Nov28/29Nov_filter_LGCLVOvsBaseline_chap_joined_v2.csv")
