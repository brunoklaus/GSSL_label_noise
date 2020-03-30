'''
Created on 28 de mar de 2019

@author: klaus
'''
import output.plot_core as plt
import gssl.graph.gssl_utils as gutils

def plot_labeled_indexes(X,Y,labeledIndexes,W=None,title="labeled indexes",plot_filepath = None,
                            mode="discrete",palette=None):  
    print(Y.shape)
    print(X.shape)
    assert Y.shape[0] > 1

    Y = gutils.get_pred(Y) if mode == "discrete" else Y
    assert Y.shape[0] > 1

    #Plot 1: labeled indexes
    vertex_opt = plt.vertexplotOpt(Y=Y,mode=mode,size=14,labeledIndexes=labeledIndexes,
                                   palette=palette,UNLABELED_SIZE_MULTIPLIER=0.2)
    plt.plotGraph(X,W=W,labeledIndexes=labeledIndexes, vertex_opt= vertex_opt,edge_width=0.75,\
                  interactive = False,title=title,plot_filepath=plot_filepath,labeled_only=True)
             
            
    
def plot_all_indexes(X,Y,labeledIndexes,W=None,plot_filepath = None,title="True classes",mode="discrete",palette=None):
    assert Y.shape[0] > 1

    #Plot 2: True classif
    Y = gutils.get_pred(Y) if mode == "discrete" else Y 
    assert Y.shape[0] > 1

    vertex_opt = plt.vertexplotOpt(Y=Y,mode=mode,size=14,
                                   labeledIndexes=labeledIndexes,palette=palette,change_unlabeled_color=False,
                                   UNLABELED_SIZE_MULTIPLIER=0.35)
    plt.plotGraph(X,W=W, labeledIndexes=labeledIndexes, vertex_opt= vertex_opt,edge_width=0.75,\
                  interactive = False,title=title, plot_filepath=plot_filepath,labeled_only=False)


    


if __name__ == '__main__':
    pass