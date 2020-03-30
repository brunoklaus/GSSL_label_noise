"""
plot_core.py
====================================
Module related to output and plotting procedures.
"""


import plotly.tools 
import plotly.plotly as py
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyoff
import os
import os.path as path
import igraph as ig
import datetime
import numpy as np
import seaborn as sns
import scipy.sparse
from output.folders import PLOT_FOLDER

#Set a timeout of 1 minute for orca
plotly.io.orca.config.timeout = 60

def authenticate_plotly():
    """ Authenticates plotly for online usage, if possible (Executed when this module is loaded).
    
    The username and api_key are expected to be respectively, the 1st and 2nd lines of the ``auth.txt`` file located
    on the same directory as this module.
    """
    auth_file = path.join(path.dirname(__file__),"auth.txt")
    if path.isfile(auth_file):
        with open(auth_file) as f:
            username = f.readline()
            api_key = f.readline()
            plotly.tools.set_credentials_file(username=username, api_key=api_key)
    else:
        print("Could not find auth.txt on {}".format(auth_file))
authenticate_plotly()       


class vertexplotOpt(object):
    """ A class specifying properties of vertices when plotting graphs with labeled and unlabeled data."""
    
    DEFAULT_CONSTANT_COLOR = (255*np.array([0,0,0])).astype(np.int64)
    DEFAULT_UNLABELED_COLOR = (255*np.array([0,0.5,0])).astype(np.int64)
    

    def __init__(self,Y, mode = "discrete", palette = None, size = 1.5,labeledIndexes = None, change_unlabeled_color=True,
                 UNLABELED_SIZE_MULTIPLIER = 0.2):
        """ Initializes the vertex options.
    
        Args:
            Y (`NDArray[float].shape[N]`) : array from which the color information is extracted.
            mode (str):  either 'discrete','continuous' or 'constant'.Determines the number of colors and the kind of plot to be created.
            palette (str): Optional. A string specifying a color pallete.
            size (Union[float, `NDArray[float].shape[N]`)): Either a float specifying the size of all points, or an float array specifying the individual sizes.
                 specifying the size of each point individually. Default is 1.5   
            labeledIndexes(`NDArray[bool].shape[N]`)  : Optional. Indices to be marked as labeled. Unlabeled points are set to a default unlabeled color,
                and are given a smaller size.
        
        """
        
        self.mode = mode
        self.color_values = np.array(Y)
        
        if np.array(size).shape == ():
            self.size_var = np.repeat(size,Y.shape[0])
        else:
            self.size_var = size
        
        if palette == None:
            if mode == "discrete":
                palette = "bright"
            else:
                palette = "BrBG"
        
        if mode == "discrete":
            self.color_var  = color_scale_discrete(Y, palette)
            self.color_scale = None
            self.group_var = np.array(Y)
            self.names = _vertex_name(Y)
        elif mode == "continuous":
            self.color_var  = np.array(color_scale_continuous(Y, palette))
            self.color_scale = color_scale_continuous(np.linspace(0,1,10), palette)
            self.color_scale = [[y,"rgb"+str(tuple(x[0:3]))] for x,y in zip(self.color_scale,np.linspace(0,1,10))]
            self.names = [str(x) for x in Y]
        elif mode == "constant":
            self.color_var  = np.repeat([vertexplotOpt.DEFAULT_CONSTANT_COLOR],len(Y),axis=0)
            self.color_scale = None
            
        if not labeledIndexes is None:
            #Set color_var of unlabeled indexes to default
            if change_unlabeled_color:
                if mode == "discrete":
                    self.color_var[np.logical_not(labeledIndexes)] = vertexplotOpt.DEFAULT_UNLABELED_COLOR
                else:
                    c = vertexplotOpt.DEFAULT_UNLABELED_COLOR
                    self.color_var[np.logical_not(labeledIndexes),:] = np.array([c[0],c[1],c[2],255.0])
                self.color_values[np.logical_not(labeledIndexes)] = -1
                
                #Set size_var of unlabeled points to half of the original
                self.size_var[np.logical_not(labeledIndexes)] = UNLABELED_SIZE_MULTIPLIER * self.size_var[np.logical_not(labeledIndexes)]
                if mode == "discrete":
                    #Set aside an unique group for unlabeled indexes
                    self.group_var[np.logical_not(labeledIndexes)] = -1
            else:
                #Set size_var of unlabeled points to half of the original
                self.size_var[np.logical_not(labeledIndexes)] = UNLABELED_SIZE_MULTIPLIER * self.size_var[np.logical_not(labeledIndexes)]
                if mode == "discrete":
                    #Set aside an unique group for unlabeled indexes
                    self.group_var[np.logical_not(labeledIndexes)] = -1

        
        
def plotGraph(X,W,labeledIndexes,vertex_opt,plot_filepath = None, online = False,
              interactive=False,title = "", plot_size = [1000,1000], edge_width = 0.5,labeled_only=False):
        """ Plots a GSSL graph.
        
        Creates a plot showing vertices connected by edges from the affinity matrix in 2D/3D.
        A number of different configurations is possible by the use of a vertexplotOpt object.
        
        Args:
            X (`NDArray[float].shape[N,D]`): A 2D or 3D matrix containing the vertex positions. 
            W (`NDArray[float].shape[N,N]`): Optional. The affinity matrix defining the graph.
            vertex_opt (vertexOptObject) : The size/color/group vertex configuration object.
            title (string, default = ``''``) : The title to be printed above the image.
            online (bool, default = ``False``) : whether to create an online plot
            interactive (bool, default = ``False``) : whether to open an interactive plot on the browser
            plot_size (`List[int].shape[2]`, default = ``[1000,1000]``) : size of the canvas for the plotting operation.
            edge_width (float, default = ``0.5``) : thickness of the edges.
        
        Raises:
            ValueError: ``if X.shape[1] not in [2, 3]``
            
        Returns:
            None
        
        """ 
        
        if plot_filepath is None:
            plot_filepath =  path.join(PLOT_FOLDER,str(datetime.datetime.now()) + ".png")
    
        plot_dim = X.shape[1]
        
        if plot_dim < 2 or plot_dim > 3:
            raise ValueError("plot_dim must be either 2 or 3")    
        if (not W is None) and W.shape[0] != X.shape[0]:
            raise ValueError("Shapes of W, X do not match")
                
        if plot_dim > X.shape[1]:
            #Add missing dimensions
            temp = np.zeros((X.shape[0],plot_dim))
            temp[0:X.shape[0],0:X.shape[1]] = X
            X = temp
        
        if not os.path.exists(os.path.dirname(plot_filepath)): 
            os.makedirs(os.path.dirname(plot_filepath))
        
        
        
     
        def axis(dim_num):
            M = np.max(X[:,dim_num])
            m = np.min(X[:,dim_num])
            x = (M-m)/2
            M = M + (M-x)*0.2
            m = m + (m-x)*0.2
            
             
            axis=dict(
                      showline=False,
                      zeroline=False,
                      showgrid=False,
                      showticklabels=False,
                      visible=True,
                      title='',
                      range = [m,M]
                      )
            return(axis)
        
        scene = {}
        scene["xaxis"] = dict(axis(0))
        scene["yaxis"] = dict(axis(1))
        if plot_dim == 3:
            scene["zaxis"] = dict(axis(2))
         
        
        layout = go.Layout(
                 title=title,
                 font=dict(family='Courier New, bold', size=30, color='black'),
                 width=plot_size[0],
                 height=plot_size[1],
                 legend=dict(x=0,
                    y=1,
                    traceorder='normal',
                    font=dict(
                        family='sans-serif',
                        size=30,
                        color='#000'
                    ),
                    bgcolor='#E2E2E2',
                    bordercolor='#FFFFFF',
                    borderwidth=2),
                     showlegend=True,
                     xaxis=scene["xaxis"],
                     yaxis=scene["yaxis"],
                             
                         
                     margin=dict(
                        t=100
                    ),
                    hovermode='closest',
                    annotations=[
                           dict(
                           showarrow=False,
                            text="</a>",
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=0.1,
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(
                            size=14
                            )
                            )
                ],    )     
        
        if "zaxis" in scene.keys():
            layout.update(go.Layout(zaxis=scene["zaxis"]))
        
        #Create Traces
        data = []
        
        if not W is None:
            trace_edge = _traceEdges(X=X, W=W,plot_dim=plot_dim, edge_width=edge_width)
            data = trace_edge
            
        trace_vertex = _traceVertex(X=X,labeledIndexes=labeledIndexes, plot_dim=plot_dim, v_opt = vertex_opt)
        data += trace_vertex
        
        #Create figure
        fig=go.Figure(data=data, layout=layout)
        
        
        print("Plotting graph..." + title)
        if online:
            try:
                py.iplot(fig)
            except plotly.exceptions.PlotlyRequestError:
                print("Warning: Could not plot online")
                
        if interactive:
            pyoff.offline.plot(fig)
        pio.write_image(fig,plot_filepath)
        print("Done!")  


def _traceVertex(X,labeledIndexes,plot_dim, v_opt):
        if X.shape[1] != plot_dim:
            raise ValueError("data does not have dimension equal to the one specified by the plot")
        
        X = np.array(X)
        trace = []
        
        def flatten(X):
            return np.reshape(X,(-1))
        
        ''' Creates the style attribute for discrete plot'''
        def getStyle(values,color_var):
            styles = {}
            for i, x in enumerate(values):
                styles[x] = color_var[i]
            return [dict(target = x, value = dict(marker = dict(color = y))) \
                    for x,y in styles.items()]
            
        opacity = 0.5*np.ones(X.shape[0])
        opacity[np.logical_not(labeledIndexes) ] = 1.0
        if v_opt.mode == "discrete":
            #One plot for each group
            levels = np.unique(v_opt.group_var)
            call_dicts = []
            for l in levels:
                l_ids = np.where(v_opt.group_var == l)
                l_dict =\
                    dict(
                       x=flatten(X[l_ids,0]),
                       y=flatten(X[l_ids,1]),
                       z = None if plot_dim == 2 else flatten(X[l_ids,2]),
                       name=_vertex_name([v_opt.color_values[l_ids][0]])[0],
                       text = _vertex_name(v_opt.color_values[l_ids]),
                       mode='markers',
                       marker=dict(symbol="circle" if l == -1 else "diamond",
                                   opacity = opacity,
                                     size=v_opt.size_var[l_ids],
                                     color=v_opt.color_var[l_ids],
                                     line=dict(color='rgb(50,50,50)', width= 0.1 if l == -1 else 1.0),

                                     ),
                       hoverinfo='text')
                call_dicts += [l_dict]
            
            
        else:
            call_dicts = [\
                dict(  x=X[:,0],
                       y=X[:,1],
                       z=None if plot_dim == 2 else X[:,2],
                       mode='markers',
                       showlegend = True,
                       marker=dict(symbol='circle',
                                     size=v_opt.size_var,
                                     color=v_opt.color_var,
                                     line=dict(color='rgb(50,50,50)', width=0.1),
                                     ),
                       hoverinfo='text')]
        
            
        for call_dict in call_dicts:   
             
            if plot_dim == 2:
                call_dict.pop("z")
                call_dict["marker"]["color"] = ["rgb"+str(tuple(x)) for x in call_dict["marker"]["color"]]
                #print(call_dict["marker"]["color"])
                call_dict["text"] = [str(x) for x in v_opt.color_values]
                call_dict["type"] = "scattergl"    
            else:
                call_dict["text"] = v_opt.color_values
                call_dict["type"] = "scatter3d"
                
                
            
            if v_opt.mode == "continuous":
                #Add color scale
                call_dict["marker"]["colorscale"] = v_opt.color_scale 
                call_dict["marker"]["cmax"] = np.max(v_opt.color_values)
                call_dict["marker"]["cmin"] = np.min(v_opt.color_values)
                call_dict["marker"]["colorbar"] = dict(title="scale")
                
                

            
        return(call_dicts)
    
def _traceEdges(X,W,plot_dim,edge_width):
        xe=[]
        ye=[]
        ze=[]
        ce = []
        trace=[]
        
        def flatten(x):
            return(np.reshape(x,(-1)))
        if scipy.sparse.issparse(W):
            W = 0.5*(W + W.T)
            W = W.tocoo() 
            
            for i,j,w in zip(W.row,W.col,W.data):
                if i > j: continue 
                 
            
                xe.append([X[i,0],X[j,0],None])# x-coordinates of edge ends
                ye.append([X[i,1],X[j,1],None])# y-coordinates of edge ends
                if plot_dim > 2:
                    ze.append([X[i,2],X[j,2],None])# z-coordinates of edge ends
                ce.append(w)
        else:      
            for i in np.arange(X.shape[0]):
    
                for j in np.arange(X.shape[0]):
                    temp = W[i,j] + W[j,i]
                    if temp == 0: continue 
                     
                
                    xe.append([X[i,0],X[j,0],None])# x-coordinates of edge ends
                    ye.append([X[i,1],X[j,1],None])# y-coordinates of edge ends
                    if plot_dim > 2:
                        ze.append([X[i,2],X[j,2],None])# z-coordinates of edge ends
                    ce.append(0.5*temp)
                
        
        
        xe = np.array(xe)
        ye = np.array(ye)
        ze = np.array(ze)
        ce = np.array(ce)
        ids = np.argsort(ce)
        if np.max(ce) == np.min(ce):
            ce = np.linspace(0.5,0.5,ce.shape[0])
        else:
            ce = np.linspace(0,1,ce.shape[0])
        xe = xe[ids]
        ye = ye[ids]
        if plot_dim > 2:
            ze = ze[ids]
        
       
        
        splt = [list(x) for x in np.array_split(np.arange(len(ce)),10)]
        


        max_brightness = 210
        for x in splt:
            
            col = max_brightness - int(max_brightness*ce[x][0])
            col = 'rgb' + str( (col,col,col) ) if plot_dim == 2 else (col,col,col)
            new_edge = dict(x=flatten(xe[x]),
                       y=flatten(ye[x]),
                       type="scattergl",
                       mode='lines',
                       showlegend=False,
                       line=dict(color = col,
                                  width=edge_width),
                       hoverinfo='none'
                       )
            if plot_dim == 3:
                new_edge["z"] = flatten(ze[x])
                new_edge["type"] = "scatter3d"
            trace.append(new_edge)
            
        return(trace)
        
def _vertex_name(Y):
    return(["Class #" + str(x) if x != -1 else "unlabeled" for x in Y])
           
def color_scale_discrete(Y,palette="bright"):
    """ Gets the color values for a discrete palette. """
    
    if Y.shape[0] == -1:
        raise ""
    Y = Y - np.min(Y) 
    pal = sns.color_palette(palette,np.max(Y)+1)
    res = 255*np.array(list(map(lambda k: (pal[int(k)]),Y)))
    return(res)

def color_scale_continuous(Y,palette="coolwarm",num_palette=70):    
    """ Gets the color values for a continuous palette. """
    Y = (Y - np.min(Y))
    Y = Y / np.max(Y)
    Y[np.isnan(Y)] = 0.5
    
    pal = ig.AdvancedGradientPalette(sns.color_palette(palette,7),n=num_palette+1)
    res = 255*np.array(list(map(lambda k: (pal.get(int(num_palette*k))),Y)))
    res = res.astype(np.int64)
    
    return res

    



if __name__ == "__main__":
    print(PLOT_FOLDER)

    
