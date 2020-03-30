'''
Created on 2 de abr de 2019

@author: klaus
'''
import output.plots as plots
import os.path as path
import os
from inspect import signature, Parameter
from functools import partial
import numpy as np
import shutil
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import experiment.hooks.hook_skeleton as hk
import  scipy.sparse
class plotHook(hk.GSSLHook):
    """ Hook that plots labels. Uses a callback to an `Experiment` object to get X and W, if available. """
    
    def plot(self, **kwargs):
        xp = self.experiment
        
    
        #Add W if we have it
        W = None
        if "W" in kwargs.keys():
            W = kwargs["W"]
        elif hasattr(xp,"W"):
            W = xp.W
            
            
        if  self.only_labeled and not "labeledIndexes" in kwargs.keys():
            raise ValueError("It is required to present 'labeledIndexes' as an argument")


        if not self.force_Y_callback is None:
            kwargs["Y"] = getattr(xp,self.force_Y_callback)
            
        
        if not self.force_lb_callback is None:
            kwargs["labeledIndexes"] = getattr(xp,self.force_lb_callback)

        if not "Y" in kwargs.keys():
            raise ValueError("It is required to present 'Y' as an argument")


        if not W is None:
            W = scipy.sparse.csr_matrix(W)

        
        if not self.only_labeled:
            f = partial(lambda X,Y,labeledIndexes: plots.plot_all_indexes(X=self.X_transf, Y=Y,labeledIndexes=labeledIndexes,mode=self.plot_mode,palette=self.palette,
                                                            W=W,title=self.title,plot_filepath= self.filename_path))
        else:
            f = partial(lambda X,Y,labeledIndexes: plots.plot_labeled_indexes(X=self.X_transf, Y=Y,mode=self.plot_mode,palette=self.palette,
                                                           labeledIndexes=labeledIndexes,W=W,title=self.title,plot_filepath= self.filename_path))
        
        if self.X_transf is None:
            if "X"in kwargs.keys():
                X =  kwargs["X"]
            else:
                X = getattr(xp,"X")
            
            if X.shape[1] > 3 or X.shape[1] < 2:
                from sklearn.manifold import LocallyLinearEmbedding, TSNE
                self.X_transf = LocallyLinearEmbedding(n_neighbors=30,n_components=2,random_state=1308,method="modified").fit_transform(X)
                #self.X_transf = TSNE(random_state=19308).fit_transform(X)
                
            else:
                self.X_transf = X
        kwargs = hk._remove_excess_vars(f, kwargs)
        

                
        kwargs = hk._add_remaining_vars(f, kwargs, xp)
        f(**kwargs)
        
    def _begin(self, **kwargs):
        if self.when == "begin":
            self.plot(**kwargs)

    def _step(self,step, **kwargs):
        pass
    def _end(self, **kwargs):
        if self.when == "end":
            self.plot(**kwargs)
            
    def __init__(self, filename_path,title,experiment,when="begin",only_labeled=True,force_Y_callback = None,
                 force_lb_callback = None,plot_mode="discrete",palette=None):
        """ Constructor for the plotHook.
        
        Args:
            filename_path (str): Path to the output file (including filename and extension).
            title (str): The title to be displayed on the plot.
            experiment (:class:`experiment.experiments.Experiment`): The Experiment object, used for callbacks.
            when (str) : Either 'begin' or 'end'. If it is 'begin', plotting occurs when '_begin' method is called. Otherwise, it
                occurs when '_end' is called. Default is ``'begin'``.
            only_labeled (bool): If ``True``, future calls must have 'labeledIndexes' kwarg, and data marked as unlabeled will
                have a different color. Default is ``True``.
            force_Y_callback (str): Optional. Should be an attribute from :class:`experiment.experiments.Experiment` that
                will override the Y variable (which determines the color of the plot).
            plot_mode (str): Either ``'discrete'``` or ``'continuous'``. The plot type. Default is ``'discrete'``.
            palette (str): Optional. Overrides the palette setting for the plot.
        """
            
        self.filename_path = filename_path
        self.title = title
        self.experiment = experiment
        self.when = when
        self.only_labeled = only_labeled
        self.force_Y_callback = force_Y_callback
        self.force_lb_callback = force_lb_callback
        self.plot_mode = plot_mode
        self.palette = palette
        self.X_transf = None


class plotIterHook(plotHook):
    """ Hook that plots the labels iteratively. Uses a callback to an `Experiment` object to get X and W, if available. """
    def _begin(self, **kwargs):
        self.steps_taken = 0
        Y = kwargs["Y"]
        self.str_len  = int(np.ceil(np.log10(Y.shape[0]+1)))
        

    def _step(self,step, **kwargs):
        self.steps_taken = self.steps_taken + 1
        if "plot_id" in kwargs:
            if kwargs["plot_id"] != self.plot_id:
                return
        if step % self.step_size == 0:
            step = str(step)
            step = "0"* int(self.str_len - len(step)) + step 
            self.filename_path = path.join(self.filename_dir,self.temp_subfolder_name,'{}.png'.format(step))
            self.plot(**kwargs)
            
    def _end(self, **kwargs):
        self.createVideo()
        self.rmFolders()

        
    def createVideo(self):
        if not self.create_video:
            return
        
        if self.steps_taken==0:
            return
        print("Creating video...")
        video_command = "ffmpeg -r {} -y  -pattern_type glob -i '{}' -c:v libx264 -vf fps=25 -pix_fmt yuv420p '{}'".format(\
            self.steps_taken/(15.0*5.0),
            os.path.join(self.filename_dir,self.temp_subfolder_name,"*.png".format(self.str_len)),
            os.path.join(self.filename_dir,self.video_path)
            )
        print(video_command)
        os.system(video_command)
        print("Created video...")
        
    def rmFolders(self):
        if self.keep_images:
            return
        shutil.rmtree(os.path.join(self.filename_dir,self.temp_subfolder_name))
        print("Deleted images....")
        
    def __init__(self, video_path,title,experiment,only_labeled=True,step_size=5,force_Y_callback = None,
                 force_lb_callback=None,temp_subfolder_name="iter",
                 plot_mode="discrete",palette=None,keep_images=False, create_video=True,slowdown_factor=1.0,plot_id=0):
        """ Constructor for the plotHook.
        
        Args:
            video_path (str): Path to the output video (including filename and '.mp4' extension).
            title (str): The title to be displayed on the plot.
            experiment (:class:`experiment.experiments.Experiment`): The Experiment object, used for callbacks.
            only_labeled (bool): If ``True``, future calls must have 'labeledIndexes' kwarg, and data marked as unlabeled will
                have a different color. Default is ``True``.
            step_size (int): Determines after how many steps should a plotting operation occur.
            force_Y_callback (str): Optional. Should be an attribute from :class:`experiment.experiments.Experiment` that
                will override the Y variable (which determines the color of the plot).
            temp_subfolder_name (str): Determines the name of the temporary subfolder to store the plot slideshow images.
                Default is ``'iter'``.
            plot_mode (str): Either ``'discrete'``` or ``'continuous'``. The plot type. Default is ``'discrete'``.
            palette (str): Optional. Overrides the palette setting for the plot.
        """
        self.video_path = video_path
        self.filename_dir = path.dirname(self.video_path)
        self.video_name = self.video_path[len(self.filename_dir):]
        
        
        temp_subfolder_name = temp_subfolder_name + "_" + str(plot_id)
        self.temp_subfolder_name = temp_subfolder_name
        
        #Create directory for video
        if not path.isdir(self.filename_dir):
            os.makedirs(self.filename_dir)

        #Create directory for temp folder
        if not path.isdir(path.join(self.filename_dir,temp_subfolder_name)):
            os.makedirs(path.join(self.filename_dir,temp_subfolder_name))
        
        self.plot_id = plot_id
        self.title = title
        self.experiment = experiment
        self.only_labeled = only_labeled
        self.step_size = step_size
        self.force_Y_callback = force_Y_callback
        self.plot_mode = plot_mode
        self.palette = palette
        self.force_lb_callback = force_lb_callback
        
        self.keep_images = keep_images
        self.create_video = create_video
        self.slowdown_factor = slowdown_factor
        
        self.X_transf = None


class plotIterGTAMHook(plotIterHook):
    """ Hook that is specific to the GTAM algorithm. There are 3 modes:
        1. ``'Y'``: Plots the updated initial belief matrix for each iteration.
        2. ``'Q'``: Plots the argmin of each row of the gradient. The lower the values, the more the cost function is decreased 
           with the labeling of that instance.
        3. ``'F'``: Plots the updated classification  for each iteration.
    
     """
    def _step(self,step, **kwargs):
        if self.mode == "Y":
            pass
        elif self.mode == "F":
            kwargs["Y"] = kwargs["P"] @ kwargs["Z"]
        else:
            self.only_labeled = True
            Q = np.reshape(np.min(kwargs["Q"],axis=1),(-1))
            Q[kwargs["labeledIndexes"]] = 0        
            
            kwargs["labeledIndexes"] = np.logical_not(kwargs["labeledIndexes"]) 
            if any(kwargs["labeledIndexes"]):                
                Q[kwargs["labeledIndexes"]] = np.reshape(MinMaxScaler(feature_range=(-1, 1)).fit_transform(
                    StandardScaler().fit_transform(np.reshape( Q[kwargs["labeledIndexes"]],(-1, 1)))),(-1))
            kwargs["Y"] = Q
        assert kwargs["Y"].shape[0] > 1
        super()._step(step,**kwargs)
        
    
    def __init__(self, video_path,title,experiment,step_size=10,mode="Y",palette=None,keep_images=False, create_video=True):
        """ Constructor for the plotIterGTAMHook.
        
        Args:
            video_path (str): Path to the output video (including filename and '.mp4' extension).
            title (str): The title to be displayed on the plot.
            experiment (:class:`experiment.experiments.Experiment`): The Experiment object, used for callbacks.
            step_size (int): Determines after how many steps should a plotting operation occur.
            mode (str) : The mode of this plotIterGTAMHook. Must be in ``['Y','Q','F']``. Default is `Y`.
            palette (str): Optional. Overrides the palette setting for the plot.
            keep_images (bool): Whether to keep the images. Default is ``False``.
            create_video (bool): Whether to create a video showcasing a slideshow of the images. Default is ``True``.
        """
        if mode == "Y":
            super().__init__(video_path,title,experiment,only_labeled=True,plot_mode="discrete",palette=palette,\
                                              step_size=step_size,temp_subfolder_name="iter_Y",
                                              keep_images=keep_images,create_video=create_video)
        elif mode == "F":
            super().__init__(video_path,title,experiment,only_labeled=False,plot_mode="discrete",palette=palette,\
                                              step_size=step_size,temp_subfolder_name="iter_F",
                                              keep_images=keep_images,create_video=create_video)
        elif mode == "Q":    
            super().__init__(video_path,title,experiment,only_labeled=False,plot_mode="continuous",palette=palette,\
                          step_size=step_size,temp_subfolder_name="iter_Q",
                                              keep_images=keep_images,create_video=create_video)
        else:
            raise ValueError("Invalid mode. May be one of {'Y','F','Q'}")
        self.mode = mode
    