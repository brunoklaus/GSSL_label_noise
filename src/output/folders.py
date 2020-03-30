'''
Created on 25 de mar de 2019

@author: klaus
'''
import os.path as path

TOP_DIR_NAME = "src"

def get_top_dir():
    i = 0
    p = __file__
    while not p.endswith(TOP_DIR_NAME):
        p = path.dirname(p)
        i += 1
        if len(p) < len(TOP_DIR_NAME) or i > 100:
            raise FileNotFoundError("Could not go up till a directory named {} was found".format(TOP_DIR_NAME))
    return(path.dirname(p))    
        
TOP_DIR = get_top_dir()
#: Path to the folder destined for plot output.
PLOT_FOLDER = path.join(TOP_DIR,"results","python_plotly")
#: Path to the general results folder.
RESULTS_FOLDER = path.join(TOP_DIR,"results")
#: Path to the general results folder.
CSV_FOLDER = path.join(RESULTS_FOLDER,"csvs")



if __name__ == '__main__':
    pass