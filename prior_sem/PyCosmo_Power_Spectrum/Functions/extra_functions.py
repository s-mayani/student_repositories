import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def deltanorm(a,b):
    
    """
    This function returns the absolute relative difference between two given quantities.
    :param a,b: input observables;
    :return rd: relative difference (modulo)
    """
    
    return np.abs( (a-b)/a )


def setup_matplotlib():
    
    import warnings
    warnings.filterwarnings("ignore")
    
    matplotlib.rcParams['lines.linewidth'] = 2.0
    matplotlib.rcParams['lines.linestyle'] = '-'  
    matplotlib.rcParams['lines.color'] = 'black'

    #fonts & text
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.weight'] = 'normal'
    matplotlib.rcParams['font.size'] = 24.0
    matplotlib.rcParams['text.color'] = 'black'
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True

    #axes & ticks
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    matplotlib.rcParams['axes.linewidth'] = 1.0  
    matplotlib.rcParams['axes.grid'] = False
    matplotlib.rcParams['axes.titlesize'] = 'x-large'
    matplotlib.rcParams['axes.labelsize'] = 'x-large'
    matplotlib.rcParams['axes.labelweight'] = 'normal'
    matplotlib.rcParams['axes.labelcolor'] = 'black'
    matplotlib.rcParams['axes.formatter.limits'] = [-4, 4]

    matplotlib.rcParams['xtick.major.size'] = 7
    matplotlib.rcParams['xtick.minor.size'] = 4
    matplotlib.rcParams['xtick.major.pad'] = 6
    matplotlib.rcParams['xtick.minor.pad'] = 6
    matplotlib.rcParams['xtick.labelsize'] = 'x-large'
    matplotlib.rcParams['xtick.minor.width'] = 1.0
    matplotlib.rcParams['xtick.major.width'] = 1.0

    matplotlib.rcParams['ytick.major.size'] = 7
    matplotlib.rcParams['ytick.minor.size'] = 4
    matplotlib.rcParams['ytick.major.pad'] = 6
    matplotlib.rcParams['ytick.minor.pad'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 'x-large'
    matplotlib.rcParams['ytick.minor.width'] = 1.0
    matplotlib.rcParams['ytick.major.width'] = 1.0

    #legends
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.fontsize'] = 'large'
    matplotlib.rcParams['legend.shadow'] = False
    matplotlib.rcParams['legend.frameon'] = False

    matplotlib.rcParams['figure.autolayout'] = True
    
    