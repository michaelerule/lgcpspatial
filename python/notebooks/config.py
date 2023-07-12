#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Matplotlib and numpy configuration
"""

# Relative import regardless of file nesting depth
import os, sys
path = os.getcwd().split(os.sep)
while len(path) and path[-1]!='notebooks': path = path[:-1]
sys.path.append(os.sep.join(path[:-1]))
datadir = os.sep.join(path[:-2]+['example data'])+os.sep
datafiles = sorted([
    f for f in os.listdir(datadir) 
    if f.startswith('r') and f.endswith('.mat')])
print('Available files:\n\t'+'\n\t'.join(datafiles))
fitdir = os.sep.join(path+['hyperparameter_fits',''])
print('Fitted models in fitdir =',fitdir)


import pickle
import numpy
from lgcp.data  import *
from lgcp.kern  import *
from lgcp.infer import *
from lgcp.plot  import *
from lgcp.post  import *
from lgcp.sg    import *
from lgcp.gridsearch  import *


# Fonts
SMALL  = 7
MEDIUM = 8
BIGGER = 9
LARGE  = BIGGER
def configure_pylab():
    mpl.rcParams['figure.figsize'] = (8,2.5)
    mpl.rcParams['figure.dpi']   = 240
    mpl.rcParams['image.origin'] = 'lower'
    mpl.rcParams['image.cmap']   = 'magma'
    mpl.rcParams['font.size'           ]=SMALL  # controls default text sizes
    mpl.rcParams['axes.titlesize'      ]=MEDIUM # fontsize of the axes title
    mpl.rcParams['axes.labelsize'      ]=MEDIUM # fontsize of the x and y labels
    mpl.rcParams['xtick.labelsize'     ]=SMALL  # fontsize of the tick labels
    mpl.rcParams['ytick.labelsize'     ]=SMALL  # fontsize of the tick labels
    mpl.rcParams['legend.fontsize'     ]=SMALL  # legend fontsize
    mpl.rcParams['figure.titlesize'    ]=BIGGER # fontsize of the figure title
    mpl.rcParams['lines.solid_capstyle']='round'
    mpl.rcParams['savefig.dpi'         ]=140
    mpl.rcParams['figure.dpi'          ]=140
    mpl.rcParams['figure.facecolor'    ]='w'
    lw = .7
    mpl.rcParams['axes.linewidth'] = lw
    mpl.rcParams['xtick.major.width'] = lw
    mpl.rcParams['xtick.minor.width'] = lw
    mpl.rcParams['ytick.major.width'] = lw
    mpl.rcParams['ytick.minor.width'] = lw
    tl = 3
    mpl.rcParams['xtick.major.size'] = tl
    mpl.rcParams['xtick.minor.size'] = tl
    mpl.rcParams['ytick.major.size'] = tl
    mpl.rcParams['ytick.minor.size'] = tl
    numpy.seterr(all='ignore')
    numpy.set_printoptions(precision=10)
    numpy.seterr(divide='ignore', invalid='ignore');
configure_pylab();


# Heading plot configuration
colorNSEW = float32([
    [0.08,0.40,1.0], # North color
    [0.92,0.60,0.0], # South color
    [0.10,0.85,0.3], # East  color
    [0.90,0.15,0.7]  # West  color
])
cN,cS,cE,cW = colorNSEW
hE = 0
hS = pi/2
hW = pi
hN = 3*pi/2
NSEW = [hN,hS,hE,hW]
ESWNmix = mpl.colors.LinearSegmentedColormap.from_list('ESNWmix',float32([
    [ 72, 160, 230, 234, 220, 203, 170, 101,  56,  31,  27,  72],
    [194, 175, 155, 132,  93,  72,  75,  98, 128, 159, 193, 194],
    [ 92,  42,  12,  51, 131, 180, 196, 222, 235, 200, 138,  92]]).T/255)
ESWNmix.set_bad((1,1,1,0))

