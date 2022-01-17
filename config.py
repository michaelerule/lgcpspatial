#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
"""
Matplotlib and numpy configuration
"""

#from pylab import *
import numpy as np
import pylab as pl
import matplotlib as mpl

import numpy

def configure_pylab():
    mpl.rcParams['figure.figsize'] = (8,2.5)
    mpl.rcParams['figure.dpi']   = 240
    mpl.rcParams['image.origin'] = 'lower'
    mpl.rcParams['image.cmap']   = 'magma'

    # Fonts
    SMALL  = 6
    MEDIUM = 7
    BIGGER = 8
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