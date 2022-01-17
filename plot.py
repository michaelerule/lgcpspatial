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
plot.py: A subset of plotting helpers copied from `neurotools/graphics/plot`, and some new helper routines to reduce notebook clutter
"""

from pylab import *

def pscale(x,q1=0.5,q2=99.5,mask=True):
    '''
    ----------------------------------------------------------------------------
    Plot helper: Scale data by percentiles
    '''
    u  = x[mask] if not mask is None else x
    u  = float32(u)
    p1 = nanpercentile(u,q1)
    p2 = nanpercentile(u,q2)
    x  = clip((x-p1)/(p2-p1),0,1)
    return x*mask if not mask is None else x
    
def showkn(k,t=''):
    '''
    ----------------------------------------------------------------------------
    Plot helper; Shift convolution kernel to plot
    '''
    imshow(fftshift(k)); axis('off'); title(t);

def showim(x,t='',**kwargs):
    '''
    ----------------------------------------------------------------------------
    Plot helper: Show image with title, no axes
    '''
    if len(x.shape)==1: 
        L = int(round(sqrt(x.shape[0])))
        x = x.reshape(L,L)
    imshow(pscale(x,**kwargs));
    axis('off');
    title(t);
    
def simpleaxis(ax=None):
    '''
    Only draw the bottom and left axis lines
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)

def noaxis(ax=None):
    '''
    Hide all axes
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def nox():
    '''
    Hide x-axis
    '''
    plt.xticks([])
    plt.xlabel('')

def noy():
    '''
    Hide y-axis
    '''
    plt.yticks([])
    plt.ylabel('')

def noxyaxes():
    '''
    Hide all aspects of x and y axes. See `nox`, `noy`, and `noaxis`
    '''
    nox()
    noy()
    noaxis()

def figurebox(color=(0.6,0.6,0.6)):
    # new clear axis overlay with 0-1 limits
    from matplotlib import pyplot, lines
    ax2 = pyplot.axes([0,0,1,1],facecolor=(1,1,1,0))# axisbg=(1,1,1,0))
    x,y = np.array([[0,0,1,1,0], [0,1,1,0,0]])
    line = lines.Line2D(x, y, lw=1, color=color)
    ax2.add_line(line)
    plt.xticks([]); plt.yticks([]); noxyaxes()

def pixels_to_xfigureunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure width scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n/float(w_pixels)

def pixels_to_yfigureunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current
    figure height scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n/float(h_pixels)
    
def nudge_axis_left(dx,ax=None):
    '''
    Moves the left x-axis limit, keeping the right limit intact. 
    This changes the width of the plot.

    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w-dx,h))
    
def nudge_axis_right(dx,ax=None):
    '''
    Moves the right x-axis limit, keeping the left limit intact. 
    This changes the width of the plot.

    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x,y,w-dx,h))
    
def nudge_axis_x(dx,ax=None):
    '''
    This does not change the width of the axis

    Parameters
    ----------
    dx : number
        Amount (in pixels) to adjust axis
    ax : axis, default None
        If None, uses gca()
    '''
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = pixels_to_xfigureunits(dx,ax)
    ax.set_position((x+dx,y,w,h))
