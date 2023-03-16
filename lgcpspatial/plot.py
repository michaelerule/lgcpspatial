#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
plot.py: Some plotting helpers copied from 
``neurotools.graphics.plot``, and some new 
routines to reduce notebook clutter.
"""

import os
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from numpy import *
from matplotlib.pyplot import *

import scipy.stats
import scipy.linalg
from scipy.spatial  import ConvexHull
from .util import c2p,p2c

SMALL  = 7
MEDIUM = 8
BIGGER = 9
LARGE  = BIGGER
# │ Fonts
mpl.rcParams['figure.figsize'] = (8,2.5)
mpl.rcParams['figure.dpi']   = 240
mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['image.cmap']   = 'magma'
mpl.rcParams['font.size'           ]=SMALL  # │ controls default text sizes
mpl.rcParams['axes.titlesize'      ]=MEDIUM # │ fontsize of the axes title
mpl.rcParams['axes.labelsize'      ]=MEDIUM # │ fontsize of the x and y labels
mpl.rcParams['xtick.labelsize'     ]=SMALL  # │ fontsize of the tick labels
mpl.rcParams['ytick.labelsize'     ]=SMALL  # │ fontsize of the tick labels
mpl.rcParams['legend.fontsize'     ]=SMALL  # │ legend fontsize
mpl.rcParams['figure.titlesize'    ]=BIGGER # │ fontsize of the figure title
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


# Some custom colors, just for fun! 
WHITE      = np.float32(mpl.colors.to_rgb('#f1f0e9'))
RUST       = np.float32(mpl.colors.to_rgb('#eb7a59'))
OCHRE      = np.float32(mpl.colors.to_rgb('#eea300'))
AZURE      = np.float32(mpl.colors.to_rgb('#5aa0df'))
TURQUOISE  = np.float32(mpl.colors.to_rgb('#00bac9'))
TEAL       = TURQUOISE
BLACK      = np.float32(mpl.colors.to_rgb('#44525c'))
YELLOW     = np.float32(mpl.colors.to_rgb('#efcd2b'))
INDIGO     = np.float32(mpl.colors.to_rgb('#606ec3'))
VIOLET     = np.float32(mpl.colors.to_rgb('#8d5ccd'))
MAUVE      = np.float32(mpl.colors.to_rgb('#b56ab6'))
MAGENTA    = np.float32(mpl.colors.to_rgb('#cc79a7'))
CHARTREUSE = np.float32(mpl.colors.to_rgb('#b59f1a'))
MOSS       = np.float32(mpl.colors.to_rgb('#77ae64'))
VIRIDIAN   = np.float32(mpl.colors.to_rgb('#11be8d'))
CRIMSON    = np.float32(mpl.colors.to_rgb('#b41d4d'))
GOLD       = np.float32(mpl.colors.to_rgb('#ffd92e'))
TAN        = np.float32(mpl.colors.to_rgb('#765931'))
SALMON     = np.float32(mpl.colors.to_rgb('#fa8c61'))
GRAY       = np.float32(mpl.colors.to_rgb('#b3b3b3'))
LICHEN     = np.float32(mpl.colors.to_rgb('#63c2a3'))
RUTTEN  = [GOLD,TAN,SALMON,GRAY,LICHEN]
GATHER  = [WHITE,RUST,OCHRE,AZURE,TURQUOISE,BLACK]
COLORS  = [BLACK,WHITE,YELLOW,OCHRE,CHARTREUSE,MOSS,VIRIDIAN,
    TURQUOISE,AZURE,INDIGO,VIOLET,MAUVE,MAGENTA,RUST]
CYCLE   = [BLACK,RUST,TURQUOISE,OCHRE,AZURE,MAUVE,YELLOW,INDIGO]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CYCLE)
    

def simpleaxis(ax=None):
    '''
    Only draw the bottom and left axis lines
    
    Parameters
    ----------
    ax : optional, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)
    

def rightaxis(ax=None):
    '''
    Only draw the bottom and right axis lines.
    Move y axis to the right.
    
    Parameters
    ----------
    ax : optional, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.get_xaxis().tick_bottom()
    ax.autoscale(enable=True, axis='x', tight=True)


def simpleraxis(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)


def simpleraxis2(ax=None):
    '''
    Only draw the left y axis, nothing else
    
    Parameters
    ----------
    ax : optiona, defaults to plt.gca() if None
    '''
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
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


def axeson():
    ax=plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


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
    Hide all aspects of x and y axes. 
    See ``nox``, ``noy``, and ``noaxis``
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
    

def nudge_axis_y_pixels(dy,ax=None):
    '''
    moves axis dy pixels.
    Direction of dy may depend on axis orientation.
    Does not change axis height.

    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = -pixels_to_yfigureunits(float(dy),ax)
    ax.set_position((x,y-dy,w,h))
    

def adjust_axis_height_pixels(dy,ax=None):
    '''
    resize axis by dy pixels.
    Direction of dy may depends on axis orientation.
    Does not change the baseline position of axis.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    ax.set_position((x,y,w,h-pixels_to_yfigureunits(float(dy),ax)))
    

def get_ax_size(ax=None,fig=None):
    '''
    Gets tha axis size in figure-relative units
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax  = plt.gca()
    fig  = plt.gcf()
    ax   = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width  *= fig.dpi
    height *= fig.dpi
    return width, height


def yunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in units of the current y-axis to pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy  = np.diff(plt.ylim())[0]
    return n*float(h)/dy


def xunits_to_pixels(n,ax=None,fig=None):
    '''
    Converts a measurement in units of the current x-axis to pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx = np.diff(xlim())[0]
    return n*float(w)/dx
    

def yscalebar(ycenter,yheight,label,x=None,color='k',fontsize=9,ax=None,side='left'):
    '''
    Add vertical scale bar to plot
    
    Other Parameters
    ----------
    ax : axis, if None (default), uses the current axis.
    '''
    yspan = [ycenter-yheight/2.0,ycenter+yheight/2.0]
    if ax is None:
        ax = plt.gca()
    plt.draw() # enforce packing of geometry
    if x is None:
        x = -pixels_to_xunits(5)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    plt.plot([x,x],yspan,
        color='k',
        lw=1,
        clip_on=False)
    if side=='left':
        plt.text(x-pixels_to_xunits(2),np.mean(yspan),label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment='right',
            verticalalignment='center',
            clip_on=False)
    else:
        plt.text(x+pixels_to_xunits(5),np.mean(yspan),label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment='left',
            verticalalignment='center',
            clip_on=False)
        
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)


def xscalebar(xcenter,xlength,label,y=None,color='k',fontsize=9,ax=None):
    '''
    Add horizontal scale bar to plot
    
    Parameters
    ----------
    xcenter: float
        Horizontal center of the scale bar
    xlength: float
        How wide the scale bar is
    '''
    xspan = [xcenter-xlength/2.0,xcenter+xlength/2.0]
    if ax is None:
        ax = plt.gca()
    plt.draw() # enforce packing of geometry
    if y is None:
        y = -pixels_to_yunits(5)
    yl = ax.get_ylim()
    xl = ax.get_xlim()
    plt.plot(xspan,[y,y],
        color='k',
        lw=1,
        clip_on=False)
    plt.text(np.mean(xspan),y-pixels_to_yunits(5),label,
        fontsize=fontsize,
        horizontalalignment='center',
        verticalalignment='top',
        clip_on=False)
    ax.set_ylim(*yl)
    ax.set_xlim(*xl)
    

def arrow_between(A,B,size=None):
    '''
    Draw a solid arrow between two axes    
    '''
    draw()
    fig = plt.gcf()

    position = A.get_position().transformed(fig.transFigure)
    ax0,ay0,ax1,ay1 = position.x0,position.y0,position.x1,position.y1
    position = B.get_position().transformed(fig.transFigure)
    bx0,by0,bx1,by1 = position.x0,position.y0,position.x1,position.y1

    # arrow outline
    cx  = array([0,1.5,1.5,3,1.5,1.5,0])
    cy  = array([0,0,-0.5,0.5,1.5,1,1])
    cx -= (np.max(cx)-np.min(cx))/2
    cy -= (np.max(cy)-np.min(cy))/2
    cwidth = np.max(cx)-np.min(cx)

    horizontal = vertical = None
    if   max(ax0,ax1)<min(bx0,bx1): horizontal = -1 # axis A is to the left of axis B
    elif max(bx0,bx1)<min(ax0,ax1): horizontal =  1 # axis A is to the right of axis B
    elif max(ay0,ay1)<min(by0,by1): vertical   = -1 # axis A is above B
    elif max(by0,by1)<min(ay0,ay1): vertical   =  1 # axis A is below B
    assert not (horizontal is None     and vertical is None    )
    assert not (horizontal is not None and vertical is not None)

    if horizontal is not None:
        x0 = max(*((ax0,ax1) if horizontal==-1 else (bx0,bx1)))
        x1 = min(*((bx0,bx1) if horizontal==-1 else (ax0,ax1)))
        span     = x1 - x0
        pad      = 0.1 * span
        width    = span - 2*pad
        scale = width/cwidth if size is None else size
        px = -horizontal*cx*scale + (x0+x1)/2
        py = cy*scale + (ay0+ay1+by0+by1)/4
        polygon = Polygon(array([px,py]).T,facecolor=BLACK)#,transform=tt)
        fig.patches.append(polygon)

    if vertical is not None:
        y0 = max(*((ay0,ay1) if vertical==-1 else (by0,by1)))
        y1 = min(*((by0,by1) if vertical==-1 else (ay0,ay1)))
        span     = y1 - y0
        pad      = 0.1 * span
        width    = span - 2*pad
        scale = width/cwidth if size is None else size
        px = -vertical*cx*scale + (y0+y1)/2
        py = cy*scale + (ax0+ax1+bx0+bx1)/4
        polygon = Polygon(array([py,px]).T,facecolor=BLACK)#,transform=tt)
        fig.patches.append(polygon)
#


def nicey(**kwargs):
    '''
    Mark only the min/max value of y axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if ylim()[0]<0:
        plt.yticks([plt.ylim()[0],0,plt.ylim()[1]])
    else:
        plt.yticks([plt.ylim()[0],plt.ylim()[1]])


def nicex(**kwargs):
    '''
    Mark only the min/max value of x axis
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if xlim()[0]<0:
        plt.xticks([plt.xlim()[0],0,plt.xlim()[1]])
    else:
        plt.xticks([plt.xlim()[0],plt.xlim()[1]])


def nicexy(xby=None,yby=None,**kwargs):
    '''
    Mark only the min/max value of y/y axis. See ``nicex`` and ``nicey``
    '''
    nicex(by=xby,**kwargs)
    nicey(by=yby,**kwargs)


def hcolorbar(
    vmin=None,
    vmax=None,
    cmap=None,
    title='',
    ax=None,
    sideways=False,
    border=True,
    spacing=5,
    width=15,
    shrink=1.0,
    labelpad=10,
    fontsize=10,):
    '''
    Matplotlib's colorbar function is bad. 
    This is less bad.
    r'$\mathrm{\mu V}^2$'

    Parameters:
        vmin     (number)  : min value for colormap
        vmax     (number)  : mac value for colormap
        cmap     (colormap): what colormap to use
        title    (string)  : Units for colormap
        ax       (axis)    : optional, defaults to plt.gca(). axis to which to add colorbar
        sideways (bool)    : Flips the axis label sideways
        border   (bool)    : Draw border around colormap box? 
        spacing  (number)  : distance from axis in pixels. defaults to 5
        width    (number)  : width of colorbar in pixels. defaults to 15
        labelpad (number)  : padding between colorbar and title in pixels, defaults to 10
        fontsize (number)  : label font size, defaults to 12
    Returns:
        axis: colorbar axis
    '''
    if type(vmin)==matplotlib.image.AxesImage:
        img  = vmin
        cmap = img.get_cmap()
        vmin = img.get_clim()[0]
        vmax = img.get_clim()[1]
        ax   = img.axes
    oldax = plt.gca() #remember previously active axis
    if ax is None: ax=plt.gca()
    SPACING = pixels_to_yfigureunits(spacing,ax=ax)
    CWIDTH  = pixels_to_yfigureunits(width,ax=ax)
    # manually add colorbar axes 
    bb = ax.get_position()
    x,y,w,h,r,b = bb.xmin,bb.ymin,bb.width,bb.height
    r,b = bb.xmax,bb.ymax
    use = w*shrink
    extra = w-use
    pad = extra/2
    cax = plt.axes(
        (r-w+pad,y+SPACING,use,CWIDTH),
        facecolor='w',
        frameon=border)
    plt.sca(cax)
    plt.imshow(np.array([np.linspace(vmin,vmax,100)]),
        extent=(vmin,vmax,0,1),
        aspect='auto',
        origin='upper',
        cmap=cmap)
    noy()
    nicex()
    cax.xaxis.tick_bottom()
    if sideways:
        plt.text(
            np.mean(xlim()),
            ylim()[0]-pixels_to_yunits(labelpad,ax=cax),
            title,
            fontsize=fontsize,
            rotation=0,
            horizontalalignment='center',
            verticalalignment  ='top')
    else:
        plt.text(
            np.mean(xlim()),
            ylim()[0]-pixels_to_yunits(labelpad,ax=cax),
            title,
            fontsize=fontsize,
            rotation=90,
            horizontalalignment='center',
            verticalalignment  ='top')
    # Hide ticks
    noaxis()
    cax.tick_params(
        'both', length=0, width=0, which='major')
    cax.yaxis.set_label_position("right")
    cax.yaxis.tick_right()
    plt.sca(oldax) #restore previously active axis
    return cax


def subfigurelabel(
    x,
    fontsize=10,
    dx=39,
    dy=7,
    ax=None,
    bold=True,
    **kwargs):
    '''
    Parameters
    ----------
    x : label
    '''
    if ax is None: ax = plt.gca()
    fontproperties = {
        'fontsize':fontsize,
        'family':'Bitstream Vera Sans',
        'weight': 'bold' if bold else 'normal',
        'va':'bottom',
        'ha':'left'}
    fontproperties.update(kwargs)
    text(xlim()[0]-pixels_to_xunits(dx),ylim()[1]+pixels_to_yunits(dy),x,**fontproperties)
    

def pixels_to_xunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current x-axis
    scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dx  = np.diff(plt.xlim())[0]
    return n*dx/float(w)
px2x = pixels_to_xunits


def pixels_to_yunits(n,ax=None,fig=None):
    '''
    Converts a measurement in pixels to units of the current y-axis
    scale
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    dy = np.diff(ylim())[0]
    return n*dy/float(h)
px2y = pixels_to_yunits


def circular_gaussian_smooth(x,sigma):
    '''
    Smooth signal x with gaussian of standard deviation 
    ``sigma``, circularly wrapped using Fourier transform.
    
    Parameters
    ----------
    sigma: standard deviation
    x: 1D array-like signal
    
    Returns
    -------
    '''
    N = len(x)
    g = np.exp(-np.linspace(-N/2,N/2,N)**2/sigma**2)
    g/= np.sum(g)
    f = np.fft.fft(g)
    return np.fft.fftshift(
        np.fft.ifft(np.fft.fft(x)*f).real)


def circularly_smooth_colormap(cm,s):
    '''
    Smooth a colormap with cirular boundary conditions

    s: sigma, standard dev of gaussian smoothing kernel in 
       samples
    cm: color map, array-like of RGB tuples
    
    Parameters
    ----------
    cm : colormap
    s : smoothing radius

    Returns
    -------
    RBG : np.ndarray
        Colormap smoothed with circular boundary conditions.
    '''
    # Do circular boundary conditions the lazy way
    cm = np.array(cm)
    N = cm.shape[0]
    R,G,B = cm.T
    R = circular_gaussian_smooth(R,s)
    G = circular_gaussian_smooth(G,s)
    B = circular_gaussian_smooth(B,s)
    return np.array([R,G,B]).T

# Colors inspired by Bridget Riley's "Gather"
colors = [VIOLET,
          MAUVE,
          RUST,
          OCHRE,
          MOSS,
          TURQUOISE,
          AZURE,
          VIOLET]
riley0 = mpl.colors.LinearSegmentedColormap.from_list(
    'riley0',colors)
colors = circularly_smooth_colormap(
    np.array(riley0(np.linspace(0,1,1000)))[:,:3],45)
riley  = mpl.colors.LinearSegmentedColormap.from_list(
    'riley',colors)
if not 'riley0' in plt.colormaps():
    plt.register_cmap(name='riley0', cmap=riley0)
if not 'riley' in plt.colormaps():
    plt.register_cmap(name='riley', cmap=riley)


# Colors for North South East West Plots
cW = [0.9,.05,0.6] # Magenta
cS = [1.0,0.5,0.0] # Orange
cE = [0.1,.95,0.4] # Bright green
cN = [0.4,0.5,1.0] # Azure
colorNSEW = float32([cN,cS,cE,cW])

# New circular map matching defined colors
# Order should be W S E N to match plot.riley
# Soften these slightly by averaging w. riley
cardinal0 = mpl.colors.LinearSegmentedColormap.from_list(
    'cardinal0',[cW,cS,cE,cN,cW])
xx = np.linspace(0,1,1000)
cardinal  = mpl.colors.LinearSegmentedColormap.from_list(
    'cardinal',0.52*(riley(xx)[:,:3]+circularly_smooth_colormap(
    np.array(cardinal0(xx))[:,:3],45)))
if not 'cardinal' in plt.colormaps():
    plt.register_cmap(name='cardinal', cmap=cardinal)
cardinal

    
def force_aspect(aspect=1,a=None):
    '''
    Parameters
    ----------
    aspect : aspect ratio
    '''
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    a.set_aspect(np.abs((x2-x1)/(y2-y1))/aspect)
    
    
"""
Plotting routines specific to the GP code
"""

def pscale(x,q1=0.5,q2=99.5,mask=None):
    '''
    Plot helper: Scale data by percentiles
    
    Parameters
    ----------
    x: np.float32
    
    Other Parameters
    ----------------
    q1: float in (0,1); default 0.5
        Fraction of data below which we will set to zero.
    q2: float in (0,1); default 0.95
        Fraction of data above which we will set to one.
    mask: np.bool; default None
        Subset of points to consider 
        
    Returns
    -------
    result: np.array
        Rescaled data
    '''
    u  = x[mask] if not mask is None else x
    u  = float32(u)
    p1 = nanpercentile(u,q1)
    p2 = nanpercentile(u,q2)
    x  = clip((x-p1)/(p2-p1),0,1)
    return x*mask if not mask is None else x
    

def showkn(k,t=''):
    '''
    Plot helper; Re-center a FFT convolution kernel and
    plot it. 
    '''
    plt.imshow(fftshift(k)); axis('off'); title(t);


def showim(x,t='',**kwargs):
    '''
    Plot helper: Show image with title, no axes
    '''
    if len(x.shape)==1: 
        L = int(round(sqrt(x.shape[0])))
        x = x.reshape(L,L)
    plt.imshow(pscale(x,**kwargs));
    axis('off');
    title(t);
    

def inference_summary_plot(
    model,
    fit,
    data,
    ftitle='',
    cmap='bone_r',
    ax=None,
    caxprops=dict(fontsize=8,vscale=0.5,width=10),
    titlesize = 10,
    draw_scalebar = True
    ):
    '''
    Summarize the result of log-Gaussian process variational 
    inference in four plots.
    
    This function exists to simplify the example notebooks,
    it is not for general use. 
    
    Parameters
    ----------
    model: lgcp2d.DiagonalFourierLowrank
        Initialized model
    fit: tuple
        Tuple of (low-rank mean, marginal variance, loss)
        returned by ``lgcp2d.coordinate_descent()``.
    data: loaddata.Dataset
        Prepared dataset
    ftitle: str
        Figure title
    '''
    y       = data.y
    L       = data.L
    mask    = data.arena.mask
    nanmask = data.arena.nanmask
    Fs      = data.position_sample_rate
    scale   = data.scale
    μz      = data.prior_mean
    μh,v,_  = fit
    y = np.array(y).reshape(L,L)
    v = np.array(v).ravel()
    
    # Convert from frequency to spatial coordinates and add back prior mean
    μ  = model.F.T@μh+μz.ravel()
    μλ = np.exp(μ+v/2).reshape(L,L)*Fs
    vλ = (np.exp(2*μ + v)*(np.exp(v)-1)).reshape(L,L)*Fs**2
    σλ = np.sqrt(vλ)
    cv = σλ/μλ
    
    if isinstance(cmap,str):
        cmap = matplotlib.pyplot.get_cmap(cmap)
    
    if ax is None:
        figure(figsize=(9,2.5),dpi=120)
        subplots_adjust(left=0.0,right=0.95,top=.8,wspace=0.8)
        ax={}
        ax[1]=subplot(141)
        ax[2]=subplot(142)
        ax[3]=subplot(143)
        ax[4]=subplot(144)
    cax = {}
    
    q = 0.5/data.L
    extent = (0-q,1-q)*2

    sca(ax[1])
    toshow = y*Fs*nanmask
    vmin,vmax = nanpercentile(toshow,[1,97.5])
    vmin = floor(vmin*10)/10
    vmax = ceil (vmax*10)/10
    im = plt.imshow(y*Fs,vmin=vmin,vmax=vmax,cmap=cmap,extent=extent)
    title('Rate histogram',pad=0,fontsize=titlesize)
    axis('off')
    
    # Add a scale bar
    if draw_scalebar:
        x0 = where(any(mask,axis=0))[0][0]
        x1 = x0 + scale*(L+1)
        y0 = where(any(mask,axis=1))[0][0]-L//25
        xscalebar((x0+x1)/2/L,(x1-x0)/L,'1 m',y=y0/L)
    
    # Add color bar with marker for neurons mean-rate
    cax[1] = good_colorbar(vmin,vmax,title='Hz',cmap=cmap,**caxprops)
    sca(cax[1])
    μy = mean(y[mask])*Fs
    #axhline(μy,color='w',lw=0.8)
    #text(xlim()[1]+.6,μy,r'$\langle y \rangle$:%0.2f Hz'%μy,va='center',fontsize=7)

    sca(ax[3])
    #vmin,vmax = nanpercentile(μλ[mask],[.5,99.5])
    #vmin = floor(vmin*10)/10
    #vmax = ceil (vmax*10)/10
    plt.imshow(
        μλ.reshape(L,L)*nanmask,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent)
    plt.plot(*data.arena.perimeter.T,lw=2,color='w')
    title('Mean Rate',pad=0,fontsize=titlesize)
    cax[3]=good_colorbar(vmin,vmax,title='Hz',cmap=cmap,**caxprops)
    axis('off')

    sca(ax[2])
    toshow = 10*log10(exp(μ-μz.ravel())).reshape(L,L)*nanmask
    vmin,vmax = nanpercentile(toshow,[0,100])
    vmax += (vmax-vmin)*.2
    vmin = floor(vmin*10)/10
    vmax = ceil (vmax*10)/10
    im = plt.imshow(
        toshow,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent)
    plt.plot(*data.arena.perimeter.T,lw=2,color='w')
    title('Log-Rate (minus background)',pad=0,fontsize=titlesize)
    axis('off')
    cax[2]=good_colorbar(vmin,vmax,title='ΔdB',labelpad=30,cmap=cmap,**caxprops)
    
    sca(ax[4])
    toshow = cv*nanmask
    vmin,vmax = nanpercentile(toshow,[.5,99.5])
    vmin = floor(vmin*100)/100
    vmax = ceil (vmax*100)/100
    im = plt.imshow(
        toshow,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent)
    plt.plot(*data.arena.perimeter.T,lw=2,color='w')
    title('Marginal c.v. of λ (σ/μ)',pad=0,fontsize=titlesize)
    cax[4] = good_colorbar(vmin,vmax,title='σ/μ',cmap=cmap,**caxprops)
    axis('off')
    
    suptitle(ftitle)
    return ax,cax


def plot_convex_hull(px,py,**kwargs):
    '''
    Calculate convex hull of points (px,py) and plot it.
    
    Parameters
    ----------
    px: length NPOINTS 1D np.float32
    py: length NPOINTS 1D np.float32
    
    Other Parameters
    ----------------
    **kwargs: dict
        Keyword arguments are forwarded to ``pyplot.plot()``        
        
    Returns
    -------
    result: ConvexHull(points)
    '''
    points = array([px,py]).T
    hull   = ConvexHull(points)
    vv     = concatenate([hull.vertices,hull.vertices[:1]])
    plot(points[vv,0],points[vv,1],**kwargs)
    return hull


def unit_crosshairs(draw_ellipse=True,draw_cross=True):
    '''
    '''
    lines  = []
    if draw_ellipse:
        circle = np.exp(1j*np.linspace(0,2*np.pi,361))
        lines += list(circle)
    if draw_cross==True:
        lines += [np.nan]+list(1.*np.linspace(-1,-.4,2))
        lines += [np.nan]+list(1j*np.linspace(-1,-.4,2))
        lines += [np.nan]+list(1.*np.linspace( 1, .4,2))
        lines += [np.nan]+list(1j*np.linspace( 1, .4,2))
    elif draw_cross=='full':
        lines += [np.nan]+list(1.*np.linspace(-1,1,10))
        lines += [np.nan]+list(1j*np.linspace(-1,1,10))
    lines = np.array(lines)
    return np.array([lines.real,lines.imag])

    
def covellipse_from_points(q,**kwargs):
    '''
    Helper to construct covariance ellipse from 
    collection of 2D points ``q`` encoded as complex numbers.
    '''
    q = q[isfinite(q)]
    pxy = c2p(q)
    μ = mean(pxy,1)
    Δ = pxy-μ[:,None]
    Σ = mean(Δ[:,None,:]*Δ[None,:,:],2)
    return p2c(covariance_crosshairs(Σ,**kwargs) + μ[:,None])


def covariance_crosshairs(
    S,
    p=0.95,
    draw_ellipse=True,
    draw_cross=True,
    mode='2D'):
    '''
    For 2D Gaussians these are the confidence intervals
    
        p   | sigma
        90  : 4.605
        95  : 5.991
        97.5: 7.378
        99  : 9.210
        99.5: 10.597

    - Get the eigenvalues and vectors of covariance S
    - Prepare crosshairs for a unit standard normal
    - Transform crosshairs into covariance basis

    Parameters
    ----------
    S: 2D covariance matrix
    p: fraction of data ellipse should enclose
    draw_ellipse: whether to draw the ellipse
    draw_cross: whether to draw the crosshairs
    mode: 2D or 1D; 
        If 2D, ellipse will reflect inner p of probability
        mass
        If 1D, ellipse will represent threshold for p 
        significance for a 1-tailed test in any direction.
    
    Returns
    -------
    path: np.float32
        2×NPOINTS array of (x,y) path data for plotting
    
    '''
    S = np.float32(S)
    if not S.shape==(2,2):
        raise ValueError('S should be a 2x2 covariance '
                         'matrix, got %s'%S)
    if mode=='2D':
        # Points in any direction within percentile
        sigma = sqrt(scipy.stats.chi2.ppf(p,df=2))
    elif mode=='1D':
        # Displacement in specific direction within %ile
        sigma = scipy.stats.norm.ppf(p)
    else:
        raise ValueError("Mode must be '2D' or '1D'")
    try:
        e,v   = scipy.linalg.eigh(S)
    except:
        raise ValueError('Could not get covariance '
            'eigenspace, is S=%s full-rank?'%repr(S))
    e     = np.maximum(0,e)
    lines = unit_crosshairs(draw_ellipse,draw_cross)*sigma
    return v.T.dot(lines*(e**0.5)[:,None])


def good_colorbar(vmin=None,
    vmax=None,
    cmap=None,
    title='',
    ax=None,
    sideways=False,
    border=True,
    spacing=5,
    width=15,
    labelpad=10,
    fontsize=10,
    vscale=1.0,
    va='c'):
    '''
    Matplotlib's colorbar function is pretty bad. 
    This is less bad.
    r'$\mathrm{\mu V}^2$'

    Parameters:
        vmin: scalar
            min value for colormap
        vmax: scalar
            max value for colormap
        cmap: Matplotlib.colormap
            what colormap to use
        title: str
            Units for colormap
        ax: axis
            Optional, defaults to plt.gca(). 
            Axis to which to add colorbar
        sideways: bool
            Flip the axis label sideways?
        border: bool
            Draw border around colormap box? 
        spacing: scalar
            Distance from axis in pixels. 
            defaults to 5
        width: scalar
            Width of colorbar in pixels. 
            Defaults to 15
        labelpad: scalar
            Padding between colorbar and title in pixels, 
            defaults to 10
        fontsize: scalar
            Label font size, defaults to 12
        vscale: float
            Height adjustment relative to parent axis, 
            defaults to 1.0
        va: str
            Verrtical alignment; "bottom" ('b'), 
            "center" ('c'), or "top" ('t')
    Returns:
        axis: colorbar axis
    '''
    if type(vmin)==matplotlib.image.AxesImage:
        img  = vmin
        cmap = img.get_cmap()
        vmin = img.get_clim()[0]
        vmax = img.get_clim()[1]
        ax   = img.axes
    oldax = plt.gca() #remember previously active axis
    if ax is None: ax=plt.gca()
    SPACING = pixels_to_xfigureunits(spacing,ax=ax)
    CWIDTH  = pixels_to_xfigureunits(width,ax=ax)
    # manually add colorbar axes 
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    r,b = bb.xmax,bb.ymax
    y0 = {
        'b':lambda:b-h,
        'c':lambda:b-(h+h*vscale)/2,
        't':lambda:b-h*vscale
    }[va[0]]()
    cax = plt.axes(
        (r+SPACING,y0,CWIDTH,h*vscale),frameon=True)
    plt.sca(cax)
    plt.imshow(np.array([np.linspace(vmax,vmin,100)]).T,
        extent=(0,1,vmin,vmax),
        aspect='auto',
        origin='upper',
        cmap=cmap)
    plt.box(False)
    nox()
    nicey()
    cax.yaxis.tick_right()
    if sideways:
        plt.text(
            xlim()[1]+pixels_to_xunits(labelpad,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=0,
            horizontalalignment='left',
            verticalalignment  ='center')
    else:
        plt.text(
            xlim()[1]+pixels_to_xunits(labelpad,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=90,
            horizontalalignment='left',
            verticalalignment  ='center')
    # Hide ticks
    #noaxis()
    cax.tick_params(
        'both', length=0, width=0, which='major')
    cax.yaxis.set_label_position("right")
    cax.yaxis.tick_right()
    plt.sca(oldax) #restore previously active axis
    return cax

def draw_compass(
    xy0=1.1+.65j,
    r=0.03,
    cmap_fn=riley,
    delta=18):
    '''
    Draw a hue-wheel compass rose.
    Equivalent of "colorbar" for polar data.
    
    This uses plot-coordinates for drawing, and assumes
    that the plot's aspect-ratio is equal. 


    This is designed to be used with the ``riley`` 
    colormap from ``lgcpspatial.plot``. This map starts at 
    mauve, continues through rust, olive, blue, before 
    circling back to mauve. We use these color–direction
    conventions: 
    
     - North: blue/azure/cyan
     - South: red/rust
     - East:  green/olive
     - West:  purple/mauve/magenta
    
    For comatibility, then, the direction ordering
    for the colormap parameter ``color`` should be
    
        {West, South, East, North}

    
    Parameters
    ----------
    xy0: np.complex64
        Center of compass encoded as x+iy complex number.
    r: positive float; default 0.03
        Size of compass (radius of circle).
    cmap_fn: matplotlib.colors.Colormap
        circular colormap to use for compass.
    delta: positive float
        Radial spacing for NSEW labels relative to the
        color wheel. 
    '''
    y0,y1 = ylim()
    flip = -1 if y1<y0 else 1
    for i,θi in enumerate(linspace(0,2*pi,180)+pi):
        scatter(*c2p(xy0+r*exp(flip*1j*θi)),
                marker='o',
                lw=0,s=7,
                color=cmap_fn(i/180),
                clip_on=False)
    tx,ty = c2p(xy0)
    dx,dy = abs(px2x(delta))+r,abs(px2y(delta))+r
    # There's a bug here; assume axis is always square
    dx = dy = (dx+dy)/2
    dy*=flip
    text(tx,ty+dy,'N',ha='center',va='center')
    text(tx,ty-dy,'S',ha='center',va='center')
    text(tx+dx,ty,'E',ha='center',va='center')
    text(tx-dx,ty,'W',ha='center',va='center')

roma_cm_data = [[.49684,0.099626,0],[.50141,0.11159,0.0038271],[.50595,0.12281,0.0075362],[.51049,0.13362,0.011176],[.51502,0.14397,0.014691],[.51953,0.15397,0.017936],[.52403,0.16373,0.02102],[.52851,0.17319,0.023941],[.53298,0.18247,0.026768],[.53742,0.19159,0.02974],[.54181,0.20053,0.032886],[.5462,0.20938,0.036371],[.55055,0.2181,0.03985],[.55486,0.22671,0.043275],[.55915,0.23522,0.046908],[.56342,0.24362,0.050381],[.56764,0.25199,0.053844],[.57184,0.26026,0.057332],[.57601,0.26848,0.06081],[.58015,0.27665,0.064287],[.58425,0.28474,0.067738],[.58833,0.29281,0.071178],[.59239,0.30083,0.074547],[.59641,0.30883,0.077891],[.60041,0.31677,0.081344],[.60439,0.32468,0.084709],[.60834,0.33258,0.088059],[.61225,0.34043,0.091425],[.61616,0.34827,0.094762],[.62004,0.35608,0.098071],[.6239,0.36387,0.10135],[.62774,0.37164,0.10462],[.63157,0.37941,0.10794],[.63537,0.38716,0.11129],[.63916,0.3949,0.11452],[.64294,0.40263,0.11786],[.64671,0.41037,0.12112],[.65046,0.41807,0.12443],[.65421,0.42579,0.12776],[.65795,0.43351,0.13108],[.66169,0.44123,0.13442],[.66541,0.44896,0.13776],[.66914,0.45668,0.14112],[.67287,0.46443,0.1445],[.6766,0.47218,0.1479],[.68033,0.47994,0.15134],[.68407,0.48772,0.15484],[.68783,0.49551,0.1584],[.69159,0.50332,0.162],[.69537,0.51117,0.1656],[.69916,0.51905,0.16938],[.70298,0.52695,0.17315],[.70681,0.53488,0.17706],[.71067,0.54285,0.18103],[.71456,0.55087,0.18517],[.71848,0.55892,0.18939],[.72244,0.56704,0.19377],[.72644,0.57519,0.19826],[.73049,0.5834,0.20295],[.73457,0.59168,0.20781],[.7387,0.60002,0.21285],[.74289,0.60842,0.21813],[.74712,0.61688,0.22361],[.75142,0.6254,0.22933],[.75576,0.634,0.23533],[.76016,0.64265,0.24156],[.76463,0.65137,0.24809],[.76914,0.66014,0.2549],[.77371,0.66897,0.262],[.77833,0.67784,0.26943],[.783,0.68674,0.27716],[.78771,0.69568,0.2852],[.79246,0.70462,0.29358],[.79722,0.71357,0.30227],[.80201,0.72249,0.31128],[.80681,0.73138,0.32059],[.81159,0.74021,0.33017],[.81635,0.74896,0.34004],[.82108,0.75761,0.35015],[.82576,0.76614,0.36047],[.83037,0.77452,0.37103],[.8349,0.78274,0.38176],[.83934,0.79077,0.39264],[.84366,0.7986,0.40365],[.84785,0.80619,0.41475],[.8519,0.81354,0.42591],[.8558,0.82064,0.43711],[.85953,0.82748,0.44831],[.86308,0.83404,0.4595],[.86643,0.84031,0.47065],[.86958,0.84629,0.48173],[.87253,0.85199,0.49272],[.87526,0.8574,0.50362],[.87777,0.86254,0.51441],[.88004,0.86739,0.52506],[.88209,0.87197,0.53557],[.8839,0.87629,0.54595],[.88546,0.88035,0.55615],[.88677,0.88417,0.56622],[.88783,0.88775,0.57613],[.88864,0.89111,0.58587],[.88918,0.89426,0.59544],[.88946,0.8972,0.60485],[.88947,0.89994,0.61409],[.88921,0.9025,0.62319],[.88867,0.90488,0.6321],[.88785,0.90709,0.64085],[.88674,0.90914,0.64945],[.88534,0.91104,0.65787],[.88364,0.91279,0.66612],[.88165,0.9144,0.67421],[.87934,0.91587,0.68212],[.87673,0.91722,0.68988],[.87381,0.91842,0.69745],[.87058,0.9195,0.70485],[.86703,0.92046,0.71207],[.86316,0.92129,0.71912],[.85897,0.92201,0.72598],[.85447,0.9226,0.73266],[.84965,0.92307,0.73915],[.84452,0.92342,0.74544],[.83906,0.92365,0.75155],[.8333,0.92375,0.75746],[.82723,0.92373,0.76318],[.82086,0.92358,0.7687],[.81418,0.9233,0.77403],[.80722,0.92289,0.77916],[.79997,0.92234,0.7841],[.79243,0.92166,0.78883],[.78462,0.92082,0.79337],[.77654,0.91986,0.79771],[.7682,0.91873,0.80185],[.7596,0.91747,0.80581],[.75077,0.91603,0.80957],[.74169,0.91444,0.81313],[.7324,0.91268,0.81651],[.72287,0.91075,0.8197],[.71314,0.90865,0.8227],[.70322,0.90636,0.82551],[.69311,0.90389,0.82814],[.68283,0.90124,0.83059],[.67239,0.89839,0.83284],[.6618,0.89535,0.83492],[.65107,0.89211,0.83682],[.64024,0.88868,0.83853],[.6293,0.88504,0.84006],[.61828,0.8812,0.84141],[.60721,0.87716,0.84258],[.59608,0.87292,0.84357],[.58494,0.86849,0.84438],[.57379,0.86386,0.84502],[.56267,0.85903,0.84548],[.55159,0.85402,0.84576],[.54058,0.84884,0.84588],[.52966,0.84347,0.84582],[.51886,0.83795,0.8456],[.50819,0.83227,0.84522],[.49767,0.82643,0.84467],[.48733,0.82046,0.84397],[.47718,0.81436,0.84312],[.46725,0.80814,0.84213],[.45755,0.8018,0.84099],[.44809,0.79537,0.83973],[.43889,0.78885,0.83833],[.42997,0.78225,0.83681],[.42131,0.77557,0.83517],[.41296,0.76883,0.83343],[.40486,0.76204,0.83159],[.39707,0.75521,0.82964],[.38957,0.74833,0.82761],[.38235,0.74142,0.82549],[.37542,0.7345,0.8233],[.36877,0.72754,0.82104],[.36238,0.72058,0.8187],[.35627,0.71361,0.81632],[.3504,0.70664,0.81387],[.34477,0.69966,0.81138],[.33939,0.69269,0.80884],[.33422,0.68572,0.80626],[.32926,0.67875,0.80364],[.32448,0.67181,0.801],[.31992,0.66486,0.79832],[.31551,0.65795,0.79562],[.31127,0.65104,0.7929],[.30718,0.64414,0.79015],[.30322,0.63727,0.78739],[.29942,0.63042,0.78462],[.29571,0.62358,0.78184],[.29213,0.61676,0.77904],[.28864,0.60995,0.77624],[.28523,0.60318,0.77343],[.28193,0.59642,0.77063],[.2787,0.58967,0.76781],[.27554,0.58296,0.765],[.27241,0.57628,0.76218],[.26939,0.56959,0.75937],[.26638,0.56295,0.75656],[.26345,0.5563,0.75375],[.26053,0.5497,0.75095],[.25766,0.54311,0.74814],[.25486,0.53655,0.74534],[.25205,0.53,0.74255],[.24928,0.52347,0.73977],[.24654,0.51697,0.73698],[.24382,0.51048,0.73421],[.24114,0.50402,0.73143],[.23846,0.49758,0.72867],[.23583,0.49117,0.72592],[.23317,0.48475,0.72317],[.23056,0.47838,0.72043],[.22798,0.47202,0.7177],[.22538,0.46567,0.71496],[.22282,0.45936,0.71224],[.22026,0.45306,0.70953],[.2177,0.44678,0.70682],[.21514,0.44051,0.70412],[.21262,0.43427,0.70142],[.21009,0.42806,0.69874],[.20758,0.42184,0.69606],[.20507,0.41566,0.69339],[.20256,0.4095,0.69071],[.20005,0.40335,0.68806],[.19757,0.3972,0.6854],[.19509,0.3911,0.68275],[.19259,0.38498,0.6801],[.1901,0.3789,0.67748],[.18765,0.37283,0.67484],[.18515,0.36678,0.67222],[.18262,0.36073,0.66959],[.18013,0.35472,0.66697],[.17766,0.3487,0.66436],[.17513,0.34271,0.66176],[.17259,0.33671,0.65915],[.17007,0.33072,0.65656],[.16752,0.32475,0.65396],[.16494,0.3188,0.65137],[.16238,0.31285,0.64878],[.15974,0.30691,0.64619],[.15712,0.30097,0.64361],[.15446,0.29504,0.64103],[.15176,0.28914,0.63846],[.14904,0.28322,0.63589],[.14627,0.2773,0.63331],[.14346,0.27138,0.63075],[.1406,0.26547,0.62818],[.13769,0.25957,0.62561],[.1347,0.25365,0.62305],[.13163,0.24774,0.62049],[.12849,0.24182,0.61792],[.12528,0.2359,0.61535],[.12194,0.22993,0.61279],[.11859,0.22399,0.61023],[.11502,0.21805,0.60768],[.11142,0.21209,0.60511],[.10761,0.20611,0.60255],[.1037,0.20006,0.59999]]

roma   = mpl.colors.LinearSegmentedColormap.from_list('roma',roma_cm_data)
roma_r = mpl.colors.LinearSegmentedColormap.from_list('roma_r',roma_cm_data[::-1])
if not 'roma' in plt.colormaps():
    plt.register_cmap(name='roma', cmap=roma)
if not 'roma_r' in plt.colormaps():
    plt.register_cmap(name='roma_r', cmap=roma_r)

    
############################################################
############################################################
############################################################
############################################################

from .util import is_in_hull
def tracking_match_plot(
    data,
    centroids, 
    paths,
    fieldcolor=MAUVE
    ):
    ok = {*where(
        is_in_hull(centroids,data.arena.hull)
    )[0]}
    ok = sorted(list(ok))
    scatter(*centroids[:,ok],facecolor=(0,)*4,edgecolor=fieldcolor,lw=0.5,s=150)
    for i,c in enumerate(centroids.T):
        if not i in ok: continue
        tx,ty = c
        text(tx,ty,str(i),ha='center',va='center',color=fieldcolor,fontsize=8)
    for i,p in enumerate(paths):
        plot(*p.T,color='k',lw=.6)
        center = c2p(nanmean(p2c(p)))
        #scatter(*center,marker='x',lw=.5,color='r')
        tx,ty = center
        text(tx+px2x(10),ty+px2y(10),str(i))

    plot(*data.arena.perimeter.T,color='k')
    noxyaxes()
    xlim(0,1)
    ylim(0,1)
    force_aspect()
    
def shiftplot(
    z1,s1,z2,s2,
    centers = None,
    zscore = True,
    scale = 0.01,
    draw_ellipse = True,
    draw_lines = True,
    line_color = 'r',
    line_width = 1.0,
    **kwargs
    ):
    s  = s1 + s2
    kwargs = {'color':'k','lw':.6,**kwargs}
    lines = []
    ellipses = []
    for ii,(si,za,zb) in enumerate(zip(s,z1,z2)):
        if not np.all(np.isfinite(si)): continue
        if centers is None:
            z0 = (za+zb)/2
        else:
            z0 = centers[ii]
        if zscore:
            delta = get_whitener(si)@(zb-za)*scale
            cxy = covariance_crosshairs(eye(2),draw_cross=False)*scale
        else:
            delta = zb-za
            cxy = covariance_crosshairs(si,draw_cross=False)
        cxy += z0[:,None]
        
        lines.extend(float32([z0-delta,z0+delta]))
        lines.append([NaN,NaN])
        ellipses.extend(cxy.T)
        ellipses.append([NaN,NaN])
    if draw_lines:
        plot(*float32(lines).T,color=line_color,lw=line_width)
    if draw_ellipse:
        plot(*float32(ellipses).T,**kwargs)
    xlim(0,1)
    ylim(0,1)
    force_aspect()
    noxyaxes()

import lgcpspatial
def colored_shift_density_images(
    data, 
    NSEW_densities, 
    pct  = 99.9 # Saturate pixels above this
    ):
    '''
    Prepare colored (North - South) and (East - West)
    peak density shift images.
    
    Parameters
    ----------
    data: lgcpspatial.loaddata.Dataset
        Prepared dataset object.
    NSEW_densities: list
        Length 4 list of [N,S,E,W] results, each containing
        a ``(L*resolution)×(L*resolution)`` smoothed 
        peak-density map.
    
    Other Parameters
    ----------------
    pct: float ∈(0,100); default 99.9
        Percentile of maximum saturation.
    
    Returns
    -------
    RGBNS:
        Rendered RGB values for NS plot
    RGBEW:
        Rendered RGB values for EW plot
    '''
    
    # Unpack reference result
    L = data.L
    N,S,E,W = NSEW_densities
    
    LL = N.shape[0]
    resolution = LL//L
    
    mask = lgcpspatial.loaddata.Arena(data.px,data.py,L,resolution).mask

    colors  = 1.-array([cE,cW])
    RGBEW   = 1.-clip(einsum('dc,dxy->xyc',colors,
        array([
            (E/np.percentile(E,pct)),
            (W/np.percentile(W,pct))])),0,1)
    RGBEW[~mask,...] = 1
    colors  = 1.-array([cN,cS])
    RGBNS   = 1.-clip(einsum('dc,dxy->xyc',colors,
        array([
            (N/np.percentile(N,pct)),
            (S/np.percentile(S,pct))])),0,1)
    RGBNS[~mask,...] = 1
    return RGBNS, RGBEW
