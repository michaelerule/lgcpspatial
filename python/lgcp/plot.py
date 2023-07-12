#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Plotting helpers copied from 
``neurotools.graphics.plot``, and some new 
routines to reduce notebook clutter.
"""

from pylab import *
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    

def noaxis(ax=None):
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
def nox():
    plt.xticks([]); plt.xlabel('')
def noy():
    plt.yticks([]); plt.ylabel('')
def noxyaxes():
    nox(); noy(); noaxis()
def figurebox(color=(0.6,0.6,0.6)):
    from matplotlib import pyplot, lines
    ax2 = pyplot.axes([0,0,1,1],facecolor=(1,1,1,0))# axisbg=(1,1,1,0))
    x,y = np.array([[0,0,1,1,0], [0,1,1,0,0]])
    line = lines.Line2D(x, y, lw=1, color=color)
    ax2.add_line(line)
    plt.xticks([]); plt.yticks([]); noxyaxes()
def simpleaxis(ax=None):
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)
def rightaxis(ax=None):
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.get_xaxis().tick_bottom()
    ax.autoscale(enable=True, axis='x', tight=True)
def simpleraxis(ax=None):
    if ax is None: ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.autoscale(enable=True, axis='x', tight=True)
def px2xf(n,ax=None,fig=None):
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w_pixels = fig.get_size_inches()[0]*fig.dpi
    return n/float(w_pixels)
def px2yf(n,ax=None,fig=None):
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    h_pixels = fig.get_size_inches()[1]*fig.dpi
    return n/float(h_pixels)
def nudge_axis_left(dx,ax=None):
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = px2xf(dx,ax)
    ax.set_position((x+dx,y,w-dx,h))
def nudge_axis_right(dx,ax=None):
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = px2xf(dx,ax)
    ax.set_position((x,y,w-dx,h))
def nudge_axis_x(dx,ax=None):
    if ax is None: ax = plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dx = px2xf(dx,ax)
    ax.set_position((x+dx,y,w,h))
def nudge_axis_y_pixels(dy,ax=None):
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    dy = -px2yf(float(dy),ax)
    ax.set_position((x,y-dy,w,h))
def adjust_axis_height_pixels(dy,ax=None):
    if ax is None: ax=plt.gca()
    bb = ax.get_position()
    x,y,w,h = bb.xmin,bb.ymin,bb.width,bb.height
    ax.set_position((x,y,w,h-px2yf(float(dy),ax)))
def nicey(**kwargs):
    if ylim()[0]<0: plt.yticks([plt.ylim()[0],0,plt.ylim()[1]])
    else:           plt.yticks([plt.ylim()[0],plt.ylim()[1]])
def nicex(**kwargs):
    if xlim()[0]<0: plt.xticks([plt.xlim()[0],0,plt.xlim()[1]])
    else:           plt.xticks([plt.xlim()[0],plt.xlim()[1]])
def nicexy(xby=None,yby=None,**kwargs):
    nicex(by=xby,**kwargs)
    nicey(by=yby,**kwargs)
def right_legend(*args,fudge=0.0,**kwargs):
    defaults = {'loc':'center left','bbox_to_anchor':(1+fudge,0.5),}
    defaults.update(kwargs)
    lg = legend(*args,**defaults)
    lg.get_frame().set_linewidth(0.0)
    return lg
def xticklen(l=0,w=None,ax=None,which='both',**kwargs):
    if ax is None: ax = plt.gca()
    ax.xaxis.set_tick_params(length=l, width=w, which=which, **kwargs)
def yticklen(l=0,w=None,ax=None,which='both',**kwargs):
    if ax is None: ax = plt.gca()
    ax.yaxis.set_tick_params(length=l, width=w, which=which, **kwargs)
def noclip(ax=None):
    if ax is None: ax = plt.gca()
    for o in ax.findobj(): o.set_clip_on(False)
def get_ax_size(ax=None,fig=None):
    '''Gets tha axis size in figure-relative units'''
    if fig is None: fig = plt.gcf()
    if ax is None: ax  = plt.gca()
    fig  = plt.gcf()
    ax   = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width  *= fig.dpi
    height *= fig.dpi
    return width, height
def px2x(n,ax=None,fig=None):
    '''Convert pixels to units of the current x-axis.'''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    return n*np.diff(plt.xlim())[0]/float(w)
def px2y(n,ax=None,fig=None):
    '''Convert pixels to units of the current y-axis scale.'''
    if fig is None: fig = plt.gcf()
    if ax  is None: ax  = plt.gca()
    w,h = get_ax_size()
    return n*np.diff(plt.ylim())[0]/float(h)
def force_aspect(aspect=1,a=None):
    if a is None: a = plt.gca()
    x1,x2=a.get_xlim()
    y1,y2=a.get_ylim()
    a.set_aspect(np.abs((x2-x1)/(y2-y1))/aspect)
def lighten(color,amount=0.2):
    color = np.float32(mpl.colors.to_rgb(color))
    amount = np.clip(float(amount),0,1)
    color = 1.0 * amount + (1-amount) * color
    return color
def darken(color,amount=0.2):
    color = np.float32(mpl.colors.to_rgb(color))
    amount = np.clip(float(amount),0,1)
    color = (1-amount) * color
    return color
def subfigurelabel(x,fontsize=10,dx=39,dy=7,ax=None,bold=True,**kwargs):
    if ax is None: ax = plt.gca()
    fontproperties = {
        'fontsize':fontsize,
        'family':'Bitstream Vera Sans',
        'weight': 'bold' if bold else 'normal',
        'va':'bottom',
        'ha':'left'}
    fontproperties.update(kwargs)
    plt.text(
        plt.xlim()[0]-px2x(dx),
        plt.ylim()[1]+px2y(dy),
        x,**fontproperties)
def yscalebar(ycenter,yheight,label,x=None,color='k',fontsize=9,ax=None,side='left'):
    '''Add vertical scale bar to plot'''
    yspan = [ycenter-yheight/2.0,ycenter+yheight/2.0]
    if ax is None: ax = plt.gca()
    plt.draw()
    if x is None: x = -px2xf(5)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    plt.plot([x,x],yspan,
        color='k', 
        lw=1,
        clip_on=False)
    if side=='left':
        plt.text(x-px2xf(2),np.mean(yspan),label,
            rotation=90,
            fontsize=fontsize,
            horizontalalignment='right',
            verticalalignment='center',
            clip_on=False)
    else:
        plt.text(x+px2xf(5),np.mean(yspan),label,
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
    Args:    
        xcenter (float): Horizontal center of the scale bar
        xlength (float): How wide the scale bar is
    '''
    xspan = [xcenter-xlength/2.0,xcenter+xlength/2.0]
    if ax is None:
        ax = plt.gca()
    plt.draw() # enforce packing of geometry
    if y is None:
        y = -px2y(5)
    yl = ax.get_ylim()
    xl = ax.get_xlim()
    plt.plot(xspan,[y,y],
        color='k',
        lw=1,
        clip_on=False)
    plt.text(np.mean(xspan),y-px2y(5),label,
        fontsize=fontsize,
        horizontalalignment='center',
        verticalalignment='top',
        clip_on=False)
    ax.set_ylim(*yl)
    ax.set_xlim(*xl)
    

def inference_summary_plot(
    model,
    data          = None,
    fit           = None,
    ftitle        = '',
    cmap          = 'bone_r',
    ax            = None,
    caxprops      = dict(fontsize=8,vscale=0.5,width=10),
    titlesize     = 10,
    draw_scalebar = True
    ):
    '''
    Summarize the result of log-Gaussian process variational 
    inference in four plots.
    
    This function exists to simplify the example notebooks,
    it is not for general use. 
    
    Parameters
    ----------
    model: lgcp2d.LGCPResult
        Result returned by ``lgcp2d.lgcpregress()``
        
    Other Parameters
    ----------------
    ftitle: str; default ''
        Figure title
    '''
    from lgcpspatial.lgcp2d import LGCPResult
    if isinstance(model, LGCPResult):
        data  = model.data
        fit   = model.fit
        model = model.model
    
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
    
    if isinstance(μz, np.ndarray):
        μz = μz.ravel()
    
    # Convert from frequency to spatial coordinates and add back prior mean
    μ  = model.F.T@μh+μz
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
    toshow = 10*log10(exp(μ-μz)).reshape(L,L)*nanmask
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

def unit_crosshairs(draw_ellipse=True,draw_cross=True):
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
    if type(vmin)==mpl.image.AxesImage:
        img  = vmin
        cmap = img.get_cmap()
        vmin = img.get_clim()[0]
        vmax = img.get_clim()[1]
        ax   = img.axes
    oldax = plt.gca() #remember previously active axis
    if ax is None: ax=plt.gca()
    SPACING = px2xf(spacing,ax=ax)
    CWIDTH  = px2xf(width,ax=ax)
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
            xlim()[1]+px2xf(labelpad,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=0,
            horizontalalignment='left',
            verticalalignment  ='center')
    else:
        plt.text(
            xlim()[1]+px2xf(labelpad,ax=cax),
            np.mean(ylim()),
            title,
            fontsize=fontsize,
            rotation=90,
            horizontalalignment='left',
            verticalalignment  ='center')
    # Hide ticks
    cax.tick_params(
        'both', length=0, width=0, which='major')
    cax.yaxis.set_label_position("right")
    cax.yaxis.tick_right()
    plt.sca(oldax) #restore previously active axis
    return cax

def colored_boxplot(
    data,
    positions,
    color,
    filled      = True,
    notch       = False,
    showfliers  = False,
    lw          = 1,
    whis        = [5,95],
    bgcolor     = WHITE,
    mediancolor = None,
    **kwargs):
    '''
    Boxplot with nicely colored default style parameters
    
    Parameters
    ----------
    data: NPOINTS × NGROUPS np.float32
        Data sets to plot
    positions: NGROUPS iterable of numbers
        X positions of each data group
    color: matplotlib.color
        Color of boxplot   
    
    Other Parameters
    ----------------
    filled: boolean; default True
        Whether to fill boxes with color
    notch: boolean; default False
        Whether to inset a median notch
    showfliers: boolean; default False
        Whether to show outlies as scatter points
    lw: positive float; default 1.
        Width of whisker lines
    which: tuple; default (5,95)
        Percentile range for whiskers
    bgcolor: matplotlib.color; default WHITE
        Background color if ``filled=False``
    mediancolor: matplotlib.color; default None
        Defaults to BLACK unless color is BLACK, in which
        case it defaults to WHITE.
    **kwargs:
        Additional arguments fowarded to ``pyplot.boxplot()``
    '''
    if 'linewidth' in kwargs:
        lw = kwargs[linewidth]
    b = matplotlib.colors.to_hex(BLACK)
    if mediancolor is None:
        try:
            mediancolor = [
                BLACK if matplotlib.colors.to_hex(c)!=b \
                else WHITE for c in color]
        except:
            mediancolor = BLACK \
                if matplotlib.colors.to_hex(color)!=b \
                else WHITE
    bp = plt.boxplot(data,
        positions    = positions,
        patch_artist = True,
        showfliers   = showfliers,
        notch        = notch,
        whis         = whis, 
        medianprops  = {'linewidth':lw,'color':mediancolor},
        whiskerprops = {'linewidth':lw,'color':color},
        flierprops   = {'linewidth':lw,'color':color},
        capprops     = {'linewidth':lw,'color':color},
        boxprops     = {'linewidth':lw,'color':color,
                  'facecolor':color if filled else bgcolor},
        **kwargs);
    return bp