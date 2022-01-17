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
Routines for loading and preparing Krupic lab datasets
"""

datadir = "../"

from scipy.io import loadmat
from pylab import *

from util       import find
from lgcp2d     import slog
from basics     import blur, fft_acorr, radial_average, acorr_peak, zerolag, radial_acorr
from estimators import kdeλ
from plot       import *

# Load a given dataset, rescaling the spatial locations
def load_dataset(fn,dataindex,pad):
    '''
    ----------------------------------------------------------------------------
    
    Parameters
    ----------
    '''
    # Get dataset
    data = loadmat(fn,squeeze_me=True)
    xy50 = data['xy50'] # Position
    sp50 = data['sp50'] # Spikes
    hd50 = data['hd50'] # Head angle
    
    # Rescale location data from all datasets using same transformation
    allxy50 = concatenate(xy50)
    x,y     = allxy50.T
    minx,maxx,miny,maxy = np.min(x),np.max(x),np.min(y),np.max(y)
    delta   = np.max([maxx-minx,maxy-miny])
    scale   = (1-1e-6)/(delta*pad*2)
    
    # Get specified dataset
    print(shape(xy50))
    x,y  = xy50[dataindex].T
    s    = sp50[dataindex]
    x    = (x-(maxx+minx)/2+delta*pad)*scale
    y    = (y-(maxy+miny)/2+delta*pad)*scale
    
    return s,x,y,scale

def bin_spikes(px,py,s,L,w=None):
    '''
    ----------------------------------------------------------------------------
    
    Parameters
    ----------
    '''
    # Bin spike counts
    #N    = histogram2d(y,x,(bins,bins),density=0,weights=w)[0]
    #ws   = s if w is None else array(s)*array(w)
    #K    = histogram2d(y,x,(bins,bins),density=0,weights=ws)[0]
    #return N,K
    
    if w is None:
        w = ones(len(px))
        
    assert np.max(px)<1 and np.max(py)<1 and np.min(px)>=0 and np.min(py)>=0
    ix,fx = divmod(px*L,1)
    iy,fy = divmod(py*L,1)
    
    assert np.max(ix)<L-1 and np.max(iy<L-1)
    
    w11 = fx*fy*w
    w10 = fx*(1-fy)*w
    w01 = (1-fx)*fy*w
    w00 = (1-fx)*(1-fy)*w
    
    qx  = concatenate([ix,ix+1,ix,ix+1])
    qy  = concatenate([iy,iy,iy+1,iy+1])
    z   = concatenate([w00,w10,w01,w11])
    
    ibins = arange(L+1)
    
    N   = histogram2d(qy,qx,(ibins,ibins),density=False,weights=z)[0]
    
    ws  = z*concatenate((s,)*4)
    K   = histogram2d(qy,qx,(ibins,ibins),density=0,weights=ws)[0]
    
    return float32(N),float32(K)

def prepare_dataset(fn,idx,L,pad,doplot=True):
    '''
    ----------------------------------------------------------------------------
    
    Parameters
    ----------
    '''
    # Get specific dataset
    idata = 1
    s,px,py,scale = load_dataset(fn,idx,pad)

    # Grab spikes
    st    = find(s>0)                # samples with a spike
    xs,ys = px[st],py[st]            # locations where spikes happened
    sk    = s[st]                    # number of spikes per spike event
    N,K   = bin_spikes(px,py,s,L)    # count number of visits and number of spikes per bin
    mask  = blur(N>0,2)>0.1          # mask out areas with no data
    y     = float32(nan_to_num(K/N)) # spikes/second

    # Calibrate grid scale
    λhat   = y # spikes/second
    acorr2 = fft_acorr(λhat,mask) # Get 2D autocorrelation
    acorrR = radial_average(acorr2) # Get radial autocorrelation
    res    = 50                     # Subsampling resolution
    P,acup = acorr_peak(acorrR,res) # Distance to first peak in bins
    
    # Precompute variables; Passed as globals to jac/hess
    fgσ  = P/pi           # In units of linear-bins
    bgσ  = fgσ*5          # In units of linear-bins
    n    = N.ravel()      # seconds/bin
    y    = y.ravel()      # spikes/second within each bin
    λhat = kdeλ(N,K,fgσ)  # KDE estimated rate
    λbg  = kdeλ(N,K,bgσ)  # Background rate
    lλh  = slog(λhat)     # Log rate
    lλb  = slog(λbg)      # Log background
    lλf  = lλh - lλb      # Foreground log rate

    # Initial guess for kernel height
    σ0   = zerolag(radial_acorr(lλf,mask))

    if doplot:
        
        figure(figsize=(6,4.5),dpi=200)
        
        subplots_adjust(left=0,right=0.7,top=1,bottom=0.1,wspace=0.1,hspace=0.2)
        ax = {}
        
        ax[1]=subplot2grid((3,3),(0,0))
        title('Visits');
        axis('off')
        imshow(N,vmin=percentile(N,5),vmax=percentile(N,98));
        
        ax[2]=subplot2grid((3,3),(0,1))
        title('Activity');
        axis('off')
        imshow(y.reshape(L,L),vmin=percentile(y,1),vmax=percentile(y,98));
        
        ax[3]=subplot2grid((3,3),(0,2))
        showim(mask,'Mask');
        subplots_adjust(top=0.8)
        
        ax[4]=subplot2grid((3,3),(1,0))
        title('KDE-smoothed rate');
        axis('off')
        imshow(λhat*mask,vmin=0,vmax=percentile(λhat,99));
        
        ax[5]=subplot2grid((3,3),(1,1))
        showim(lλf,'Foreground log-rate',mask=mask);
        
        ax[6]=subplot2grid((3,3),(1,2))
        showim(lλb,'Background log-rate',mask=mask);
        suptitle('%s dataset %d'%(fn,idx))
        
        ax[7]=subplot2grid((3,3),(2,0))
        imshow(acorr2)
        axis('off')
        
        ax[8]=subplot2grid((3,3),(2,1),colspan=2)
        plot((arange(L)-L/2),acorrR)
        plot((linspace(-L/2,L/2,L*res)),acup)
        xlim(-L/2,L/2)
        xlabel('Δbins')
        ylabel('spikes²/sample²')
        axvline(P,color='k',lw=0.5)
        simpleaxis()
        
        tight_layout()
        nudge_axis_left(240)
        nudge_axis_right(35)
        
        
        sca(ax[1])
        colorbar(label='samples/bin')
        sca(ax[2])
        colorbar(label='spikes/sample')
        sca(ax[3])
        # spacing hack
        colorbar(label='spikes/sample').remove()
        sca(ax[4])
        colorbar(label='spikes/sample')
        sca(ax[5])
        colorbar(label='log-rate')
        sca(ax[6])
        colorbar(label='log-rate')
        sca(ax[7])
        colorbar(label='variance\n(spikes/samples)²')
        
        figurebox(color='w')
    
    return scale,mask,n,y,P,bgσ,lλb,fgσ,lλf,σ0