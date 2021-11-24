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
"""

# For progress bar and code timing
import time

# Load a Matlab-like namespace
from pylab import *

# For the statistical summary function
from scipy.stats import pearsonr

ttic = None
def tic(msg=''):
    '''
    ----------------------------------------------------------------------------
    Timer routine to track performance
    '''
    global ttic
    t = time.time()*1000
    if ttic and msg: 
        print(('Δt = %d ms'%(t-ttic)).ljust(14)\
              +'elapsed for '+msg)
    ttic = t

def progress_bar(x,N=None):
    '''
    ----------------------------------------------------------------------------
    Progress bar wrapper for loops
    '''
    if N is None:
        x = list(x)
        N = len(x)
    K = int(np.floor(np.log10(N)))+1
    pattern = ' %%%dd/%d'%(K,N)
    wait_til_ms = time.time()*1000
    for i,x in enumerate(x):
        time_ms = time.time()*1000
        if time_ms>=wait_til_ms:
            r = i*50/N
            k = int(r)
            q = ' ▏▎▍▌▋▊▉'[int((r-k)*8)]
            print('\r['+('█'*k)+q+(' '*(50-k-1))+']%3d%%'%(i*100//N)+(pattern%i),end='',flush=True)
            wait_til_ms = time_ms+250
        yield x
    print('\r'+' '*70+'\r',end='',flush=True)
    
def pscale(x,q1=0.5,q2=99.5,domask=True):
    '''
    ----------------------------------------------------------------------------
    Plot helper: Scale data by percentiles
    '''
    u  = x[mask] if domask else x
    u = float32(u)
    p1 = percentile(u,q1)
    p2 = percentile(u,q2)
    x  = clip((x-p1)/(p2-p1),0,1)
    return x*mask if domask else x
    
def showim(x,t='',**kwargs):
    '''
    ----------------------------------------------------------------------------
    Plot helper: Show image with title, no axes
    '''
    if len(x.shape)==1: x=x.reshape(L,L)
    imshow(pscale(x,**kwargs));
    axis('off');
    title(t);
    
def slog(x,minrate = 1e-10):
    '''
    ----------------------------------------------------------------------------
    Safe log function; Avoids numeric overflow by clipping
    '''
    return log(maximum(minrate,x))

def sexp(x,bound = 10):
    '''
    ----------------------------------------------------------------------------
    Safe exponential function; Avoids under/overflow by clipping
    '''
    return exp(np.clip(x,-bound,bound))

def zgrid(L):
    '''
    ----------------------------------------------------------------------------
    2D grid coordinates as complex numbers
    '''
    c = arange(L)-L//2
    return 1j*c[:,None]+c[None,:]

def pscale(x,q1=0.5,q2=99.5,mask=True):
    '''
    ----------------------------------------------------------------------------
    Plot helper: Scale data by percentiles
    '''
    u  = x[mask] if not mask is None else x
    u  = float32(u)
    p1 = percentile(u,q1)
    p2 = percentile(u,q2)
    x  = clip((x-p1)/(p2-p1),0,1)
    return x*mask if not mask is None else x
    
def showim(x,t='',**kwargs):
    '''
    ----------------------------------------------------------------------------
    Plot helper: Show image with title, no axes
    '''
    if len(x.shape)==1: 
        L = int(round(sqrt(x.shape[0])))
        x=x.reshape(L,L)
    imshow(pscale(x,**kwargs));
    axis('off');
    title(t);

def showkn(k,t=''):
    '''
    ----------------------------------------------------------------------------
    Plot helper; Shift convolution kernel to plot
    '''
    imshow(fftshift(k)); axis('off'); title(t);

def printstats(a,b,message='',mask=None):
    '''
    ----------------------------------------------------------------------------
    Print RMSE and correlation between two rate maps
    '''
    a,b = a.ravel(),b.ravel()
    if not mask is None:
        a = a[mask.ravel()]
        b = b[mask.ravel()]
    NMSE = mean((a-b)**2)/mean(a**2)#sqrt(mean(a**2)*mean(b**2))
    print(message+':')
    print('∙ Normalized MSE: %0.1f%%'%(100*NMSE))
    print('∙ Pearson correlation: %0.2f'%pearsonr(a,b)[0])