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
`util.py`: Miscellaneous utility functions used in the notebooks
"""

import os
import errno

# For progress bar and code timing
import time

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
from numpy import *

# For the statistical summary function
from scipy.stats import pearsonr

# used by the cinv function
import scipy.linalg.lapack

# Used to locate peaks in the autocorrelogram
from scipy.signal import find_peaks

# Used to explicitly construct covariance matrix from a convolution kernel
# (Generally we shouldn't be doing this for large problems; it's only used to
# verify code on smaller problems as a sanity check)
from scipy.linalg import circulant 

# Used to make radially symmetric 2D kernel from 1D radial kernel
from scipy.interpolate import interp1d

# Allows us to work with linear operators without constructing them as matrices
# This is very useful for large covariance priors, which may not fit in memory,
# but can be calculated easily as convolution kernels using the FFT. 
from scipy.sparse.linalg import minres,LinearOperator

# Varies depending on scipy version
try:
    from scipy.fft import *
except:
    from scipy.fftpack import *

import time as systime
def current_milli_time():
    '''
    Returns the time in milliseconds
    '''
    return int(round(systime.time() * 1000))
    
__GLOBAL_TIC_TIME__ = None
def tic(doprint=True,prefix=''):
    ''' 
    Similar to Matlab tic 
    
    Parameters
    ----------
    doprint: bool
        if True, print elapsed time. Else, return it.
    
    Returns
    -------
    t: int
        `current_milli_time()`
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            if doprint:
                print(prefix,'t=%dms'%(t-__GLOBAL_TIC_TIME__))
        elif doprint:
            print("timing...")
    except: 
        if doprint: print("timing...")
    __GLOBAL_TIC_TIME__ = current_milli_time()
    return t

def toc(doprint=True,prefix=''):
    ''' 
    Similar to Matlab toc 
    
    Parameters
    ----------
    doprint: bool
        if True, print elapsed time. Else, return it.
    prefix: str
    
    Returns
    -------
    t: number
        Current timestamp
    dt: number
        Time since the last call to the tic() or toc() function.
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            dt = t-__GLOBAL_TIC_TIME__
            if doprint: print(prefix,'dt=%dms'%(dt))
            return t,dt
        elif doprint:
            print("havn't called tic yet?")
    except: 
        if doprint: print("havn't called tic yet?")
    return t,None

def progress_bar(x,N=None):
    '''
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
            print('\r['+('█'*k)+q+(' '*(50-k-1))+
                ']%3d%%'%(i*100//N)+(pattern%i),
                end='',
                flush=True)
            wait_til_ms = time_ms+250
        yield x
    print('\r'+' '*70+'\r',end='',flush=True)

def zgrid(L):
    '''
    2D grid coordinates as complex numbers
    '''
    c = arange(L)-L//2
    return 1j*c[:,None]+c[None,:]

def printstats(a,b,message='',mask=None):
    '''
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

def find(a):
    '''
    The old "find" syntax is cleaner!
    
    Parameters
    --------------------------------------------------------
    '''
    return np.where(np.array(a).ravel())[0]

def speak(text):
    '''
    For jupyter notebooks: trigger browser to notify when done
    
    Parameters
    --------------------------------------------------------
    text: str
        Text to peak
    '''
    from IPython.display import Javascript as js, clear_output
    # Escape single quotes
    text = text.replace("'", r"\'")
    display(js('''
    if(window.speechSynthesis) {{
        var synth = window.speechSynthesis;
        synth.speak(new window.SpeechSynthesisUtterance('{text}'));
    }}
    '''.format(text=text)))

def notify(what='attention'):
    #os.system("echo -n '\a'")
    speak(what)

def get_edges(signal,pad_edges=True):
    '''
    Assuming a binary signal, get the start and stop times 
    of each treatch of "1s"
    
    Parameters
    --------------------------------------------------------
    signal : 1-dimensional array-like
    
    Other Parameters
    --------------------------------------------------------
    pad_edges : True
        Should we treat blocks that start or stop at the 
        beginning or end of the signal as valid?
    
    Returns
    --------------------------------------------------------
    2xN array of bin start and stop indecies
    '''
    if len(signal)<1:
        return np.array([[],[]])
    if tuple(sorted(np.unique(signal)))==(-2,-1):
        raise ValueError('signal should be bool or int∈{0,1};'+
            ' (using ~ on an int array?)')
    signal = np.int32(np.bool8(signal))
    starts = list(np.where(np.diff(np.int32(signal))==1)[0]+1)
    stops  = list(np.where(np.diff(np.int32(signal))==-1)[0]+1)
    if pad_edges:
        # Add artificial start/stop time to incomplete blocks
        if signal[0 ]: starts = [0]   + starts
        if signal[-1]: stops  = stops + [len(signal)]
    else:
        # Remove incomplete blocks
        if signal[0 ]: stops  = stops[1:]
        if signal[-1]: starts = starts[:-1]
    return np.array([np.array(starts), np.array(stops)])

def interpolate_NaN(u):
    '''
    Fill in NaN (missing) data in a one-dimensional 
    timeseries via linear interpolation.
    '''
    u = np.array(u)
    for s,e in zip(*get_edges(~np.isfinite(u))):
        if s==0: 
            u[:e+1] = u[e+1]
        elif e==len(u): 
            u[s:] = u[s-1]
        else:
            a = u[s-1]
            b = u[e]
            u[s:e+1] = a + (b-a)*np.linspace(0,1,e-s+1)
        #assert all(isfinite(u[s:e]))
    #assert all(isfinite(u))
    u[~np.isfinite(u)] = np.mean(u[np.isfinite(u)])
    return u

def onehot(x,N=None):
    '''
    Parameters
    --------------------------------------------------------
    x: int23
        List of non-negative indecies that are 1
    N: int32
        Length of desired vector
    
    '''
    x = array(x)
    scalar = False
    try:
        T = len(x)
    except:
        x = array([x])
        T = len(x)
        scalar=True
    if N is None:
        N = int(np.max(x))+1
    result = zeros((T,N))
    result[arange(T),x] = 1
    if scalar:
        result = result.ravel()
    return result
        
        
        
def slog(x,minrate = 1e-10, rtype=float32):
    '''
    Safe log function; Avoids numeric overflow by clipping
    '''
    return log(maximum(minrate,x), dtype=rtype)


def is_in_hull(P,hull):
    '''
    Determine if the list of points P lies inside the hull
    credit: https://stackoverflow.com/a/52405173/900749
    
    Parameters
    --------------------------------------------------------
    P: points
    hull: Convex Hull object
    
    Returns
    --------------------------------------------------------
    isInHull: Length NPOINTS 1D np.bool
        Array of booleans indicating which points are within
        the convex hull.
    '''
    P = c2p(p2c(P)).T # lazy code reuse: ensure array shape
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    isInHull = np.all(
        (A @ np.transpose(P))<=np.tile(-b,(1,len(P))),
        axis=0)
    return isInHull
    

"""
------------------------------------------------------------
Routines for Gaussian blurs, FFT-based convolutions and 
autocorrelation, constructing radially-symmetric kernels, 
and common types of linear operators.
"""

def blurkernel(L,σ,normalize=False):
    '''
    1D Gaussian blur convolution kernel    
    
    Parameters
    --------------------------------------------------------
    L: int
        Size of L×L spatial domain
    σ: positive float
        kernel radius exp(-x²/σ) (standard deviation in x 
        and y ×⎷2)
    normalize: boolean
        Whether to make kernel sum to 1
    '''
    k = exp(-(arange(-L//2,L//2)/σ)**2)
    if normalize: 
        k /= sum(k)
    return fftshift(k)

def blurkernel2D(L,σ,normalize=False):
    '''
    2D Gaussian blur convolution kernel    
    
    Parameters
    --------------------------------------------------------
    L: Size of L×L spatial domain
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    normalize: boolean; whether to make kernel sum to 1
    '''
    k = blurkernel(L,σ)
    k = outer(k,k)
    if normalize: 
        k /= sum(k)
    return k

def conv(x,K):
    '''
    Compute circular 2D convolution using FFT
    Kernel K should already be fourier-transformed
    
    Parameters
    --------------------------------------------------------
    x: 2D array
    K: Fourier-transformed convolution kernel
    '''
    return ifft2(fft2(x.reshape(K.shape))*K).real

def blur(x,σ,**kwargs):
    '''
    2D Gaussian blur via fft
    
    Parameters
    --------------------------------------------------------
    x: 2D np.array
    σ: float
        kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    '''
    kern = fft(blurkernel(x.shape[0],σ,**kwargs))
    return conv(x,outer(kern,kern))

def zeromean(x, mask):
    '''
    Mean-center data, accounting for masked-out regions
    
    Parameters
    --------------------------------------------------------
    x: L×L 2D np.array
    mask: L×L np.bool
        Binary mask of "in bounds" regions 
    '''
    x = x.reshape(mask.shape)
    return (x-mean(x[mask]))*mask

def fft_acorr(x,mask=None,window=True):
    '''
    Zero-lag normalized to match signal variance
    
    Parameters
    --------------------------------------------------------
    x: L×L 2D np.array
    mask: L×L np.bool
        Binary mask of "in bounds" regions 
    '''
    if not mask is None:
        mask = float32(mask)>0
        x    = zeromean(x,mask)
    else:
        x    = x - mean(x)
    L    = x.shape[0]
    # Calculate autocorrelation using FFT
    if window:
        # Window attenuates boundary artefacts
        # 
        win = hanning(L)
        win = outer(win,win)
        psd = (abs(fft2(x*win))/L)**2
    else:
        psd = (abs(fft2(x))/L)**2
    acr  = fftshift(ifft2(psd).real)
    if not mask is None:
        # Adjust peak for effects of mask, window
        acr  = acr*var(x[mask])/np.max(acr)
    return acr

def radial_average(y):
    '''
    Get radial autocorrelation by averaging 2D 
    autocorrelogram
    
    Parameters
    --------------------------------------------------------
    y: np.float32
        2D LxL array of firing-rate spatial autocorrelogram.
    '''
    y = float32(y)
    if not len(y.shape)==2 and y.shape[0]==y.shape[1]:
        raise ValueError('y should be a square np.array')
    L = y.shape[0]
    coords = zgrid(L)
    i = int32(abs(coords)) # Radial distance
    a = array([mean(y[i==j]) for j in range(L//2+1)])
    return concatenate([a[::-1],a[1:-1]])

def radial_acorr(y,mask):
    '''
    Autocorrelation as a function of distance
    '''
    return radial_average(fft_acorr(y,mask))

def fft_upsample_1D(x,factor=4):
    '''
    Upsample 1D array using the FFT
    '''
    n  = len(x)
    n2 = n*factor
    f  = fftshift(fft(x))*hanning(n)
    f2 = np.complex128(np.zeros(n2))
    r0 = (n2+1)//2-(n+0)//2
    f2[r0:r0+n] = f
    return np.real(ifft(fftshift(f2)))*factor

def acorr_peak(r,F=6):
    '''
    sinc upsample at ×F resolution to get distance to first
    peak
    '''
    r2 = fft_upsample_1D(r,F)
    peaks = find_peaks(r2[len(r2)//2:])[0]
    if len(peaks):
        return min(peaks)/F,r2
    return NaN,r2
    
def acorr_trough(r,F=6):
    '''
    sinc upsample at ×F resolution to get distance to first 
    peak
    '''
    r2 = fft_upsample_1D(r,F)
    troughs = find_peaks(-r2[len(r2)//2:])[0]
    if len(troughs):
        return min(troughs)/F*2,r2
    return NaN,r2

def kernel_to_covariance(kern):
    '''
    Explicitly construct covariance matrix from a 
    convolution kernel (Generally we shouldn't be doing this
    for large problems; it's only used to verify code
    on smaller problems as a sanity check)
    
    Covariance is a doubly block-circulant matrix Use 
    np.circulant to build locks, then copy with shift to 
    make 2D block-circulant matrix.
    '''
    assert(argmax(kern.ravel())==0)
    L = kern.shape[0]
    b = array([circulant(r) for r in kern])
    b = b.reshape(L**2,L).T
    s = array([roll(b,i*L,1) for i in range(L)])
    return s.reshape(L**2,L**2)

def repair_small_eigenvalues(kern,mineig=1e-6):
    '''
    Kernel must be positive; fix small eigenvalues
    '''
    assert(argmax(kern.ravel())==0)
    kfft = np.array(fft2(kern))
    keig = abs(kfft)
    υmin = mineig*np.max(keig)
    zero = keig<υmin
    kfft[zero] = υmin
    kern = real(ifft2(maximum(υmin,kfft)))
    return kern

def solveGP(kern,y,τe,mask,tol=1e-4,reg=1e-5):
    '''
    Minimum residual solver is fast
    '''
    L = kern.shape[0]
    kern = repair_small_eigenvalues(kern,reg)
    knft = fft2(kern)
    τy   = τe*zeromean(y,mask).ravel()
    Στy  = conv(τy,knft).ravel()
    Hv   = lambda v:conv(τe*v,knft).ravel() + v
    ΣτεI = LinearOperator((L**2,L**2),Hv,Hv,dtype=np.float64)
    μ    = minres(ΣτεI,Στy,tol=tol)[0]
    return μ.reshape(L,L) + mean(y.ravel()[mask.ravel()])

def mirrorpad(y,pad):
    '''
    Reflect boundary for convolution.
    '''
    y = np.array(y)
    y[:pad, :]=flipud(y[ pad: pad*2,:])
    y[:, :pad]=fliplr(y[:, pad: pad*2])
    y[-pad:,:]=flipud(y[-pad*2:-pad,:])
    y[:,-pad:]=fliplr(y[:,-pad*2:-pad])
    return y

def radial_kernel(rk):
    '''
    Make radially symmetric 2D kernel from 1D radial kernel
    
    Parameters
    --------------------------------------------------------
    as: 1D np.array
        1D kernel that will be used to generate radially
        symmetric 2D kernel.
    '''
    L = rk.shape[0]
    coords = zgrid(L)
    r    = abs(coords)
    kern = interp1d(arange(L//2),rk[L//2:],fill_value=0,bounds_error=0)(r)
    return fftshift(kern)

def zerolag(ac,r=3):
    '''
    Estimate true zero-lag variance via quadratic 
    interpolation.
    
    Parameters
    --------------------------------------------------------
    as: 1D np.array
        Autocorrelation
    r: int
        Samples in the vicinity of the zero-lag correlation
        to mask out and replace with a quadratic 
        interpolation.
    '''
    L = ac.shape[0]
    z = array(ac[L//2-r:L//2+r+1])
    v = float32(arange(r*2+1))
    return polyfit(v[v!=r],z[v!=r],2)@array([r**2,r,1])

def op(M,Av):
    '''
    Construct a symmetric linear operator from a function 
    which computes the product of said operator with a 
    vector. 
    
    Parameters
    --------------------------------------------------------
    M: int
        Dimension of operator
    Av: f:R^M→R^M 
        linear operator acting on length M vectors
    '''
    return LinearOperator((M,)*2,Av,Av,dtype=np.float64)

def cop(k):
    '''
    Construct a convolution operator
    
    Parameters
    --------------------------------------------------------
    k: array; convolution kernel Fourier transform. 
    '''
    M = prod(k.shape)
    return op(M, lambda v:conv(v,k).ravel())

def diagop(d):
    '''
    Construct a diagonal operator
    
    Parameters
    --------------------------------------------------------
    d: vector; diagonal of matrix operator
    '''
    d = d.ravel()
    M = d.shape[0]
    return op(M, lambda v:v.ravel()*d)

"""
------------------------------------------------------------
Routines for working with the Hartley transform. 
"""

def dx_op(L):
    '''
    2D finite difference in the 1st coordinate
    {-.5,0,.5}
    
    Parameters
    --------------------------------------------------------
    L: positive integer
        The size of the 2D L×L grid
        
    Returns
    --------------------------------------------------------
    dx: shape (L,L) float32 array
        Discrete derivative in the x (column) direction.
        Use with e.g. FFT convolve.
        Transpose to get dy.
    '''
    dx = zeros((L,L),dtype=float32)
    dx[0, 1]= .5
    dx[0,-1]=-.5
    return dx

def hessian_2D(q):
    '''
    Get Hessian at all points
    
    Parameters
    --------------------------------------------------------
    q: shape (L,L) ndarray
        2D spatial function for which to compute Hessian
    
    Returns
    --------------------------------------------------------
    hessian: shape (L,L,2,2)
        2×2 Hessian in (x,y) for all points in L×L grid q
    '''
    dx  = dx_op(q.shape[0])
    fx  = fft2(dx)
    fy  = fft2(dx.T)
    dxx = conv(q,fx*fx)
    dxy = conv(q,fy*fx)
    dyy = conv(q,fy*fy)
    return array([[dxx,dxy],[dxy,dyy]]).transpose(2,3,0,1)

def h2f_2d_truncated(u,L,use2d):
    '''
    Convert from 2D truncated low-rank Hartley 
    representation to full-rank Fourier representation.
    
    The Hartley transform is Re[F(x)] + Im[F(x)] where F is
    the unitary Fourier transform. In the 2D FT of real-
    valued signals, nonzero frequency components have
    rotational symmetry. The real components equal
    themselves after a 180 degree rotation of the
    coefficient matrix. The imaginary components equal
    their negative after a 180 degree rotation. From this
    we can recover Fourier coefficients from the Hartley
    transform. The components with zero frequency behave as
    the 1D fourier transform.
    
    Or just
    
    Parameters
    --------------------------------------------------------
    u: shape (R,...) float32 ndarray
        Array where the first dimension is the vector of 
        low-rank Hartley transform coefficients; Operation
        is broadcast over remaining dimensions
    L: positive int
        The size of the original 2D L×L spatial grid
    use2d: shape (L,L) boolean ndarray
        Indicator mask of which of 2D Fourier/Hartley 
        coefficients were retained in the low-dimensional 
        representation. Get this from model.use2d
    
    Returns
    --------------------------------------------------------
    fx.T: shape (L,L,...) complex64 ndarray
        Low-rank Fourier-space representation of u.
    '''
        
    f1 = zeros((L,L)+u.shape[1:],dtype=float32)
    f1[use2d,...] = u
    f2 = empty(f1.shape,dtype=float32)
    f2[0 ,0 ,...] = f1[0 ,0 ,...]
    f2[1:,1:,...] = f1[1:,1:,...][::-1,::-1,...]
    f2[0 ,1:,...] = f1[0 ,1:,...][::-1,...]
    f2[1:,0 ,...] = f1[1:,0 ,...][::-1,...]
    u2 = f2[use2d,...]
    return ((u+u2) + 1j*(u-u2))*0.5
    #fx = ((f1+f2) + 1j*(f1-f2))*.5
    #return fx[use2d,...]

def f2h_2d_truncated(x,use2d):
    '''
    Convert from full 2D spatial fourier space to truncated
    low-D Hartley space representation. 
    
    Parameters
    --------------------------------------------------------
    x: shape (L,L,...) complex64 ndarray
        Low-D Fourier coefficients
    use2d: shape (L,L) boolean ndarray
        Indicator mask of which of 2D Fourier/Hartley 
        coefficients were retained in the low-dimensional 
        representation. Get this from model.use2d
    
    Returns
    --------------------------------------------------------
    xh: shape (R,...) float32 ndarray
        Low-D (retaining R components) Hartley transform.
    '''
    #x = x[use2d,...]
    return float32(real(x) + imag(x))

def h_conv2d_truncated(hk,hx,L,use2d):
    '''
    Apply a 2D FFT convolution on data (and kernel) packed 
    as a low-rank (R) Hartley transform (float32).
    
    Parameters
    --------------------------------------------------------
    hk: shape (R,) float32 ndarray
        Truncated (R<L²) 2D Hartley transform of the kernel. 
    hx: shape (R,...) float32 ndarray
        Truncated (R<L²) 2D Hartley transform of the data.
        Operation is broadcast over trailing dimensions.
    L: positive int
        The size of the original 2D L×L spatial grid
    use2d: shape (L,L) boolean ndarray
        Indicator mask of which of 2D Fourier/Hartley 
        coefficients were retained in the low-dimensional 
        representation. Get this from model.use2d
        
    Returns
    --------------------------------------------------------
    shape (R,...) float32 ndarray
        Convolved result low-rank Hartley representation.
    '''
    fk = h2f_2d_truncated(hk,L,use2d)
    fx = h2f_2d_truncated(hx,L,use2d)
    result = f2h_2d_truncated((fk.T*fx.T).T,use2d)
    return result

def pdist(a,b):
    '''
    Pairwise distances between two lists of scalars
    '''
    return abs(a[:,None]-b[None,:])


def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not,
    attempt to create it.
    
    Parameters
    --------------------------------------------------------
    dirname : str
    """
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass


def make_rebroadcast_slice(x,axis=0,verbose=False):
    '''
    '''
    x = np.array(x)
    naxes = len(np.shape(x))
    if verbose:
        print('x.shape=',np.shape(x))
        print('naxes=',naxes)
    if axis<0:
        axis=naxes+axis
    if axis==0:
        theslice = (None,Ellipsis)
    elif axis==naxes-1:
        theslice = (Ellipsis,None)
    else:
        a = axis
        b = naxes - a - 1
        theslice = (np.s_[:],)*a + (None,) + (np.s_[:],)*b
    if verbose:
        print('axis=',axis)
    return theslice 


def zscore(x,axis=0,regularization=1e-30,verbose=False,ignore_nan=True):
    '''
    Z-scores data, defaults to the first axis.
    A regularization factor is added to the standard 
    deviation to preven numerical instability when the 
    standard deviation is extremely small. The default 
    regularization is 1e-30.
    
    Parameters
    --------------------------------------------------------
    x:
        Array-like real-valued signal.
    axis: 
        Axis to zscore; default is 0.

    Returns
    --------------------------------------------------------
    x: np.ndarray
        (x-mean(x))/std(x)
    '''
    x = zeromean(x,axis=axis,ignore_nan=ignore_nan)
    if np.prod(x.shape)==0:
        return x
    theslice = make_rebroadcast_slice(x,axis=axis,verbose=verbose)
    ss = (np.nanstd if ignore_nan else np.std)(x,axis=axis)+regularization


def unitscale(signal,axis=None):
    '''
    Rescales `signal` so that its minimum is 0 and its maximum is 1.

    Parameters
    --------------------------------------------------------
    signal (np.array): real-valued signal
    
    Returns
    --------------------------------------------------------
    signal: np.array
        Rescaled signal-min(signal)/(max(signal)-min(signal))
    '''
    signal = np.float64(np.array(signal))
    if axis==None:
        # Old behavior
        signal-= np.nanmin(signal)
        signal/= np.nanmax(signal)
        return signal
    # New behavior
    theslice = make_rebroadcast_slice(signal, axis)
    signal-= np.nanmin(signal,axis=axis)[theslice]
    signal/= np.nanmax(signal,axis=axis)[theslice]
    return signal

"""
------------------------------------------------------------
Routines for working with 2D points
"""

def p2c(p):
    '''
    Convert a point in terms of a length-2 iterable into 
    a complex number
    '''
    p = np.array(p)
    if np.any(np.iscomplex(p)): return p
    if not np.any(int32(p.shape)==2):
        raise ValueError('Shape %s not (x,y)'%(p.shape,))
    which = np.where(int32(p.shape)==2)[0][0]
    p = p.transpose(which,
        *sorted(list({*arange(len(p.shape))}-{which})))
    return p[0]+1j*p[1]

def c2p(z):
    ''' 
    Convert complex point to tuple
    '''
    z = np.array(z)
    return np.array([z.real,z.imag])

def to_xypoint(z):
    '''
    Convert possible complex (x,y) point intoformation
    into float32 (x,y) points.
    
    Parameters
    --------------------------------------------------------
    z: np.complex64
        Array of (x,y) points encoded as x+iy complex64
        
    Returns
    --------------------------------------------------------
    np.float32
        (x,y) poiny array with shape 2 × z.shape
    '''
    z = array(z)
    if np.any(iscomplex(z)):
        return complex64([z.real,z.imag])
    # Possibly already a point? 
    z = float32(z)
    if len(z.shape)<=0:
        raise ValueError(
            'This looks like a scalar, not a point')
    if z.shape[0]==1:
        return z
    if np.sum(int32(z.shape)==2)!=1:
        raise ValueError(
            ('Expected exactly one length-2 axis for (x,y)'+
             'points, got shape %s')%(z.shape,))
    which = np.where(int32(z.shape)==2)[0][0]
    other = {*arange(len(z.shape))}-{which}
    return z.transpose(which,*sorted(list(other)))

def closest(point,otherpoints,radius=inf):
    '''
    Find nearest (x,y) point witin a collection of 
    other points, with maximum distance `radius`
    
    Parameters
    --------------------------------------------------------
    point: np.float32 with shape 2
        (x,y) point to match
    otherpoints: np.float32 with shape 2×NPOINTS
        List of (x,y) points to compare
    radius: float
        Maximum allowed distance
        
    Returns
    --------------------------------------------------------
    imatch: int
        index into otherpoints of the match, or None
        if there is no match within radius
    xymatch: np.float32 with shape 2
        (x,y) coordinates of closestmatch
    distance: float
        distance to match, or NaN if no match
    '''
    radius = float(radius)
    if radius<=0:
        raise ValueError(
            'Error, radius should be positive')
    point       = to_xypoint(point)
    otherpoints = to_xypoint(otherpoints)
    if not point.shape==(2,):
        raise ValueError(
            'Expected (x,y) point as 1st argument')
    otherpoints = otherpoints.reshape(
        2,np.prod(otherpoints.shape[1:]))
    
    distances = norm(point[:,None] - otherpoints,2,0)
    nearest   = argmin(distances)
    distance  = distances[nearest]
    if distance<=radius:
        return nearest, otherpoints[:,nearest], distance
    return None,full(2,NaN,'f'),NaN

def paired_distances(z1,z2):
    '''
    Calculate pairwise distances between two sets of 
    (x,y) points encoded as x+iy complex numbers
    
    Parameters
    --------------------------------------------------------
    z1: 1D np.complex64 
        List of x+iy points
    z2: 1D np.complex64 
    
    Returns
    --------------------------------------------------------
    distance: np.float32 with shape z1.shape+z2.shape
        Array of paired distances
    '''
    z1,z2 = p2c(z1),p2c(z2)
    s1,s2 = z1.shape,z2.shape
    z1,z2 = z1.ravel(),z2.ravel()
    distance = abs(z1[:,None]-z2[None,:])
    return distance.reshape(*(s1+s2))

def pair_neighbors(z1,z2,radius=inf):
    '''
    
    Parameters
    --------------------------------------------------------
    z1: 1D np.complex64 
        List of x+iy points
    z2: 1D np.complex64 
        List of x+iy points
    radius: float, default `inf`
        Maximum connection distance
    
    Returns
    --------------------------------------------------------
    edges: NPOINTS × 2 np.int32
        Indecies (i,j) into point lists (z1,z2) of pairs
    points: NPOINTS × 2 np.complex64
        x+iy points from (z1,x2) pairs
    delta: NPOINTS np.float32:
        List of distances for each pair
    '''
    radius = float(radius)
    if radius<=0:
        raise ValueError('Radius should be positive')
        
    z1,z2 = p2c(z1),p2c(z2)
    n1,n2 = len(z1),len(z2)
    distance = abs(z1[:,None]-z2[None,:])
    unused1,unused2 = {*arange(n1)},{*arange(n2)}
    paired = set()
    while len(unused1) and len(unused2):
        # useID → pointID
        ix1 = int32(sorted(list(unused1)))
        ix2 = int32(sorted(list(unused2)))
        # useID → z
        zz1 = z1[ix1]
        zz2 = z2[ix2]
        # useID → nearest useID
        D = distance[ix1,:][:,ix2]
        neighbors1to2 = argmin(D,1)
        neighbors2to1 = argmin(D,0)
        # 
        ok1   = arange(len(ix1)) == neighbors2to1[neighbors1to2]
        ok2   = arange(len(ix2)) == neighbors1to2[neighbors2to1]
        e1to2 = {*zip(ix1[ok1],ix2[neighbors1to2[ok1]])}
        e2to1 = {*zip(ix1[neighbors2to1[ok2]],ix2[ok2])}
        assert len(e1to2-e2to1)==0
        used1,used2 = map(set,zip(*e1to2))
        unused1 -= used1
        unused2 -= used2
        paired |= e1to2

    a,b   = int32([*zip(*paired)])
    pairs = np.array([z1[a],z2[b]])
    delta = abs(pairs[0]-pairs[1])
    keep  = delta<radius
    edges = int32([a,b])
    return edges.T[keep], pairs.T[keep], delta[keep]
