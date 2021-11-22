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

# Load a Matlab-like namespace
from pylab import *

# We keep a separate handle specifically to numpy. This allows us to (if
# desired) replace numpy/pylab with the Jax namespace, but retain access to some
# true numpy functions as needed. This is mainly needed to create mutable
# arrays, which we don't need to differentiate. The ability to assign to array 
# locations creates more readable initialization code, but isn't supported by
# the Jax array type. 
import numpy as np0

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

def regλ(N,K,ρ=1.3,γ=0.5):
    ''' 
    ----------------------------------------------------------------------------
    Regularized per-bin rate estimate. This simply divides the spike count by 
    the number of visits, with a small regularization parameter to prevent 
    division by zero. This estimator is not very good, and you shouldn't use it;
    It's provided as straw-man to show how much better the other estimators are.
    
    Parameters
    ----------
    N: 2D array; Number of visits to each location
    K: 2D array; Number of spikes observed at each location
    ρ: Regularization: small parameter to add to N to avoid ÷0
    γ: Bias parameter; defaults to 0.5
    
    Returns
    -------
    2D array: Regularized (biased) estimate of firing rate in each bin
    '''
    return (K+ρ*(sum(K)/sum(N)-γ)+γ)/(N+ρ)

def blurkernel(L,σ,normalize=False):
    '''
    ----------------------------------------------------------------------------
    Gaussian convolution kernel    
    
    Parameters
    ----------
    L: Size of L×L spatial domain
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    normalize: boolean; whether to make kernel sum to 1
    '''
    k = exp(-(arange(-L//2,L//2)/σ)**2)
    if normalize: 
        k /= sum(k)
    return fftshift(k)

def conv(x,K):
    '''
    ----------------------------------------------------------------------------
    Compute circular 2D convolution using FFT
    Kernel K should already be fourier-transformed
    
    Parameters
    ----------
    x: 2D array
    K: Fourier-transformed convolution kernel
    '''
    return real(ifft2(fft2(x.reshape(K.shape))*K))

def blur(x,σ,**kwargs):
    '''
    ----------------------------------------------------------------------------
    2D Gaussian blur via fft
    
    Parameters
    ----------
    x: 2D array
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    '''
    kern = fft(blurkernel(x.shape[0],σ,**kwargs))
    return conv(x,outer(kern,kern))

def kdeλ(N,K,σ,**kwargs):
    '''
    ----------------------------------------------------------------------------
    Estimate rate using Gaussian KDE smoothing. This is better than estimating
    the rate using a binned histogram, but worse than a Gaussian-Process 
    estimator. 
    
    Parameters
    ----------
    N: 2D array; Number of visits to each location
    K: 2D array; Number of spikes observed at each location
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    '''
    return regλ(blur(N,σ),blur(K,σ),**kwargs)

def zeromean(x, mask):
    '''
    ----------------------------------------------------------------------------
    Mean-center data, accounting for masked-out regions
    
    Parameters
    ----------
    x: L×L 2D array
    mask: L×L binary mask of "in bounds" regions 
    '''
    x = x.reshape(mask.shape)
    return (x-mean(x[mask]))*mask

def fft_acorr(x,mask):
    '''
    ----------------------------------------------------------------------------
    Zero-lag normalized to match signal variance
    
    Parameters
    ----------
    x: L×L 2D array
    mask: L×L binary mask of "in bounds" regions 
    '''
    x   = zeromean(x,mask)
    # Window attenuates boundary artefacts
    L = x.shape[0]
    win = hanning(L)
    win = outer(win,win)
    # Calculate autocorrelation using FFT
    psd = (abs(fft2(x*win))/L)**2
    acr = fftshift(real(ifft2(psd)))
    # Adjust peak for effects of mask, window
    return acr*var(x[mask])/acr[L//2,L//2]

def radial_average(y):
    '''
    ----------------------------------------------------------------------------
    Get radial autocorrelation by averaging 2D autocorrelogram
    '''
    i = int32(abs(coords)) # Radial distance
    a = array([mean(y[i==j]) for j in range(L//2+1)])
    return concatenate([a[::-1],a[1:-1]])

def radial_acorr(y):
    '''
    ----------------------------------------------------------------------------
    Autocorrelation as a function of distance
    '''
    return radial_average(fft_acorr(y))

def fft_upsample_1D(x,factor=4):
    '''
    ----------------------------------------------------------------------------
    Upsample 1D array using the FFT
    '''
    n  = len(x)
    n2 = n*factor
    f  = fftshift(fft(x))*hanning(n)
    f2 = np0.complex128(np.zeros(n2))
    r0 = (n2+1)//2-(n+0)//2
    f2[r0:r0+n] = f
    return np.real(ifft(fftshift(f2)))*factor

def acorr_peak(r,F=6):
    '''
    ----------------------------------------------------------------------------
    sinc upsample at ×F resolution to get distance to first peak
    '''
    r2 = fft_upsample_1D(r,F)
    return min(find_peaks(r2[len(r2)//2:])[0])/F-1,r2

def kernel_to_covariance(kern):
    '''
    ----------------------------------------------------------------------------
    Explicitly construct covariance matrix from a convolution kernel (Generally 
    we shouldn't be doing this for large problems; it's only used to verify code
    on smaller problems as a sanity check)
    
    Covariance is a doubly block-circulant matrix Use np.circulant to build 
    blocks, then copy with shift to make 2D block-circulant matrix.
    '''
    assert(argmax(kern.ravel())==0)
    L = kern.shape[0]
    b = array([circulant(r) for r in kern])
    b = b.reshape(L**2,L).T
    s = array([roll(b,i*L,1) for i in range(L)])
    return s.reshape(L**2,L**2)

def repair_small_eigenvalues(kern,mineig=1e-6):
    '''
    ----------------------------------------------------------------------------
    Kernel must be positive; fix small eigenvalues
    '''
    assert(argmax(kern.ravel())==0)
    kfft = np0.array(fft2(kern))
    keig = abs(kfft)
    υmin = mineig*np.max(keig)
    zero = keig<υmin
    kfft[zero] = υmin
    kern = real(ifft2(maximum(υmin,kfft)))
    return kern

def solveGP(kern,y,τe,tol=1e-4,reg=1e-5):
    '''
    ----------------------------------------------------------------------------
    Minimum residual solver is fast
    '''
    kern = repair_small_eigenvalues(kern,reg)
    knft = fft2(kern)
    τy   = τe*zeromean(y).ravel()
    Στy  = conv(τy,knft).ravel()
    Hv   = lambda v:conv(τe*v,knft).ravel() + v
    ΣτεI = LinearOperator((L**2,L**2),Hv,Hv,dtype=np.float64)
    μ    = minres(ΣτεI,Στy,tol=tol)[0]
    return μ.reshape(L,L) + mean(y[mask])

def mirrorpad(y,pad):
    '''
    ----------------------------------------------------------------------------
    Reflected boundary for convolution
    '''
    y = np0.array(y)
    y[:pad, :]=flipud(y[ pad: pad*2,:])
    y[:, :pad]=fliplr(y[:, pad: pad*2])
    y[-pad:,:]=flipud(y[-pad*2:-pad,:])
    y[:,-pad:]=fliplr(y[:,-pad*2:-pad])
    return y

def radial_kernel(rk):
    '''
    ----------------------------------------------------------------------------
    Make radially symmetric 2D kernel from 1D radial kernel
    '''
    r    = abs(coords)
    kern = interp1d(arange(L//2),rk[L//2:],fill_value=0,bounds_error=0)(r)
    return fftshift(kern)

def zerolag(ac,r=3):
    '''
    ----------------------------------------------------------------------------
    Estimate true zero-lag variance via quadratic interpolation.
    '''
    z = array(ac[L//2-r:L//2+r+1])
    v = float32(arange(r*2+1))
    return polyfit(v[v!=r],z[v!=r],2)@array([r**2,r,1])

def op(M,Av):
    '''
    ----------------------------------------------------------------------------
    Construct a symmetric linear operator from a function which computes the
    product of said operator with a vector. 
    
    Parameters
    ----------
    M: dimension of operator
    Av: f:R^M→R^M: linear operator acting on length M vectors
    '''
    return LinearOperator((M,)*2,Av,Av,dtype=np.float64)

def cop(k):
    '''
    ----------------------------------------------------------------------------
    Construct a convolution operator
    '''
    return op(lambda v:conv(v,k).ravel())

def newton_raphson(lλh,J,H,tol=1e-3,mtol=1e-5):
    '''
    ----------------------------------------------------------------------------
    '''
    u = lλh.ravel()
    for i in range(50):
        Δ = -minres(H(u),J(u),tol=mtol,M=M)[0]
        u += Δ
        if max(abs(Δ))<tol: return u
    print('Iteration did not converge')
    return u