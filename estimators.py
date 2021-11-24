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
Basic rate-map estimators
"""

# Load matlab-like namespace and helper routines
from basics import *

def estimate_period_via_autocorrelation(N,K,mask,res=5):
    '''
    ----------------------------------------------------------------------------
    Estimate grid period using autocorrelation
    
    Parameters
    ----------
    N: L×L 2D array; Number of visits to each location
    K: L×L 2D array; Number of spikes observed at each location
    mask: L×L 2D array; boolean mask for "in bounds" regions to analyze
    res: small positive integer; Upsampling resolution; default is 5s
    
    Returns
    -------
    P: Number; Estimated grid periodicity in pixels
    Δs: 1D array; Bin separations for upsampled radial autocorrelation
    acup: 1D array; Upsampled radial autocorrelation
    acorr2: L×L 2D array; 2D autocorrelogram
    '''
    L = N.shape[0]
    if not N.shape[1]==L and K.shape==(L,L):
        raise ValueError('N and K should be L×L square arrays')
    λhat   = kdeλ(N,K,L/75)         # Small blur for initial esitmate
    acorr2 = fft_acorr(λhat,mask)   # Get 2D autocorrelation
    acorrR = radial_average(acorr2) # Get radial autocorrelation
    P,acup = acorr_peak(acorrR,res) # Distance to first peak in bins 
    Δs     = linspace(-L/2,L/2,L*res)-.5/res # Subsampled grid spacing
    return P, Δs, acup, acorr2

def biased_rate(N,K,ρ=1.3,γ=0.5):
    ''' 
    ----------------------------------------------------------------------------
    Regularized per-bin rate estimate. This divides the spike count by the 
    number of visits, with a small regularization parameter to prevent division
    by zero. 
    
    This estimator is not very good, and you shouldn't use it; It's provided as
    a straw-man to show how much better the other estimators are.
    
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
    
    Returns
    -------
    2D array: KDE rate estimate of firing rate in each bin
    '''
    return biased_rate(blur(N,σ),blur(K,σ),**kwargs)

def linearGP(N,K,σ,mask,tol=1e-4,reg=1e-5):
    '''
    ----------------------------------------------------------------------------
    Linear Gaussian process rate map. This is a linear (not log-linear) model.
    
    Error are approximated. The average firing rate of the cell is calculated, 
    and the per-timepoint measurement error is assumed to equal this. Multiple
    visits to the same location improve reduce measurement error proportionaly.
    
    It is not recommended for inferring firing-rate maps, but it is a fast 
    example of GP inference on a 2D arena which may be instructive. 
    
    This uses the Minimum residual solver, which is fast, and only requires 
    a function which can compute Hessian-vector products. This can be done 
    quickly with circulant (convolutional) covariance priors using the FFT. 
    
    Parameters
    ----------
    N: 2D array; Number of visits to each location
    K: 2D array; Number of spikes observed at each location
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    
    Returns
    -------
    rate: 2D array; GP rate estimate of firing rate in each bin
    kern: 2D array; Prior covariance kernel used for inference
    y:    2D array; Binned rates (K/N) used as observations
    '''
    L = N.shape[0]
    if not N.shape[1]==L and K.shape==(L,L):
        raise ValueError('N and K should be L×L square arrays')
    # Prepare error model for GP
    ε0 = mean(K)/mean(N) # variance per measurement
    τe = N.ravel()/ε0    # precision per bin
    # Build 2D kernel for the prior
    # Scale kernel height to match data variance (heuristic)
    k1   = blurkernel(L,σ*2)
    y    = nan_to_num(K/N)
    kern = outer(k1,k1)*var(y[mask])
    kern = repair_small_eigenvalues(kern,reg)
    knft = fft2(kern)
    τy   = τe*zeromean(y,mask).ravel()
    Στy  = conv(τy,knft).ravel()
    ΣτεI = op(L**2,lambda v:conv(τe*v,knft).ravel() + v)
    μ    = minres(ΣτεI,Στy,tol=tol)[0]
    return μ.reshape(L,L) + mean(y[mask]), kern, y

def convolutionalLinearGP(N,K,σ,mask,pad=None,tol=1e-4,reg=1e-5):
    '''
    ----------------------------------------------------------------------------
    Special case of a linear Gaussian process (see `linearGP`) which can be 
    calculated extremely quickly. 
    
    This is not recommended for inferring firing rate maps, but rather to 
    provide an instructive example of how GP inference relates to simpler KDE
    smoothers.
    
    This assumes that the measurement error is spatially uniform. This is 
    assumption is wrong, but the computational simpicity and  connection to the
    KDE estimate are instructive.
    
    Parameters
    ----------
    N: 2D array; Number of visits to each location
    K: 2D array; Number of spikes observed at each location
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    
    Returns
    -------
    λcnv: 2D array; GP rate estimate of firing rate in each bin
    gft:  2D array; Fourier transform of computed convolution kernel
    '''
    L = N.shape[0]
    if not N.shape[1]==L and K.shape==(L,L):
        raise ValueError('N and K should be L×L square arrays')
    if pad is None:
        pad  = L*1//10
    ε0 = mean(K)/mean(N) # variance per measurement
    τe = N.ravel()/ε0    # precision per bin
    # Build 2D kernel for the prior
    # Scale kernel height to match data variance (heuristic)
    k1   = blurkernel(L,σ*2)
    y    = nan_to_num(K/N)
    kern = outer(k1,k1)*var(y[mask])
    kern = repair_small_eigenvalues(kern,reg)
    knft = fft2(kern)
    τy   = τe*zeromean(y,mask).ravel()
    # Uniform measurement error ⇒ GP = convolution
    μτ   = mean((N/ε0)[mask])
    kft  = fft2(kern)
    gft  = (kft*μτ)/(kft*μτ+1)
    y    = mirrorpad(nan_to_num(K/N),pad)
    μy   = mean(y[mask])
    λcnv = conv(y-μy,gft)+μy
    return λcnv, gft
