#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
simulate_data.py: Routines to simulate spiking observations 
from grid fields. Used for ground-truth for inference
routines.
"""

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
from lgcpspatial.util  import zgrid
from lgcpspatial.plot  import pscale

def ideal_hex_grid(L,P):
    '''
    Build a hexagonal grid by summing three cosine waves
    
    Parameters:
        L (int): Rectangular binning grid size (L×L bins)
        P (positive float): Grid cell period spacing
    '''
    θs = np.exp(1j*np.float32([0,np.pi/3,2*np.pi/3]))
    coords = zgrid(L)
    return sum(
        np.float32([np.cos((θ*coords).real*2*np.pi/P) 
        for θ in θs]),0)

def simulate_data(L=128,P=13,α=0.5,μ=0.09):
    '''
    Simulates spiking observations from a grid cell
    
    Parameters:
        L (int): Rectangular binning grid size (L×L bins)
        P (positive float): Grid cell period spacing
        α (positive float): Grid cell "sharpness"
        μ (positive float): Mean firing rate (spikes/sample)
    
    Returns
    --------------------------------------------------------
    mask:L×L np.bool
        Binary mask of "in bounds" regions 
    λ0: L×L np.array
        "True" sampled grid firing rate
    λ0_bg :L×L np.array
        Rate map with background variations
    N:L×L np.array
        № visits to each spatial bin
    K:L×L np.array
        № spikes recorded in each bin
    '''
    
    # Generate intensity map: Exponentiate and scale mean rate
    λ0 = np.exp(ideal_hex_grid(L,P)*α)
    λ0 = λ0*μ/np.mean(λ0)
    
    # Zero pad edges
    pad  = L*1//10
    mask = np.zeros((L,L),dtype='bool')
    mask[pad:-pad,pad:-pad]=1
    λ0 = λ0*mask
    
    # Add some background rate changes
    coords = zgrid(L)
    λ0_bg = λ0*(1-np.abs(coords/(L-2*pad)))
    
    # Simulated a random number of visits to each location 
    # as well as Poisson spike counts at each location
    N = np.random.poisson(2*(1-np.abs(coords/L-0.2j)),size=(L,L))*mask
    K = np.random.poisson(λ0_bg*N)
    
    return mask,λ0,λ0_bg,N,K
    
