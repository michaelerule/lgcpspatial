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

# Common helper functions
from util import *

def ideal_hex_grid(L,P):
    '''
    ----------------------------------------------------------------------------
    Build a hexagonal grid by summing three cosine waves
    '''
    θs = exp(1j*array([0,pi/3,2*pi/3]))
    coords = zgrid(L)
    return sum(array([cos((θ*coords).real*2*pi/P) for θ in θs]),0)

def simulate_data(L=128,P=13,α=0.5,μ=0.09):
    '''
    ----------------------------------------------------------------------------
    Simulate spiking observations from a grid cell
    
    Parameters
    ----------
    L: Grid size
    P: Grid cell spacing
    α: Grid cell "sharpness"
    μ: Mean firing rate (spikes per sample)
    
    Returns
    -------
    mask : L×L binary mask of "in bounds" regions 
    λ0   : L×L array; "True" sampled grid firing rate
    λ0_bg: L×L array; Grid firing rate corrupted by background rate variations
    N    : L×L array; Number of visits to each spatial bin
    K    : L×L array; Number of spikes recorded in each spatial bin
    '''
    # Generate intensity map: Exponentiate and scale mean rate
    λ0 = exp(ideal_hex_grid(L,P)*α)
    λ0 = λ0*μ/mean(λ0)
    # Zero pad edges
    pad  = L*1//10
    mask = np0.zeros((L,L),dtype='bool')
    mask[pad:-pad,pad:-pad]=1
    # Simulate oddly shaped arena
    mask[:-L*4//10,L*3//10:L*4//10] = False
    λ0 = λ0*mask
    # For realism, add some background rate changes
    coords = zgrid(L)
    λ0_bg = λ0*(1-abs(coords/(L-2*pad)+0.1))
    # Simulated a random number of visits to each location 
    # as well as Poisson spike counts at each location
    N = poisson(2*(1-abs(coords/L-0.2j)),size=(L,L))*mask
    K = poisson(λ0_bg*N)
    return mask,λ0,λ0_bg,N,K
    