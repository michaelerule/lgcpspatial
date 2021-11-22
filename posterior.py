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

def findpeaks(q,th=-inf,r=1):
    '''
    ----------------------------------------------------------------------------
    Find points higher than threshold th, that are also higher than all other 
    points in a square neighborhood with radius r. 
    '''
    L  = q.shape[0]
    D  = 2*r
    Δ  = range(D+1)
    q0 = q[r:-r,r:-r,...]
    p  = q0>th
    for i,j in {(i,j) for i in Δ for j in Δ if i!=r or j!=r}:
        p &= q0>=q[i:L+i-D,j:L+j-D,...]
    p2 = zeros(q.shape,bool)
    p2[r:-r,r:-r,...] = p
    return p2

def dx_op(L):
    '''
    ----------------------------------------------------------------------------
    '''
    # 2D difference operator in the 1st coordinate
    dx = zeros((L,L))
    dx[0, 1]=-.5
    dx[0,-1]= .5
    return dx

def hessian_2D(q):
    '''
    ----------------------------------------------------------------------------
    Get Hessian at all points
    '''
    dx  = dx_op(q.shape[0])
    f1  = fft2(dx)
    f2  = fft2(dx.T)
    d11 = conv(q,f1*f1)
    d12 = conv(q,f2*f1)
    d22 = conv(q,f2*f2)
    return array([[d11,d12],[d12,d22]]).transpose(2,3,0,1)

def circle_mask(nr,nc):
    '''
    ----------------------------------------------------------------------------
    Zeros out corner frequencies
    '''
    r = (arange(nr)-(nr-1)/2)/nr
    c = (arange(nc)-(nc-1)/2)/nc
    z = r[:,None]+c[None,:]*1j
    return abs(z)<.5

def fft_upsample_2D(x,factor=4):
    '''
    ----------------------------------------------------------------------------
    Upsample 2D array using the FFT
    '''
    if len(x.shape)==2:
        x = x.reshape((1,)+x.shape)
    nl,nr,nc = x.shape
    f = fftshift(fft2(x),axes=(-1,-2))
    f = f*circle_mask(nr,nc)
    nr2,nc2 = nr*factor,nc*factor
    f2 = complex128(zeros((nl,nr2,nc2)))
    r0 = (nr2+1)//2-(nr+0)//2
    c0 = (nc2+1)//2-(nc+0)//2
    f2[:,r0:r0+nr,c0:c0+nc] = f
    x2 = real(ifft2(fftshift(f2,axes=(-1,-2))))
    return squeeze(x2)*factor**2
