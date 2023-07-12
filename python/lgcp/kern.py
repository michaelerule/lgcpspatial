#!/usr/bin/python
# -*- coding: UTF-8 -*-
from .util import *
from scipy.special import j0

def ensurePSD(kern,eps=0.0):
    assert eps>=0
    return ifftn(np.maximum(eps,fft2(kern).real)).real

def grid_kernel(
    P,
    shape,
    style      = 'radial',
    angle  = 0.0,
    doclip     = True,
    k          = 3,
    window     = 'square',
    doblur     = True,
    blurradius = None,
    eps        = 1e-9,
    ex         = (1,0),
    ey         = (0,1),
    ):
    '''
    Generate a periodic grid kernel. 
    
    To construct oriented kernels, set ``oriented=True`` 
    and specify the orientation ``angle``.
    To add anisotropy or skew, provide two basis vectors 
    ``e1`` and ``e2`` for the "horizontal" and "vertical" 
    components.
    
    Parameters
    ----------
    P: float>0
        Grid cell period in units of bins
    W: int>1
        Grid width (or grid size, if ``H`` is not given)
        
    Other Parameters
    ----------------
    H: int>1 or ``None`
        Grid height if different from ``W``.
    style: str; default 'radial'
        Kernel style. 
        - ``"radial"``: radially-symmetric, no orientation
        - ``"grid"``: hexagonal grid kernel
        - ``"band"``: a single plane wave
        - ``"square"``: two orthogonal plane waves
        - ``"rbf"``: a radial basis function with σ²=½(P/π)² 
            (matches grid-field size)
    angle: float, default None
        Grid orientation
    doclip: boolean; default True:
        Clip the resulting kernel to a local neighborhood?
    k: positive int; default 3
        Bessel function zero to truncate the kernel at. 
        ``k=2``: inhibitory surround
        ``k=3``: nearest neighbor grid field
        ``k≥4``: Longer-range correlations
    window: str; default ``"parzen"''
        Radial window function.  
         - ``None``: No neighborhood windowing
         - ``"square"'': Square (i.e. disk) window
         - ``"parzen": Parzen window
         - ``"triangular": Triangular window
         - ``"gaussian": Gaussian window
    doblur: boolean; default True
        Low-pass filter the resulting kernel? 
    eps: positive float; default 1e-5
        Minimum kernel eigenvalue.
    ex: 2-vector; default (1,0)
        Basis vector for "horizontal" direction.
    ey: 2-vector; default (0,1)
        Basis vector for "vertical" direction.
    '''
    if isinstance(shape,int): shape = (shape,shape)
    H,W = shape
    
    scale = 2*pi/P
    B     = np.linalg.pinv([[*ex],[*ey]])
    pxy   = fftshift(zgrid(W,H))*(exp(1j*(angle))*scale)
    pxy   = p2c(np.einsum('ab,bwh->awh',B,c2p(pxy)))
    r     = abs(pxy)

    style = str(style).lower()
    if style=='radial':
        kern = j0(r)
    elif style=='grid':
        component1 = cos(real(pxy))
        component2 = cos(real(pxy*exp(1j*(pi/3))))
        component3 = cos(real(pxy*exp(1j*(-pi/3))))
        kern = component1 + component2 + component3  
    elif style=='band':
        kern = cos(real(pxy))
    elif style=='square':
        component1 = cos(real(pxy))
        component2 = cos(real(pxy*exp(1j*(pi/2))))
        kern = component1 + component2
    elif style=='rbf':
        kern = exp(-0.25*(r)**2)
        doblur = False
    else: raise ValueError('Style %s not implemented'%str(style))
    
    # Windowing to avoid ringing in FT
    kern *= fftshift(outer(hanning(H),hanning(W)))
    # Local neighborhood window
    if doclip and not (k is None or window is None):
        cutoff = jn_zeros(0,k)[-1]#/scale
        if window=='gaussian':
            sigma = cutoff/sqrt(2)
            clip  = exp(-0.5*(r/sigma)**2)
        elif window=='parzen':
            disk  = r<cutoff/sqrt(2)
            clip  = ifft2(fft2(disk)**4).real
        elif window=='triangle':
            disk  = r<cutoff
            clip  = ifft2(fft2(disk)**2).real
        elif window=='square':
            clip  = r<cutoff
        else: raise ValueError('Window %s not implemented'%str(window))
        kern *= clip
    
    if doblur:
        kern = blur2d(kern,(P/pi)/sqrt(2))
    kern = kern/np.max(kern)
    kern = ensurePSD(kern,eps)
    return kern

def truncate(kf,wn=0.0,eth=0.1):
    if size(kf)>1:
        kept = kf >= np.max(kf.ravel()[1:])*eth
    else: kept = np.full(kf.shape,True)
    return (kf + wn)*kept

def kernelft(shape,P=None,V=1.0,angle=0.0,dc=1e3,wn=0,eth=0.1,kept=None,**kw):
    if isinstance(shape,int): shape=(shape,shape)
    shape = tuple([*shape])
    k2 = grid_kernel(P,shape,angle=angle,**kw)*V
    kf = fft2(k2).real
    kf[0,0] += dc
    if kept is None: 
        return truncate(kf,wn,eth)
    else: 
        kept = float32(kept>0)
        return (kf + wn)*kept





