#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from pylab import *

def zgrid(W,H=None,):
    if H is None: H=W
    cw = arange(W)-W//2
    ch = arange(H)-H//2
    return 1j*ch[:,None]+cw[None,:]
def p2c(p):
    p = np.array([*p])
    if np.any(np.iscomplex(p)): return p
    which = np.where(int32(p.shape)==2)[0][0]
    p = p.transpose(which,*sorted(list({*arange(len(p.shape))}-{which})))
    return p[0]+1j*p[1]
def c2p(z):
    z = np.complex64(z)
    if np.any(np.isnan(z)): z[np.isnan(z)] = np.NaN*(1+1j)
    return np.array([z.real,z.imag])
def pdist(x1,x2):
    x1,x2 = array(x1),array(x2)
    s1,s2 = x1.shape,x2.shape
    d = abs(x1.ravel()[:,None]-x2.ravel()[None,:])
    return d.reshape((s1+s2))

############################################################
def outerslice(D,d):
    d = {d} if isinstance(d,int) else {*d}
    return tuple([np.s_[:] if j in d else None for j in range(D)])
def ndbroadcast(x,d=2):
    s = int32(np.shape(x))
    i = where(s==d)[0]
    if len(i)<=0: raise RuntimeError('No axis length %d'%d)
    i = i[0]
    l = [None,]*len(s)
    l[i] = np.s_[:]
    return tuple(l)
def truncateslice(shape):
    sl = [np.s_[:s] for s in shape]
    return tuple(sl)
def sliceat(D,d,sl,default=np.s_[:]):
    s = np.array([default,]*D,object)
    s[d] = sl
    return tuple(s)
def ndouter(*args):
    args   = [np.array(a) for a in args]
    shape  = (*(i for a in args for i in a.shape),)
    result = np.ones(shape,dtype=args[0].dtype)
    D = len(shape)
    i = 0
    for a in args:
        n = len(np.shape(a))
        d = arange(n)+i
        result = result*a[outerslice(D,d)]
        i += n
    return result
############################################################
from scipy.linalg import cholesky
from scipy.linalg.lapack import dtrtri
def chinv(X,dtype=float32): # inv(x)=C.T@C
    X = dtype(X)
    X = cholesky(X,lower=True)
    X,info = dtrtri(X,lower=True)
    if info!=0: raise ValueError('lapack.dtrtri: '+(
        'arg %d invalid'%-info if info<0 else 'diagonal element %d is 0'%info))
    return dtype(X)
from scipy.linalg import solve_triangular as stri 
def chsolve(H,v):
    C = cholesky(H)
    return float32(stri(C,stri(C.T,v,lower=True)))
############################################################
def sexp(x):
    return exp(np.clip(x,-10,10,dtype=float32),dtype=float32)
def slog(x):
    return log(maximum(1e-10,x,dtype=float32),dtype=float32)
def ssum(u,**kw):
    return np.nansum(u,dtype=float32,**kw)
def smean(x,*args,default=NaN,**kwargs):
    if np.size(x)<1: return default
    return np.nanmean(x,*args,**kwargs)   
def sdiv(A,B,fill=0.0,eps=1.0842022e-19,inf=1.8446743e+19):
    A,B = np.array(A), np.array(B,copy=True)
    assert shape(A)==shape(B)
    s,z = np.abs(A)<=eps, np.abs(B)<=eps
    B[z] = 1.0
    x = A/B
    x[z& s] = fill                 # 0/0 gets the fill value
    x[z&~s] = inf*np.sign(A[z&~s]) # Finite/0 gets "∞"
    return x
    
############################################################
import time as systime
def current_milli_time():
    return int(round(systime.time() * 1000))
__TIC_TIME__ = None
def tic(vb=True,pfx=''):
    global __TIC_TIME__
    t = current_milli_time()
    try:
        assert __TIC_TIME__ is not None
        if vb: print(pfx,'t=%dms'%(t-__TIC_TIME__))
    except: 
        if vb: print("timing...")
    __TIC_TIME__ = t
    return t
def toc(vb=True,pfx=''):
    global __TIC_TIME__
    t = current_milli_time()
    try:
        assert not __TIC_TIME__ is None
        dt = t-__TIC_TIME__
        if vb: print(pfx,'dt=%dms'%(dt))
        return t,dt
    except: 
        if vb: print("havn't called tic yet?")
    return t,None
def pbar(x,N=None):
    if N is None:
        x = list(x)
        N = len(x)
        if N<=0: return None
    K = int(np.floor(np.log10(N)))+1
    pattern = ' %%%dd/%d'%(K,N)
    wait_til_ms = systime.time()*1000
    for i,x in enumerate(x):
        time_ms = systime.time()*1000
        if time_ms>=wait_til_ms:
            r = i*50/N
            k = int(r)
            q = ' ▏▎▍▌▋▊▉'[int((r-k)*8)]
            print('\r['+('█'*k)+q+(' '*(50-k-1))+
                ']%3d%%'%(i*100//N)+(pattern%i),
                end='',flush=True)
            wait_til_ms = time_ms+1000
        yield x
    print('\r'+' '*70+'\r',end='',flush=True)
    
############################################################
"""
def zeromean(x, mask=None):
    x = float32(x)
    mask = full(x.shape,True) if mask is None else array(mask)>0
    return (x - nanmean(x.ravel()[mask.ravel()]))*mask
"""
def unitscale(x, mask=None, q0=0,q1=100):
    x = float32(x)
    mask = full(x.shape,True) if mask is None else array(mask)>0
    a,b = nanpercentile(x.ravel()[mask.ravel()],[q0,q1])
    return clip((x-a)/(b-a),0,1)*mask

############################################################
def fftfreqn(shape,shift=False):
    shape = tuple(shape)
    D  = len(shape)
    xy = np.zeros(shape+(D,))
    for d,L in enumerate(shape):
        s = fftfreq(L,1/L)
        if shift: s = fftshift(s)
        xy[...,d] = s[outerslice(D,d)]
    return float32(xy)
def blurkernel2D(V,W,H=None,normalize=False):
    if H is None: H=W
    V = np.array(V)
    Λ = eye(2)/V**2 if size(V)==1 else pinv((V + V.T)*.5)
    xy = fftfreqn((H,W),True)
    k = np.exp(-0.5*np.einsum('hwd,dD,hwD->hw',xy,Λ,xy))
    if normalize: k /= np.sum(k)
    return fftshift(k)
def fftconvf(x,K):
    if x.shape!=K.shape and size(x)==size(K):
        x = x.reshape(K.shape)
    assert x.shape==K.shape
    return ifftn(fftn(x)*K).real
def fftconvx(x,K):
    return fftconvf(x,fftn(K))
def blur2d(x,V,normalize=False):
    H,W = np.shape(x)
    k   = blurkernel2D(V,W,H,normalize)
    return fftconvx(x,k)
def fft_upsample_2D(x,factor=4):
    def circle_mask(nr,nc):
        r = (arange(nr)-(nr-1)/2)/nr
        c = (arange(nc)-(nc-1)/2)/nc
        z = r[:,None]+c[None,:]*1j
        return abs(z)<.5
    if len(x.shape)==2: x = x.reshape((1,)+x.shape)
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
def kde(N,K,sigma):
    '''
    Estimate rate using Gaussian KDE smoothing. 
    Args:
        N (2D np.array): Number of visits to each location
        K (2D np.array): Number of spikes observed at each location
        sigma (float): kernel standard deviation
    Returns: 
        KDE rate estimate of firing rate in each bin
    '''
    H,W = N.shape
    assert K.shape==N.shape
    μ = sum(K)/sum(N)
    N = blur2d(N,sigma)+0.5
    K = blur2d(K,sigma)+0.5*μ
    Y = K/N
    assert all(isfinite(Y))
    return Y

############################################################
def dyo(shape):
    dy = zeros(shape,dtype=float32)
    dy[0, 1]=-.5
    dy[0,-1]= .5
    return dy
def dxo(shape):
    return dyo(shape[::-1]).T
def hessian_2D(q):
    dx  = dxo(q.shape)
    dy  = dyo(q.shape)
    fx  = fft2(dx)
    fy  = fft2(dy)
    dxx = fftconvf(q,fx*fx)
    dxy = fftconvf(q,fy*fx)
    dyy = fftconvf(q,fy*fy)
    return array([[dxx,dxy],[dxy,dyy]]).transpose(2,3,0,1)
def h2f_2d_truncated(u,shape,use2d):
    f1 = zeros((*shape,)+u.shape[1:],dtype=float32)
    f1[use2d,...] = u
    f2 = empty(f1.shape,dtype=float32)
    f2[0 ,0 ,...] = f1[0 ,0 ,...]
    f2[1:,1:,...] = f1[1:,1:,...][::-1,::-1,...]
    f2[0 ,1:,...] = f1[0 ,1:,...][::-1,...]
    f2[1:,0 ,...] = f1[1:,0 ,...][::-1,...]
    u2 = f2[use2d,...]
    return ((u+u2) + 1j*(u-u2))*0.5

############################################################
def rgrid(shape,ftstyle=True):
    '''Distance from ND array's midpoint, in bins.'''
    D  = len(shape)
    r2 = np.zeros(shape)
    for d,L in enumerate(shape):
        m = L//2 if ftstyle else 0.5*L
        r2 += ((arange(L)-m)**2)[outerslice(D,d)]
    return r2**0.5
def radial_average(q):
    '''
    Radial average of 2D autocorrelation.
    Returns: 
        tuple (r,a): Radius (bins) and autocorrelation
    '''
    q = float32(q)
    H,W = q.shape
    i = int32(np.round(rgrid(q.shape,True)))
    r = arange(np.max(i)+1)
    a = array([smean(q[i==j]) for j in r])
    return r,a
def fft_acorr(x,mask=None,window=True):
    H, W = x.shape
    v0 = var(x) if mask is None else var(x[mask])
    if mask is None:
        x = x - mean(x)
    else:
        x = (x - mean(x[mask])) * mask
    x   = x*outer(hanning(H),hanning(W))
    psd = abs(fft2(x))**2/(W*H)
    acr = fftshift(ifft2(psd).real)
    acr = acr*v0/np.max(acr)
    return acr
def rac(y,mask=None):
    '''Radial autocorrelation of a 2D signal.'''
    return radial_average(fft_acorr(y,mask))
from scipy.signal import find_peaks
def racpeak(a,upsample=6):
    '''First autocorrelogram peak at nonzero lag.'''
    a2 = fft_upsample_1D(a,upsample)
    peaks = find_peaks(a2)[0]
    return min(peaks)/upsample if len(peaks) else NaN
def ractrough(a,upsample=6):
    a2 = fft_upsample_1D(a,upsample)
    peaks = find_peaks(-a2)[0]
    return min(peaks)/upsample if len(peaks) else NaN
def fft_upsample_1D(x,factor=4,circular=False):
    '''Upsample 1D array using the FFT.'''
    assert np.all(np.isfinite(x))
    n  = len(x)
    n2 = n*factor
    if circular: 
        f  = fftshift(fft(x))*hanning(n)
        f2 = np.complex128(np.zeros(n2))
        r0 = (n2+1)//2-(n+0)//2
        f2[r0:r0+n] = f
        return np.real(ifft(fftshift(f2)))*factor
    else:
        x = np.concatenate([x[::-1],x])
        x = fft_upsample_1D(x,factor,True)
        return x[n2:]
from scipy.special import jn_zeros
def racperiod(x, mask=None, res=50):
    r,a = rac(x,mask)      # radial acorr
    Pp  = racpeak(a,res)   # bins to 1st peak
    Pt  = ractrough(a,res) # bins to 1st trough
    # Scale with Bessel zeros to convert peaks → period
    Pp *= 2*np.pi/jn_zeros(1,2)[-1] 
    Pt *= 2*np.pi/jn_zeros(1,1)[-1]
    return Pp, Pt
    
############################################################
from scipy.spatial import ConvexHull
def points_to_qhull(px,py):
    # Encircle points in a convex hull
    points = np.array([px,py]).T
    hull   = ConvexHull(points)
    verts  = np.concatenate([hull.vertices,hull.vertices[:1]])
    perim  = points[verts]
    return hull, perim
def xygrid(shape,res=1,z=False,scale=(1.0,1.0),):
    # Generate discrete grid at desired resolution
    H,W = shape
    Wr  = W*res
    Hr  = H*res
    dx,dy = scale
    gx  = np.linspace(0,1,Wr+1+Wr)[1::2]*dx
    gy  = np.linspace(0,1,Hr+1+Hr)[1::2]*dy
    g   = np.array([
        gx[None,:]*np.ones((Hr,Wr)),
        gy[:,None]*np.ones((Hr,Wr))]) #2HW
    if z:
        return p2c(g) #HW
    return g.transpose(1,2,0) # HW2
def is_in_hull(P,hull):
    if np.size(P)<=0: return np.empty(np.shape(P),bool)
    P = c2p(p2c(P)).T # lazy code reuse: ensure array shape
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    isInHull = np.all(
        (A @ np.transpose(P))<=np.tile(-b,(1,len(P))),
        axis=0)
    return isInHull
def qhull_to_mask(hull,W,H,resolution=1):
    gxy  = xygrid((H,W),resolution)#H×W×2 
    H,W  = gxy.shape[:2]
    pts  = gxy.reshape(W*H,2).T
    return is_in_hull(pts,hull).reshape(H,W)
def mask_to_qhull(mask):
    H,W   = mask.shape
    py,px = np.array(np.where(mask))
    py=(py+.5)/H
    px=(px+.5)/W
    return points_to_qhull(px,py)
def nan_mask(mask,nanvalue=False,value=None):
    nanvalue = int(not(not nanvalue))
    if value is None: value = [1,0][nanvalue]
    use = np.float32([[np.NaN,value],[value,np.NaN]])[nanvalue]
    return use[np.int32(mask)]
def disk_kernel(W,H,R,u=10):
    '''
    2D disk kernel with oversampling anti-aliasing
    Args:
        W (int): width
        H (int): height
        R (int): radius
        u (int, default 10): oversampling resolution
    '''
    r = rgrid((H*u,W*u))/u
    k = r<=R
    k = fftshift(k)
    dh = np.kron(np.eye(H),np.full(u,1/H))
    dw = np.kron(np.eye(W),np.full(u,1/W))
    k = dh@k@dw.T
    return k
def extend_mask(mask,radius):
    '''Extend ``mask`` by distance ``radius``'''
    H,W = mask.shape
    k = disk_kernel(W,H,radius)
    return fftconvx(mask, k)>0.5

############################################################
import os
import multiprocess as multi
import traceback
from   threadpoolctl import threadpool_info,threadpool_limits
def limit_cores(CORES_PER_THREAD=1): 
    keys = ['MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
        'OMP_NUM_THREADS ','OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS']
    for k in keys: os.environ[k] = str(CORES_PER_THREAD)
    os.environ["XLA_FLAGS"] = \
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=%d"\
    %CORES_PER_THREAD
    threadpool_limits(limits=CORES_PER_THREAD, user_api='blas')
def parmaphelper(f):
    def foo(args):
        limit_cores(1)
        np.random.seed()
        i,args = args
        try:
            return i,f(*args)
        except:
            return i,f(args)
    return foo
def parmap(f,jobs):
    f = parmaphelper(f)
    jobs  = [*enumerate([*jobs])]
    njobs = len(jobs)
    with multi.Pool(multi.cpu_count()) as pool:
        results = dict(pbar(pool.imap(f,jobs),njobs))
    return [results[i] if i in results else None for i,k in enumerate(jobs)]

############################################################
def ideal_hex_grid(L,P):
    '''
    Build a hexagonal grid by summing three cosine waves
    Args:
        L (int): Rectangular binning grid size (L×L bins)
        P (positive float): Grid cell period spacing
    '''
    θs = np.exp(1j*np.float32([0,np.pi/3,2*np.pi/3]))
    coords = zgrid(L)
    return sum(
        np.float32([np.cos((θ*coords).real*2*np.pi/P) 
        for θ in θs]),0)

############################################################
def ndpoints(a,d=2):
    a = np.array(a)
    if np.any(np.iscomplex(a)):
        if d!=2: raise ValueError(
            'Cannot interpret complexas dimension=%d points'%d)
        return np.real(a), np.imag(a)
    s = np.int32(np.shape(a))
    n = len(s)
    if n==1:
        if len(a)==d:
            return a.reshape(d,1)
        if not d==1: raise ValueError(
            'Cannot interpret 1D array as %d-d points'%d)
        return a.reshape(1,len(a))
    if sum(s==d)<1: raise ValueError(
        'Array needs at least one axis dimension %d'%d)
    if sum(s==d)>2: raise ValueError((
        'Array shape %s has multiple lenght-%d dimensions, '
        'cannot unambiguously determine which axis has '
        'point coordinates.')%(s,d))
    i = np.where(s==d)[0][0]
    return a.transpose(i,*sorted([*({*arange(n)}-{i})]))
import itertools
def bin_points(p,shape,w=None,wrap=None,method='linear'):
    '''
    Bin points, using linear interpolation to distribute 
    point mass in a 2^D neighborhood.
    
    Parameters
    ----------
    p: np.float32
        D×NPOINTS array of points in (0,1)²
    shape: tuple
        Grid shape
    w: np.float32
        Weights to apply to each point, or ``None``.
    wrap: list of booleans; default None
        Either a boolean or length-D list of booleans 
        indicating whether to treat a given axis as
        circular.
    method: str
        Method to use; Can be ``"linear"`` or ``"nearest"``.
    '''
    D = len(shape)
    # Implement wrapping by adding bin and removing later
    if wrap is None:   wrap = False
    if isscalar(wrap): wrap = [float32(wrap)!=0,]*D
    wrap = float32(wrap)!=0
    assert len(wrap)==D
    oldshape = int32(shape)
    shape = [i+1 if w else i for (i,w) in zip(shape,wrap)]
    # Clean up point data
    p = ndpoints(p,D)
    N = np.prod(p.shape[1:])
    p = p.reshape(D,N)
    for pi in p:
        if not (np.min(pi)>=0 and np.max(pi)<=1):
            raise ValueError('All coords must be in (0,1)')
    # Clean up weight data
    w = np.ones(N) if w is None else np.float32(w)
    assert len(w)==N
    if method=='nearest':
        # Bin spike counts simple version
        bins = [np.linspace(0,1,L+1) for L in shape]
        N = np.histogramdd(p.T,bins,density=False,weights=w)[0]
    elif method=='linear':
        q0 = np.ones((2,)*D+(N,))
        q = []
        i,b = [],[]
        for d,(pd,L) in enumerate(zip(p,shape)):
            # Integer part: top-right bin in 2×2 neighborhood. 
            # Fractional part: how to distribute point mass
            ip,fp = divmod(pd*(L-1),1)
            # Points that are 1 go entirely in distant bin
            limit = pd>=1.0
            ip[limit] = L-1
            fp[limit] = 1.0
            assert np.max(ip)<L
            # Integer points
            qd = np.zeros((2,)*D+(N,))
            sl = [np.s_[:] if d==j else None for j in range(D)]
            sl = tuple(sl)+(np.s_[:],)
            qd = q0*float32([ip,ip+1])[sl]
            q.append(qd)
            # Fractional weights
            b.append(fp)
        q = np.float32(q)
        q = q.reshape(D,2**D*N)
        # a: weight in lower bin
        # b: weight in higher bin
        b  = np.float32(b)
        ab = [1.0-b,b]
        # All weight combinations
        z = []
        combos = int32([*itertools.product(*[(0,1)]*D)])
        for combo in combos:
            x = [ab[j][d] for d,j in enumerate(combo)]
            x = float32(x)
            x = np.prod(x,axis=0)
            z.append(x)
        z = float32(z).ravel()
        # Needs N × D shape
        bins = [np.arange(L+1) for L in shape]
        z *= np.concatenate((w,)*(2**D))
        N = np.histogramdd(q.T,bins,density=False,weights=z)[0]
    else: raise ValueError(
        'method should be "linear" or "nearest"')
    # Handle wrapped variables
    for d in where(wrap)[0]:
        N[sliceat(D,d,0)] += N[sliceat(D,d,-1)]
    N = N[truncateslice(oldshape)]
    return N
    
def bin_spikes(px,py,s,shape,w=None):
    s = float32(s)
    w = ones(s.shape) if w is None else float32(w)
    p = [py,px]
    N = bin_points(p,shape,w=w)
    K = bin_points(p,shape,w=w*s)
    return N,K
def get_edges(signal,pad_edges=True):
    if len(signal)<1:
        return array([[],[]])
    if tuple(sorted(unique(signal)))==(-2,-1):
        raise ValueError('signal should be bool or int∈{0,1};'+
            ' (using ~ on an int array?)')
    signal = int32(bool8(signal))
    starts = list(where(diff(int32(signal))==1)[0]+1)
    stops  = list(where(diff(int32(signal))==-1)[0]+1)
    if pad_edges:
        # Add artificial start/stop time to incomplete blocks
        if signal[0 ]: starts = [0]   + starts
        if signal[-1]: stops  = stops + [len(signal)]
    else:
        # Remove incomplete blocks
        if signal[0 ]: stops  = stops[1:]
        if signal[-1]: starts = starts[:-1]
    return array([array(starts), array(stops)])
def interpolate_NaN(u):
    u = array(u)
    for s,e in zip(*get_edges(~isfinite(u))):
        if s==0: 
            u[:e+1] = u[e+1]
        elif e==len(u): 
            u[s:] = u[s-1]
        else:
            a,b = u[s-1],u[e]
            u[s:e+1] = a+(b-a)*linspace(0,1,e-s+1)
    u[~isfinite(u)] = mean(u[isfinite(u)])
    return u
def patch_position_data(px,py,delta_threshold=0.01):
    '''Interpolate across glitches in tracked position.'''
    pz = px + 1j*py
    bad = where(abs(diff(pz))>delta_threshold)[0]
    pz[bad]=NaN
    z = interpolate_NaN(pz)
    assert all(isfinite(z))
    return z.real,z.imag





















