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
12345678901234567890123456789012345678901234567890123456789012345678901234567890
The goal is to design a single monolithic function that encapsulates 
several different 2D gaussian process routines. This is in the style of how 
functionare are usually organized in Matlab, and the aim is to re-design the
code to the point where it can be converted and provided as a single Matlab
source file
"""

from pylab import *

from scipy.sparse.linalg import LinearOperator
from util   import zgrid, find
from basics import blur, repair_small_eigenvalues, op, cop

from scipy.sparse.linalg.isolve.minres import minres
import scipy.linalg

# Used to construct periodic kernels
from scipy.special import j0,jn_zeros

# It's useful to be able to control precision: float32 is much faster, but
# sometimes doesn't work. Jax supports this correctly, but Numpy has decided 
# not to add it. Sadly, Jax's FFT routines are a bit slow, so we need to use 
# Scipy's. This means we need to control precision manually
typesizes = {
    '32' :(np.float32   ,np.complex64),
    '64' :(np.float64   ,np.complex128),
    '128':(np.longdouble,np.longcomplex),
}
nbits = '32'
rtype = float32
ctype = complex64
def set_precision(bits):
    global nbits, rtype, ctype
    bits = '%d'%bits
    if not bits in typesizes:
        raise ValueError('# bits should be 32, 64, or 128')
    nbits = bits
    rtype,ctype = typesizes[nbits]

def precision(x,copy=False):
    '''
    ----------------------------------------------------------------------------
    Recursively cast to float32/complex64 or float64/complex128. This 
    compensates for numpy's aggressive promotion from 32→64 bit, while 
    preventing clutter elsewhere in the codebase.
    
    Parameters
    ----------
    '''
    global nbits, rtype, ctype
    if isinstance(x,ndarray):
        if np.isrealobj(x):    return rtype(array(x)) if copy or x.dtype!=rtype else x
        if np.iscomplexobj(x): return ctype(array(x)) if copy or x.dtype!=ctype else x
    try:
        if np.isrealobj(x):    return rtype(array(x))
        if np.iscomplexobj(x): return ctype(array(x))
    except ValueError:
        pass
    return (*map(precision,x),)

def coordinate_descent(gp,
        μ0           = None,
        Σ0           = None,
        maxiter      = 100,
        maxmeaniter  = 10,
        maxvariter   = 1,
        tol          = 1e-5,
        showprogress = False):
    '''
    Abstract mean/variance coordinate descent subroutine
    '''
    try: 
        report = print if showprogress else lambda *x:()
        report('(checking arguments)')
        if μ0 is None: μ0 = gp.μ0()
        if Σ0 is None: Σ0 = gp.Σ0()
        μ = precision(array(μ0))
        Σ = precision(array(Σ0))
        report('(optimizing)')
        for i in range(maxiter):
            for j in range(maxmeaniter):
                Δμ = precision(gp.mean_update(μ,Σ))
                εμ = np.max(abs(Δμ))
                report('εμ',εμ,'iteration',i,j)
                μ += Δμ
                if εμ<tol: break
            for j in range(maxvariter):
                ΔΣ = precision(gp.variance_update(μ,Σ))
                εΣ = np.max(abs(ΔΣ))
                report('εΣ',εΣ,'iteration',i,j)
                Σ += ΔΣ
                if εΣ<tol: break
            if εΣ<tol and εμ<tol: break
        return μ,Σ,gp.loss(μ,Σ),gp.likelihood(μ,Σ)
    except LinAlgError as err:
        sys.stderr.write(
            'LinAlgError encountered; Singular inverse-covariance. '
            'Increasing grid resolution or lowering the kernel variance may help.\n')
        return μ,Σ,inf,-inf
        
        
def minres_update(JHvM, minrestol=1e-5):
    def update(μ, Σ, params):
        J,Hv,M = JHvM(μ, Σ, params)
        return -minres(Hv,J,tol=minrestol,M=M)[0]
    return update
    
def minres_descent(μ0,Σ0,
        params,
        JHvMμ,
        JHvMΣ,
        loss,
        model_likelihood,
        minrestol=1e-5,
        **kwargs):
    '''
    Abstract minres-based mean/variance coordinate descent subroutine
    '''
    return foo(μ0,Σ0,
        params,
        minres_update(JHvMμ, minrestol=minrestol),
        minres_update(JHvMΣ, minrestol=minrestol),
        loss,
        model_likelihood,
        **kwargs)

def make_kernel_ft(L,P,dc=1e1,k=3,eps=1e-5):
    '''
    Generate periodic kernel and return its Fourier transform
    '''
    coords   = zgrid(L)
    window   = hanning(L)
    window   = fftshift(outer(window,window))
    kern     = j0(fftshift(abs(coords*2*pi/P)))*window
    clip     = fftshift(abs(coords)<P*jn_zeros(0,k)[-1]/(2*pi))
    kern     = kern*clip
    kern     = blur(kern,P/pi)
    kern     = repair_small_eigenvalues(kern/np.max(kern),eps)
    Kf       = np.array(real(fft2(kern)))
    Kf[0,0] += dc
    return rtype(Kf)

def chol(x):
    '''
    scipy.linalg.cholesky, with inputs/outputs set
    to the current floating-point precision
    '''
    x = rtype(x)
    return scipy.linalg.cholesky(x,lower=True)

def ltinv(ch):
    '''
    scipy.linalg.lapack.dtrtri, with inputs/outputs set
    to the current floating-point precision
    '''
    ch = rtype(ch)
    q,info = scipy.linalg.lapack.dtrtri(ch,lower=True)
    if info!=0: raise ValueError('lapack.dtrtri: '+(
            'argument %d invalid'%-info if info<0 
            else 'diagonal element %d is 0'%info))
    return q

def chinv(X):
    '''
    Invert a positive definite matrix using the Cholesky decomposition
    '''
    X = rtype(X)
    X = scipy.linalg.cholesky(X,lower=True)
    X,info = scipy.linalg.lapack.dtrtri(X,lower=True)
    if info!=0: raise ValueError('lapack.dtrtri: '+(
            'argument %d invalid'%-info if info<0 
            else 'diagonal element %d is 0'%info))
    return rtype(X)
        
def slog(x,minrate = 1e-10):
    '''
    Safe log function; Avoids numeric overflow by clipping
    '''
    return log(maximum(minrate,x), dtype=rtype)

def sexp(x,bound = 10):
    '''
    Safe exponential function; Avoids under/overflow by clipping
    '''
    return exp(np.clip(x,-bound,bound), dtype=rtype)

def logdet(A):
    '''
    Calculate the log-determinant of a positive-definite matrix using the
    Choleskey factorization. 
    '''
    A = rtype(A)
    return sum(log(diag(scipy.linalg.cholesky(A,lower=True)),dtype=rtype),dtype=rtype)*2

def RI(x):
    '''
    Sum the real and imaginary components of a complex-valued array
    '''
    return rtype(x.real+x.imag)

class diagonal_fourier_lowrank:
    '''
    Class representing variational GP inference with posterior covariance
    Σ = [ Λ + diag[p] ]¯¹

    A low-rank Fourier-space representation is used to approximately compute
    this inverse.     
    '''
    def __init__(self,n,y,lλf,lλb,L,σ0,p,
                 mintol=1e-6):
        
        if not L%2==0:
            raise ValueError('Problem size L should be even (for now).')
        
        bg = rtype(lλb.ravel())
        μ  = rtype(lλf.ravel())
        
        # Prepare kernel
        Kf = make_kernel_ft(L,p,eps=1e-7).real*σ0
        
        # Define low-rank Fourier space projection 
        thr    = array(sorted(abs(Kf).ravel()))[-2]/10 # Threshold components <10% of max
        use2d  = abs(Kf)>thr                           # Get included components
        use2d  = use2d | use2d.T                       # Ensure symmetric
        R      = sum(use2d)                            # Count # used
        use    = find(use2d.ravel())                   # Indecies into LxL array to use4
        use1d  = any(use2d,axis=0)                     # Indecies to use along L
        R1d    = sum(use1d)                            # Number of 1D components
        usecut = find(use2d[:,use1d][use1d,:])         # Indecies into cropped 2D
        f1e    = fft(eye(L),norm='ortho')[use1d,:]     # Cached diagonal fourier components
        h2e    = float32(RI(f1e[:,None,:,None]*f1e[None,:,None,:]).reshape(R1d*R1d,L*L)[usecut])
        h2e    = precision(h2e)
        def Fu(u):
            # Operator to convert from full- to low-rank subspace
            # This must be semi-orthogonal and the Hermitian transpose of Ftu (below)
            return RI(fft2(rtype(u).reshape(L,L),norm='ortho')[use2d])
        def Ftu(u):
            # Operator to convert from low-rank to full space
            # This must be semi-orthogonal and the Hermitian transpose of Fu (above)
            x = zeros((L,L)+u.shape[1:],dtype=rtype)
            x[use2d,...] = u
            return RI(fft2(x,norm='ortho',axes=(0,1))).reshape(L*L,*u.shape[1:])
        F = LinearOperator((R,L*L),matvec=Fu,rmatvec=Ftu,rmatmat=Ftu)
        
        print('Using %d components out of %d'%(R,L**2))
        
        # Convolutions and preconditioner in low-rank space
        K  = rtype(Kf[use2d])
        Λ  = rtype(1/Kf[use2d])
        M  = op(R,lambda u:K*u)
        
        # Provide some OK initial conditions, if requested
        self.μ_0 = μ
        self.μh0 = F@μ
        self.v0  = μ*0
        self.μ0  = lambda:array(self.μh0)
        self.Σ0  = lambda:array(self.v0)
        
        # Store state
        self.R      = R
        self.bg     = bg
        self.mintol = mintol
        self.Kf     = Kf
        self.K      = K
        self.Λ      = Λ
        self.σ0     = σ0
        self.F      = F
        
        # Math expressions are hard to read if self. is scattered everywhere
        # but, copying every variable as var = self.var per method would lead
        # to a horrendos cluttered code. Workaround: stash it in a tuple.        
        self.cached = (n,y,bg,L,R,use2d,F,Λ,h2e,M)

    def μλ(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ = F.T@μh
        return μ, sexp(μ+ μ0 + v/2)
        
    def fast_low_rank_covariance(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ= self.μλ(μh,v)
        x  = sqrt(n*λ, dtype=rtype)[None,:]*h2e
        C  = chinv(diag(Λ) + x@x.T)
        return C.T @ C

    def fast_covariance_diagonal(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ= self.μλ(μh,v)
        x  = sqrt(n*λ, dtype=rtype)[None,:]*h2e
        A  = chinv(diag(Λ) + x@x.T)
        X  = zeros((L,L,R),dtype=rtype)
        X[use2d] = A.T
        DF = RI(fft2(X,axes=(0,1),norm='ortho')).reshape(L**2,R).T
        return sum(DF**2,0,dtype=rtype)

    def loss(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ= self.μλ(μh,v)
        x  = sqrt(n*λ, dtype=rtype)[None,:]*h2e
        C  = chinv(diag(Λ) + x@x.T)
        l1 =  n@(λ-y*μ)         #  n'(λ-y∘μ)
        l2 =  sum(μh**2*Λ)/2    #  ½μ'Λ₀μ
        l3 = -sum(log(Λ))/2     #  ½ln|Σ₀|
        l4 = -sum(log(diag(C))) # -½ln|Σ|
        l5 =  sum(C**2 * Λ)/2   #  ½tr[Λ₀Σ]
        return l1+l2+l3+l4+l5

    def likelihood(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ  = self.μλ(μh,v)
        nyλ  =  n@(y*μ - λ)
        nλ   =  n*λ
        μΛμ  =  sum(μh**2*Λ, dtype=rtype)
        Σq   =  self.fast_low_rank_covariance(μh,v)
        ldΣq =  logdet(Σq)
        ldΣz = -sum(slog(Λ))
        return nyλ - 0.5*(μΛμ + ldΣz - ldΣq)

    def JHvM(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ = self.μλ(μh,v)
        J   = Λ*μh + F@(n*(λ-y))
        nλ  = n*λ
        Hv  = op(R,lambda u:Λ*u + F@(nλ*(F.T@u)))
        return J,Hv,M

    def mean_update(self,μh,v):
        J,Hv,M = self.JHvM(μh,v)
        return - minres(Hv,J,tol=self.mintol,M=M)[0]

    def variance_update(self,μh,v):
        return self.fast_covariance_diagonal(μh,v)-v

class diagonal_fourier_lowrank:
    '''
    Class representing variational GP inference with posterior covariance
    Σ = F'QQ'F
    
    Where F is the unitary Fourier transform with most frequency components
    discarded, and Q is a low rank matirx
    
    '''
    def __init__(self,n,y,lλf,lλb,L,σ0,p,
                 mintol=1e-6):
        
        bg = rtype(lλb.ravel())
        μ  = rtype(lλf.ravel())
        
        # Prepare kernel
        Kf = make_kernel_ft(L,p,eps=1e-7).real*σ0
        
        # Define low-rank Fourier space projection 
        thr    = array(sorted(abs(Kf).ravel()))[-2]/10 # Threshold components <10% of max
        use2d  = abs(Kf)>thr                           # Get included components
        use2d  = use2d | use2d.T                       # Ensure symmetric
        R      = sum(use2d)                            # Count # used
        use    = find(use2d.ravel())                   # Indecies into LxL array to use4
        use1d  = any(use2d,axis=0)                     # Indecies to use along L
        R1d    = sum(use1d)                            # Number of 1D components
        usecut = find(use2d[:,use1d][use1d,:])         # Indecies into cropped 2D
        f1e    = fft(eye(L),norm='ortho')[use1d,:]     # Cached diagonal fourier components
        h2e    = float32(RI(f1e[:,None,:,None]*f1e[None,:,None,:]).reshape(R1d*R1d,L*L)[usecut])
        h2e    = precision(h2e)
        def Fu(u):
            # Operator to convert from full- to low-rank subspace
            # This must be semi-orthogonal and the Hermitian transpose of Ftu (below)
            return RI(fft2(rtype(u).reshape(L,L),norm='ortho')[use2d])
        def Ftu(u):
            # Operator to convert from low-rank to full space
            # This must be semi-orthogonal and the Hermitian transpose of Fu (above)
            x = zeros((L,L)+u.shape[1:],dtype=rtype)
            x[use2d,...] = u
            return RI(fft2(x,norm='ortho',axes=(0,1))).reshape(L*L,*u.shape[1:])
        F = LinearOperator((R,L*L),matvec=Fu,rmatvec=Ftu,rmatmat=Ftu)
        
        print('Using %d components out of %d'%(R,L**2))
        
        # Convolutions and preconditioner in low-rank space
        K  = rtype(Kf[use2d])
        Λ  = rtype(1/Kf[use2d])
        M  = op(R,lambda u:K*u)
        
        # Provide some OK initial conditions, if requested
        self.μ_0 = μ
        self.μh0 = F@μ
        self.v0  = μ*0
        self.μ0  = lambda:array(self.μh0)
        self.Σ0  = lambda:array(self.v0)
        
        # Store state
        self.R      = R
        self.bg     = bg
        self.mintol = mintol
        self.Kf     = Kf
        self.K      = K
        self.Λ      = Λ
        self.σ0     = σ0
        self.F      = F
        
        # Math expressions are hard to read if self. is scattered everywhere
        # but, copying every variable as var = self.var per method would lead
        # to a horrendos cluttered code. Workaround: stash it in a tuple.        
        self.cached = (n,y,bg,L,R,use2d,F,Λ,h2e,M)

    def μλ(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ = F.T@μh
        return μ, sexp(μ+ μ0 + v/2)
        
    def fast_low_rank_covariance(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ= self.μλ(μh,v)
        x  = sqrt(n*λ, dtype=rtype)[None,:]*h2e
        C  = chinv(diag(Λ) + x@x.T)
        return C.T @ C

    def fast_covariance_diagonal(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ= self.μλ(μh,v)
        x  = sqrt(n*λ, dtype=rtype)[None,:]*h2e
        A  = chinv(diag(Λ) + x@x.T)
        X  = zeros((L,L,R),dtype=rtype)
        X[use2d] = A.T
        DF = RI(fft2(X,axes=(0,1),norm='ortho')).reshape(L**2,R).T
        return sum(DF**2,0,dtype=rtype)

    def loss(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ= self.μλ(μh,v)
        x  = sqrt(n*λ, dtype=rtype)[None,:]*h2e
        C  = chinv(diag(Λ) + x@x.T)
        l1 =  n@(λ-y*μ)         #  n'(λ-y∘μ)
        l2 =  sum(μh**2*Λ)/2    #  ½μ'Λ₀μ
        l3 = -sum(log(Λ))/2     #  ½ln|Σ₀|
        l4 = -sum(log(diag(C))) # -½ln|Σ|
        l5 =  sum(C**2 * Λ)/2   #  ½tr[Λ₀Σ]
        return l1+l2+l3+l4+l5

    def likelihood(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ  = self.μλ(μh,v)
        nyλ  =  n@(y*μ - λ)
        nλ   =  n*λ
        μΛμ  =  sum(μh**2*Λ, dtype=rtype)
        Σq   =  self.fast_low_rank_covariance(μh,v)
        ldΣq =  logdet(Σq)
        ldΣz = -sum(slog(Λ))
        return nyλ - 0.5*(μΛμ + ldΣz - ldΣq)

    def JHvM(self,μh,v):
        n,y,μ0,L,R,use2d,F,Λ,h2e,M = self.cached
        μ,λ = self.μλ(μh,v)
        J   = Λ*μh + F@(n*(λ-y))
        nλ  = n*λ
        Hv  = op(R,lambda u:Λ*u + F@(nλ*(F.T@u)))
        return J,Hv,M

    def mean_update(self,μh,v):
        J,Hv,M = self.JHvM(μh,v)
        return - minres(Hv,J,tol=self.mintol,M=M)[0]

    def variance_update(self,μh,v):
        return self.fast_covariance_diagonal(μh,v)-v
