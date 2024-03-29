#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
hyperparameters.py: 

Wrapper for using grid search for optimizing the prior 
kernel's hyperparamters for the DiagonalFourierLowrank
model class.
"""

from numpy import *
from lgcpspatial.gridsearch import grid_search
from lgcpspatial.lgcp2d import DiagonalFourierLowrank
from lgcpspatial.lgcp2d import coordinate_descent


def period_and_prior_variance(data,**kwargs):
    '''
    Wrapper for ``gridsearch_optimize()`` that returns 
    the optimized period and variance
    
    Parameters
    ----------
    data: lgcpspatal.loaddata.Dataset
        Prepared dataset
    
    Other Parameters
    ----------------
    **kwargs:
        Keyword arguments are forwaded to 
        ``gridsearch_optimize``, see the documentation for
        ``gridsearch_optimize`` for more details.
    
    Returns
    -------
    period: float
        Estimated grid-cell period in pixels
    v0: 
        Estiamted kernel marginal variance
    '''
    result = gridsearch_optimize(data,**kwargs)
    bestindex,bestpars,bestresult,allresults = result
    P_use  = bestpars[0]
    v0_use = data.prior_variance/bestpars[1]
    return P_use, v0_use


def gridsearch_optimize(
    data,
    np = 201, # Period search grid resolution
    nv = 201, # Kernel height search grid resolutions
    rp = 4,   # Range (ratio) to search for optimal period
    rv = 150, # Range (ratio) to search for optimal kernel height
    kclip = 3,
    variance_attenuate  = 0.5,
    verbose             = True,
    keep_frequencies    = None, 
    use_common_subspace = True,
    ): 
    '''
    Parameters
    ----------
    data: load_data.Dataset
        Prepared dataset with the following attributes:
            L: posive int
                Size of L×L spatial grid, in bins
            P: postive float
                Grid cell's heuristic period, units of bins
            prior_variance: positive float
                Heuristic kernel prior zero-lag variance. 
    
    Other Parameters
    ----------------
    np: positive odd int; default 201
        Number of grid period values to explore
        in the grid search. 
    nv: positive odd int; default 201
        Number of prior marginal variance values to explore
        in the grid search. 
    rp: float >1.0; default 4
        Ratio above and below the provided ``dataset.P`` 
        grid cell period to explore. 
        This will explore ``np`` periods evenly spaced on
        a logarithmic scale between ``dataset.P/rp`` and
        ``dataset.P*rp``.
    rv float >1.0; default 150 
        Ratio above and below the provided ``dataset.P`` 
        grid cell period to explore. 
        This will explore ``nv`` periods evenly spaced on
        a logarithmic scale between 
        ``dataset.prior_variance/rv`` and
        ``dataset.prior_variance*rv``.
    kclip: int, default 3
        Bessel zero to clip the grid-cell kernel at. 
            - 3: Nearest-neighbor grid order
                Kernel will expect positive correlations
                between adjacent grid feilds separated by
                the cell's period
            - 2: Field-repulsion at grid scale only
                Kernel will contain a local bump at the
                typical grid field scale, as well as an 
                inhibitory surround at the same width. 
            - 1: Characteristic smoothness scale only 
                (Reduces kernel to ~Gaussian bump at the 
                 size of the average grid field). 
            - >3: Larger numbers correspond to stronger
                assumptions of long-range order.
    variance_attenuate: float, default 0.5
        We recycle the estimate of the posterior variance
        between successive evaluations of the grid search.
        This can reduce the number of iterations needed 
        to converge. However, there is risk of isntabiltiy
        if the posterior marginal variance is initially 
        too large. This fraction multiplies the variance
        carried-over from nearby parameters. Set this to
        a smaller number (I suggest 0) if you run into 
        issues with the variance iteration diverging.
    verbose: boolean, default True
        Whether to print progress update
    keep_frequencies: np.ndarray or None; default None
        boolean array of frequencies to keep
    use_common_subspace: boolean, default False
        Whether to force all models to use the same low-rank
        subspace. This can make comparison of models with
        different periods less noisy. Setting this to 
        ``True`` will force the model to use a larger
        frequency subspace suitable for all grid periods,
        and will substantially slow down inference. 
    
    Returns
    -------
    bestindex: 
        best index into parameter grid
    bestpars: 
        values of best parameters
    bestresult]: 
        (state, likelihood, info) at best parameters.
        ``info`` is determined by the third element in the
        3-tuple return-value of the ``evaluate`` function,
        passed by the user. ``state`` is also user-defined.
    allresults: 
        All other results as an object array.
        Grid points that were not evaluated are None.
    '''
    
    # Verify arguments
    np = int(np)
    nv = int(nv)
    rp = float(rp)
    rv = float(rv)
    if np<=0: raise ValueError((
        'Number of search points for grid period ``np`` '
        'should be a positive odd integer, got %s')%np)
    if nv<=0: raise ValueError((
        'Number of search points for grid period ``nv`` '
        'should be a positive odd integer, got %s')%nv)
    if not np%2: np+=1
    if not nv%2: nv+=1
    if rv<=1: raise ValueError((
        'Ratio for variance search ``rv`` should be >1.0, '
        'got %s')%rv)
    if rp<=1: raise ValueError((
        'Ratio for period search ``rp`` should be >1.0, '
        'got %s')%rp)
    
    # Start with heuristic kernel parameters
    P  = data.P
    kv = data.prior_variance
    
    # Prepare hyperparameter grid
    Ps = float32(exp(linspace(log(P/rp),log(P*rp),np)))
    βs = float32(exp(linspace(log(1/rv),log(1*rv),nv))[::-1])
    pargrid = [Ps,βs]
    
    # Calculate a shared low-rank subspace
    if use_common_subspace and keep_frequencies is None:
        keep_frequencies = sum(float32([
            DiagonalFourierLowrank(kv,p,data,kclip=kclip).keep_frequencies
            for p in Ps]),0)>0
    
    def evaluate_ELBO(parameters,state):
        '''
        Function to tell grid search know which parameters 
        are good.

        Parameters
        ----------
        Parameters: tuple
            Parameters taken from the parameter search grid
        State: List of arrays; default None
            Saves initial conditions

        Returns
        -------
        state: the inferred model fit, in the form of a list 
            of floating-point numpy arrays, to be re-used as 
            initial conditions for subsequent parameters.
        log likelihood: float
            Scalar summary of fit quality, higher is better
        info: object
            Anything else you'd like to save
        '''
        p,β     = parameters
        μ,v,mh  = (None,)*3 if state is None else state
        model   = DiagonalFourierLowrank(
            kv/β,
            p,
            data,
            kclip=kclip,
            keep_frequencies=keep_frequencies)
        μ0      = None if μ is None else model.F@μ
        v0      = None if v is None else v*variance_attenuate
        mh,v,nl = coordinate_descent(model,μ0,v0)
        μ       = model.F.T@mh 
        state   = μ,v,mh
        loglike = -nl
        return state, loglike, model

    # Run overall grid search
    result = grid_search(
        pargrid,
        evaluate_ELBO,
        verbose=verbose)
    bestindex,bestpars,bestresult,allresults,_ = result

    if verbose:
        print('')
        print('Heuristic parameters')
        print('P  = %f'%P)
        print('v0 = %f'%kv)
        print('')
        print('Optimized parameters:')
        print('P    = %f'%bestpars[0])
        print('β    = %f'%bestpars[1])
        print('v0/β = %f'%(kv/bestpars[1]))
    
    return result


