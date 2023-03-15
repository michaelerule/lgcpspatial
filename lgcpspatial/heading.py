#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
heading.py: Subroutines used in 
``example 5: heading dependence``.

Several of these routines overlap and will likely be 
merged in a clean-up of this code in future commits.
"""

import warnings
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from numpy import *

from collections   import defaultdict

from lgcpspatial.util          import *
from lgcpspatial.plot          import *
from lgcpspatial.savitskygolay import SGdifferentiate as ddt
from lgcpspatial.loaddata      import bin_spikes, Arena
from lgcpspatial.lgcp2d        import DiagonalFourierLowrank
from lgcpspatial.lgcp2d        import coordinate_descent
from lgcpspatial.posterior     import interpolate_peaks
from lgcpspatial.posterior     import SampledConfidence
from lgcpspatial.posterior     import QuadraticConfidence
from lgcpspatial.gridsearch    import grid_search


def smoothed_heading_angle(px,py,Fs=50.0,Fl=2.0):
    '''
    Calculate smoothed estimate of heading from 
    position data. 
    
    On a standard Cartesian plane, with the y axis
    increasing from bottom to top, and the x axis 
    increasing from left to right, the heading angles
    are as follows: 

      - 0   : rightwards (eastwards)
      - π/2 : upwards    (northwards)
      - π   : leftwards  (westwards)
      - 3π/2: downwards  (southwards)
    
    Note: For the Krupic lab datasets, the convention is
    *not* the standard cartesian plane.
    ***Smaller** ``y`` values correspond to "more north", 
    i.e. the ``(x,y)`` coordinates should be interpreted
    as image-like, not Cartesian-like, with the ``(0,0)``
    coordinate in the upper left, i.e. the northwest corner.
    
    This library has been prepared to use the Cartesian
    convention for compatibility. You should manually
    flip the y-axis convention if you want results to 
    match Krupic-lab conventions in previously published
    manuscripts. 
    
    For image-convention coordinates, angles should
    instead be interpreted as:
    
      - 0   : rightwards (eastwards)
      - π/2 : downwards  (southwards)
      - π   : leftwards  (westwards)
      - 3π/2: upwards    (northwards)
    
    
    Parameters
    ----------
    px: float32
        List of animal's location, x-coordinate
    py: float32
        List of animal's location, y-coordinate
        
    Other Parameters
    ----------------
    Fs: float
        Sampling rate of (px,py) position data
    Fl: float
        Low-pass cutoff frequency in Hz
        
    Returns
    -------
    heading_angle: np.float32
        Heading angle based on the low-pass derivative 
        of position.
    
    '''
    dx = ddt(px,int(Fs/Fl*4),Fl,Fs)
    dy = ddt(py,int(Fs/Fl*4),Fl,Fs)
    heading_angle  = angle(dx+1j*dy)
    return heading_angle

    
def get_peaks_at_heading_angles(
    data,
    model,
    heading_angles,
    threshold        = 10.0,
    Fs               = 50.0,
    Fl               = 2.0,
    heading_angle    = None,
    show_progress    = True,
    clearance_radius = 0.45,
    return_heights   = False,
    return_models    = False,
    return_fits      = False,
    weight_function  = 'cos'
    ):
    '''
    Check for location shifts based on heading. Re-weight 
    data based on cosine similarity to target heading angle.
    
    Parameters
    ----------
    data: lgcpspatial.loaddata.Dataset
        An object with the following attributes:
            L: float
                Size of L×L spatial grid for binned data.
            n: np.float32
                Length L² array of visits to each bin.
            y: np.float32
                Length L² array of spikes at each bin.
            prior_mean: np.float32
                Shape L×L or L² array containing the prior 
                mean-log-rate. This should background rate
                variations unrelated to the grid structure
            lograte_guess: float32 array
                Shape L×L or L² array with an initial guess
                for log rate. This should be expressed as a 
                deviation from ``prior_mean``.
            arena.hull:
                Convex Hull object describing the arena 
                perimeter
    model: lgcpspatial.lgcp2d.DiagonalFourierLowrank 
        parent model instance (fitted model without heading
        filtering)
    heading_angles: np.float32 array
        List of heading angles to check. Westward is 0 
        degrees, then rotates counterclockwise through 
        southward, eastward, and northward. 
    
    Other Parameters
    ----------------
    threshold: float
        Percentile peaks must be above to be retained.
        Should be in [0,100).
    Fs: positive float; default 50.0
        Sampling rate for position data
    Fl: positive float; default 2.0
        Low-frequency cutoff for smooothing position data
    heading_angle: np.float32; default None
        Provided heading angles.
        This must be a 1D np.float32 the same length
        as ``px`` and ``py``.
        If ``None``, angles are recalvulated as 
        ``smoothed_heading_angle(px,py,Fs,Fl)``.
    show_progress: boolean; default True
        Show progress bar
    clearance_radius: positive float; default 0.45
        Neighborhood in which a peak must be a neighborhood
        maximum.
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    return_heights: boolean; default False
        Whether to return heights of peaks
    return_models: boolean; default False
        Whether to return a list of models at each angle.
    return_fits: boolean; default False
        Whether to return a list of model fits 
        at each angle.
    weight_function: str or function; default 'cos'
        Function used to compute non-negative heading
        weights as a function of the difference between
        the animals current heading, and the reference
        orientation. Can be ``'cos'`` for 
        ``max(0,cos(Δθ))``, ``'cos2'`` for
        ``max(0,cos(Δθ))²``, or a custom function. 
    
    Returns
    -------
    peaks: list
        Length ``NANGLES`` list of peak locations for each 
        heading angle. Each item is a ``2×NPEAKS`` array of 
        ``(x,y)`` peak locations in normalized ``[0,1]²`` 
        coordinates. These peaks are not matched up to any 
        particular field-IDs.
    heights: list
        **Returned only if ``return_heights=True``;**
        List of NPEAKS float32 arrays containing the
        heights of the log-rates in ``peaks``.
    models: list
        **Returned only if ``return_models=True``;**
        List of lgcp2d.DiagonalFourierLowrank model
        objects for each angle. 
    fits: list
        **Returned only if ``return_fits=True``;**
        List of tuples ``(μh,v,loss,rate)``
        containing the low-rank posterior log-rate ``μh``,
        the posterior marginal log-rate variances ``v``, 
        the model evidence lower bound ``loss``
        (up to constants), and the posterior mean rate
        map ``rate``.
    
    '''
    L,kv,P = model.L, model.kv, model.P
    
    clearance_radius *= model.P
    
    px,py,spikes = data.px,data.py,data.spikes
    arena = Arena(px,py,L,resolution=1)
    
    if heading_angle is None:
        heading_angle = smoothed_heading_angle(px,py,Fs,Fl)

    peaks   = []        
    heights = []
    models  = []
    fits    = []
    μh,v  = None,None # propagate initial conditions 
    
    for phi in (progress_bar(heading_angles) \
                if show_progress \
                else heading_angles): 
        
        # Heading-weighted data binning
        if weight_function=='cos':
            sw  = maximum(0,cos(heading_angle-phi))
        elif weight_function=='cos2':
            sw  = maximum(0,cos(heading_angle-phi))**2
        else:
            sw  = weight_function(heading_angle-phi)
        
        # Prepare new model with reweighted data
        data2 = data.reweighted(sw)
        hmodel = DiagonalFourierLowrank(kv,P,data2,
            prior_mean   =data2.prior_mean,
            lograte_guess=data2.lograte_guess)
        
        # Infer posterior mean rate
        μh,v,l = coordinate_descent(hmodel,
            initialmean=μh,initialcov=v,tol=1e-3)
        rate   = exp(hmodel.F.T@μh+v/2).reshape(L,L)

        # Find peaks
        thresh = nanpercentile(rate[arena.mask],threshold)
        hpeaks, height = interpolate_peaks(
            rate,
            clearance_radius = clearance_radius,
            height_threshold = -inf,#thresh,
            return_heights   = True)
        hpeaks = hpeaks[:2]
        
        # Restrict to hull
        ok = is_in_hull(hpeaks.T,arena.hull)
        hpeaks = hpeaks[:,ok]
        height = height[ok]
        
        # Save
        peaks.append(hpeaks)
        heights.append(height)
        models.append(model)
        fits.append((μh,v,l,rate))
    
    result = (peaks,)
    if return_heights: result += (heights,)
    if return_models:  result += (models,)
    if return_fits:    result += (fits,)
    return result


def match_peaks(peaks,maxd):
    '''
    Matching algorithm:
    
     - Assume q come from an array of angles on [0,2pi)
     - Get distances between all adjacent angles
     - Build directed graph joining fields of adjacent angles
     - Greedy approach
        - If you're my closest match, and I'm yours, pair up.
        - Repeat until no more edges closer than ``maxd``
    
    Parameters
    ----------
    peaks: list
        Length ``NANGLES`` list of ``2×NPEAKS`` ``np.float32`` 
        arrays with (x,y) locations of peaks at each 
        heading angle.
    maxd: 
        The maximum distance between peaks allowed when 
        connecting them (in the same units as ``peaks``).
    
    Returns
    -------
    edges: list
        A length ``NANGLES`` list of edge sets for 
        each pair of headings.
        Each list entry is a 2×NEDGES int32 array.
         - This contains pairs of indecies (a,b).
           - ``a`` is the index into peaks[i]
             (the node source of this edge)
          - ``b`` is the index into ``peaks[(i+1)%Nφ]`` 
            (the node target of this edge).
    
    '''
    q  = [p2c(pk) for pk in peaks]
    Nφ = len(q)
    
    # Sets of unused incoming and outgoing nodes
    osets = [{*arange(len(qi))} for qi in q]
    isets = [{*arange(len(qi))} for qi in q]
    # Pairwise distances between all adjacent angles
    D = [abs(q[i][:,None]-q[(i+1)%Nφ][None,:]) 
         for i in range(Nφ)]
    # Sets to hold edges as we build matches
    edges = [set() for i in range(Nφ)]
    for iter in range(50):
        done = True
        for i in range(Nφ):
            d   = D[i] # out (source) -> in (target)
            if prod(shape(d))==0: continue
            done = False
            # True node indecies of remaining distances
            oId = int32(sorted(list(osets[i])))
            iId = int32(sorted(list(isets[(i+1)%Nφ])))
            # Get best matches from remaining unpaired nodes
            no = argmin(d,axis=1) # best target per source
            ni = argmin(d,axis=0) # best source per target
            r  = arange(d.shape[0])
            ok = (ni[no]==r) 
            keep = ok & (np.min(d,axis=1)<=maxd)
            # These are new edges
            a,b = oId[r[ok]],iId[no[ok]]
            edges[i] |= {*map(tuple,
                zip(oId[r[keep]],iId[no[keep]]))}
            # Remove the nodes we've used
            ikeep = set(iId)-set(b)
            okeep = set(oId)-set(a)
            # Prune the distance matrix
            isets[(i+1)%Nφ] = ikeep
            osets[i]        = okeep
            D[i] = d[[j in okeep for j in oId],:]\
                    [:,[j in ikeep for j in iId]]
        if done: break
    
    edges = [int32(sorted(list(ee))).T for ee in edges]
    return edges


def pair_points(z1,z2,connection_radius):
    '''
    Greedily associate nearest-neighbors between two
    point sets ``z1`` and ``z2``, limiting matches to points closer
    than ``connection_radius`` apart. 2D points are encoded as
    complex numbers.
    
    Parameters
    ----------
    z1: iterable
        iterable of 2D points encoded as complex numbers
    z2: iterable
        iterable of 2D points encoded as complex numbers
    connection_radius: float
        Maximum radius at which to allow connections. 
        
    Returns
    -------
    index1: int32
        index into z1 of paired points
    index2: int32
        index into z2 of paired points
    '''
    z1 = c2p(complex64([*z1]))
    z2 = c2p(complex64([*z2]))
    return match_peaks([z1,z2],connection_radius)[0]


def extract_as_paths(peaks,edges):
    '''
    Convert tracked (peaks,edges) to a list of 2D paths.
    
    Parameters
    ----------
    peaks: list 
        Length ``NANGLES`` list of 2xNPEAKS np.float32 arrays 
        containing (rx,ry) peak locations at a list of 
        heading angles, as returned  by 
        ``get_peaks_at_heading_angles()``.
    edges: list
        Length ``NANGLES`` list of edge sets for each pair of 
        headings, as returned by ``match_peaks`` or 
        ``link_peaks()``. Each list entry is a 2×NEDGES int32 
        array. This contains pairs of indecies (a,b). For 
        edge set i, index a is the index into peaks[i] (edge 
        source) and index b is the index into 
        peaks[(i+1)%Nφ] (edge target).
        
    Returns
    -------
    paths: list
        Length ``N_COMPONENTS`` list of
        ``NPOINTS`` x 2 path data for each connected
        component
    chains: list
        Length ``N_COMPONENTS`` list of
        chained node-information in format of (iphi,ipeak)
    
    '''
    # Get components sharing edges
    # Start by finding connected fields at nearby angles
    # ``cc`` stores a list of sets of connected nodes.
    # The node ID format is (angle #, point ID @ angle #)
    # The point IDs are the same as the outgoing edge info #
    # in the edges datastructure.
    
    Nphi = len(peaks)
    cc   = []
    for i0 in range(Nphi):
        i1 = (i0+1)%Nphi
        if len(edges[i0])==0 or len(edges[i1])==0:
            # No peaks for one of the directions
            continue
        (a0,b0),(a1,b1) = edges[i0],edges[i1]
        for e0,e1 in zip(*where(pdist(b0,a1)==0)):
            cc.append({(i0,e0),(i1,e1)})

    # Merge components sharing edges until all merged
    while len(set.union(*map(set,cc)))!=sum([*map(len,cc)]):
        Nc = len(cc)
        for i in range(Nc):
            if len(cc[i])<=0: continue
            for j in range(i+1,Nc):
                if len(cc[i]&cc[j])>0:
                    cc[i]|=cc[j]
                    cc[j]-=cc[i]
        cc = [c for c in cc if len(c)]

    chains = []
    paths  = []
    for ic,c in enumerate(cc):
        if len(c)<Nphi//2: continue

        # Pick up the chain 
        links = {tuple(sorted([((a+iu)%Nphi,u) 
                 for iu,u in enumerate(edges[a][:,b])])) 
                 for a,b in c}
        pluck = [*links][0]
        links-= {pluck}
        chain = list(pluck)
        while len(links):
            remove = None
            match  = {chain[0],chain[-1]}
            for link in links:
                if {*link}&match:
                    a,b = link
                    if   a==chain[0 ]: chain = [b]+chain
                    elif b==chain[ 0]: chain = [a]+chain
                    elif a==chain[-1]: chain = chain+[b]
                    elif b==chain[-1]: chain = chain+[a]
                    remove = link
                    break
            links -= {remove}
        
        '''
        ``chain`` is a list of tuples ``(iphi,ipeak)``
        where a is ``iphi`` the angle index of the node
        and ``ipeak`` is the peak index for the node
        in the list if oeajs at ``iphi``
        '''
        path = []
        for il,(iphi,ipeak) in enumerate(chain):
            pp = peaks[iphi]
            if ipeak>=pp.shape[1]:
                print(
                    'extract_as_paths(peaks,edges): error',
                    file=sys.stderr)
                print(
                    'chain %d link %d'%(ic,il),
                    '(iphi,ipeak) = (%d,%d)'%(iphi,ipeak),
                    file=sys.stderr)
                print(
                    'len(peaks[iphi]): %s'%
                    shape(peaks[iphi])[1],
                    file=sys.stderr)
                raise RuntimeError((
                    'Edge index %d is out of bounds for '
                    'peak list at angle %d, '
                    'which has length %d')%(
                    ipeak,
                    iphi,
                    shape(pp)[1])
                )
            path.append(pp[:,ipeak])
        path = np.float32(path)
                          
        paths.append(path)
        chains.append(chain)
    return paths,chains



from typing import NamedTuple
class PathInfo(NamedTuple):
    path: np.ndarray
    centroid: np.ndarray

def path_information(centroids, peaks, edges, maxd=inf):
    Nphi = len(peaks)
    paths, chains  = extract_as_paths(peaks, edges)
    path_centroids = np.float32([np.nanmean(p,0) for p in paths])
    path_regions   = assign_to_regions(centroids, path_centroids, maxd=maxd)
    q = path_regions[path_regions>=0]
    if len(q) != len({*q}):
        uu,ct = np.unique(q,return_counts=True)
        bad = uu[ct>1]
        whichbad = [i for i,qi in enumerate(q) if qi in bad]
        warnings.warn((
            'Multiple paths %s all map to region(s) %s')%(
            whichbad, bad))
            
    pathinfo = {}
    for ipath, iregion in enumerate(path_regions):
        iphis,ipeaks = map(np.int32,zip(*chains[ipath]))
        path = np.full((Nphi,2),np.NaN,'f')
        path[iphis] = paths[ipath]
        pathinfo[iregion] = PathInfo(
            path,
            np.nanmean(path,0)
        )
    if -1 in pathinfo: del pathinfo[-1]
    return pathinfo


def link_peaks(
        peaks,
        maxd,
        max_end_distance = None
    ):
    '''
    Cleans up the result from ``match_peaks()``, removing 
    any peaks that aren't tracked unambiguously over a range 
    of heading angles. 
    
    Parameters
    ----------
    peaks: list
        List of 2×NPEAKS arrays containing (x,y) locations
        of identified peaks over a range of heading angles.
    maxd: int
        Maximum distance permitted between connected peaks 
        at adjacent angles.
    
    Other Parameters
    ----------------
    max_end_distance: float
        Maximum distance allowed between endpoints
        of a tracked peak.
    
    Returns
    -------
    edges: list
        A graph connecting peaks putatively associated with
        the same grid field at different heading angles. 
        The format is the same as the result 
        returned by ``match_peaks``, but edges from peaks that
        aren't tracked unambiguously over a range of heading 
        angles have been removed.
        
        This is a length ``NANGLES`` list of edge sets for 
        each pair of headings.
        Each list entry is a 2×NEDGES int32 array.
        containing pairs of indecies (a,b):
        ``a`` is the index into peaks[i]
        (the node source of this edge);
        ``b`` is the index into ``peaks[(i+1)%Nφ]`` 
        (the node target of this edge).
    
    '''

    edges = match_peaks(peaks,maxd)    
    paths,chains = extract_as_paths(peaks,edges)
        
    if max_end_distance is None:
        max_end_distance = maxd*2
        
    Nphi = len(peaks)
    new_edges = set()
    for chain in chains: 
        
        # Trim the distant ends to length
        # - Convert to complex to simplify
        # - Find centroid
        # - Repeat until chain is the right length
        #   - Remove whichever endpoint is furthest 
        #     from the current centroid
        z  = float32([peaks[i][:,j]
            for (i,j) in chain])@[1,1j]
        
        μz = mean(z)
        while len(chain)>Nphi+1 or abs(z[0]-z[-1])>max_end_distance:
            a,b = [abs(peaks[i][:,j]@[1,1j]-μz) \
                for i,j in [chain[0],chain[-1]]]
            chain,z = (chain[1:],z[1:]) if a>b \
                else (chain[:-1],z[:-1])
            μz = mean(z)
        
        # Drop short chains
        if len(chain)<Nphi//2: 
            continue
        
        # Flip chains if needed so points are in sorted order
        step = (chain[1][0] - chain[0][0] + Nphi)%Nphi
        if step==Nphi-1: 
            chain = chain[::-1]
        else: 
            assert step==1
        
        new_edges |= {*zip(chain[:-1],chain[1:])}

    # Rebuild edge datastructure
    edges2 = [[] for i in range(Nphi)]
    for (iph0,px0),(iph1,px1) in sorted([*new_edges]):
        edges2[iph0]+=[(px0,px1)]
    edges = [int32(e).T for e in edges2]
    
    return edges


def plot_tracked_peaks(
    peaks,
    edges,
    perim=None,
    compass=True,
    color=riley,
    origin='lower',
    hideaxis=True,
    **kwargs):
    '''
    Plot connected grid fields. Use this with 
    ``get_peaks_at_heading_angles()`` and ``link_peaks()``.
    
    
    #### Angle conventions:
    
    The indecies ``iφ ∈ {0,..,NANGLES-1}``into the length-
    ``NANGLES`` arguments ``peaks`` and ``edges`` are assumed 
    to correpond to equally-spaced angles, starting at 
    ``φ=0``; that is, ``phi = linspace(0,2*pi,NANGLES+1)[:-1]``.
    
    
    #### Colors:
    
    This is designed to be used with the ``riley`` 
    colormap from ``lgcpspatial.plot``. This map starts at 
    mauve, continues through rust, olive, blue, before 
    circling back to mauve. We use these color–direction
    conventions: 
    
     - North: blue/azure/cyan
     - South: red/rust
     - East:  green/olive
     - West:  purple/mauve/magenta
    
    For comatibility, then, the direction ordering
    for the colormap parameter ``color`` should be
    
        {West, South, East, North}
    
    
    #### Axis orientation:
    
    If the keyword argument ``origin`` is ``'lower'`` (the
    default value), heading angles are interpreted as
    the usual definition of polar coordinates on the
    Cartesian plane. That is, we assume an axis where
    ``(x,y)=(0,0)`` corresponds to the *lower left* 
    (southwest) corner, and measure angles as staring
    at ``0`` for "East", then increasing as we rotate
    counter-clockwise:
    
      - 0   : rightwards (eastwards)
      - π/2 : upwards    (northwards)
      - π   : leftwards  (westwards)
      - 3π/2: downwards  (southwards)
    
    If the keyword argument ``origin`` is ``'upper'``, use
    image-coordinate conventions. This is the convention 
    used in Krupic lab datasets and publications.
    For image coordinates, ``(x,y)=(0,0)`` corresponds to the
    top-left (northwest) corner. 
    
      - 0   : rightwards (eastwards)
      - π/2 : downwards  (southwards)
      - π   : leftwards  (westwards)
      - 3π/2: upwards    (northwards)
    
    **Note:** Unlike ``matplotlib.imshow()``, using
    ``oring='upper'`` will cause the y-axis to be flipped
    so that smaller ``y`` values are at the top. 
    
    Parameters
    ----------
    peaks: list 
        Length ``NANGLES`` list of 2xNPEAKS np.float32 arrays 
        containing (rx,ry) peak locations at a list of 
        heading angles, as returned  by 
        ``get_peaks_at_heading_angles()``.
    edges: list
        Length ``NANGLES`` list of edge sets for each pair of 
        headings, as returned by ``match_peaks`` or 
        ``link_peaks()``. Each list entry is a 2×NEDGES int32 
        array. This contains pairs of indecies (a,b). For 
        edge set i, index a is the index into peaks[i] (edge 
        source) and index b is the index into 
        peaks[(i+1)%Nφ] (edge target).
    
    Other Parameters
    ----------------
    perim: np.float32
        NPOINTS x 2 Array of (x,y) points of the arena 
        perimeter to add to plot. Optional, default is None.
    compass: bool
        Draw colored compass rose if true
    cmap_fn: matplotlib.colors.Colormap
        colormap to use for heading angles
        defaults to the custom ``riley`` circular color map.
        Should be a callable :``[0,1]→(r,g,b,a)``
        where color components ``rgba`` are floats in ``[0,1]``.
    origin: str; default 'lower'
        String in ``{'lower','upper'}`` (compare to 
        ``matplotlib.imshow()``). 
    hideaxis: boolean; default True
        Hide the axis
    **kwargs:
        Forwarded to ``plot()``
    
    '''
    
    # Argument validation
    ori = str(origin).lower()[0]
    if not ori in 'lu':
        raise ValueError((
            "The ``origin`` parameter should be 'lower' or "
            "'upper'; hot '%s'")%origin)
    origin = ori
    
    # look up color map string
    if isinstance(color,str):
        try:
            color = matplotlib.colormaps.get_cmap(color)
        except:
            pass
    
    Nphi = len(peaks)
    q = [[1,1j]@pk for pk in peaks]
    
    # Plot the boundary if given
    if not perim is None:
        plot(*perim.T,lw=2,color='w')
        plot(*perim.T,lw=1,color='k')
    
    # Plot segments between each angle pair as a single
    # NaN-delimited path (faster)
    for i,ee in enumerate(edges):
        # Skip angles with no edges
        if len(ee)==0: continue
        # Get (source, target) indecies of edges
        ia,ib = ee
        # Look up point positions
        za,zb = q[i][ia],q[(i+1)%Nphi][ib]
        # Generate NaN-delimited line-segment data
        lines = ravel(array([za,zb,NaN*zeros(len(za))]).T)
        # Pick color
        if isinstance(color,matplotlib.colors.Colormap):
            r = i/Nphi
            '''
            ### For ``origin='lower'``: 
            
            Color map order + conventions: 
             - 0.00 West:  purple/mauve/magenta
             - 0.25 South: red/rust
             - 0.50 East:  green/olive
             - 0.75 North: blue/azure/cyan
            Angle order (r * 2π):
             - 0   : rightwards (eastwards)
             - π/2 : upwards    (northwards)
             - π   : leftwards  (westwards)
             - 3π/2: downwards  (southwards)
            We need to re-map as this: 
                0.00 → 0.50
                0.25 → 0.75
                0.50 → 0.00
                0.75 → 0.25
            This can be achieved by adding 0.5 and
            taking modulo 1. 

            ### For ``origin='upper'``:
            
            Angle order (r * 2π):
             - 0   : rightwards (eastwards)
             - π/2 : downwards  (southwards)
             - π   : leftwards  (westwards)
             - 3π/2: upwards    (northwards)

            We need to re-map as this: 

                0.00 → 0.50
                0.25 → 0.25
                0.50 → 0.00
                0.75 → 0.75

            This can be achieved by flipping q to 1-q,
            then adding .5 and taking modulo 0
            '''
            if origin=='u': r = 1-r
            c = color((r+.5)%1.0)
        plot(real(lines),imag(lines),
             **{'color':c,'lw':.6,**kwargs})
    
    force_aspect()
    title('Tracked peaks',pad=0)
    if hideaxis: 
        axis('off')
    
    if not perim is None:
        xyd = np.max(perim,0) - np.min(perim,0)
        xy0 = np.max(perim,0)
    else:
        xy1 = np.max([np.max(p,1) for p in peaks],0)
        xy0 = np.min([np.min(p,1) for p in peaks],0)
        xyd = xy1 - xy0
        xy0 = xy1
    
    # Flip axis first, draw_compass will notice this.
    if origin=='u':
        y0,y1 = sorted(ylim())
        ylim(y1,y0)
    
    if compass:
        draw_compass(
            xy0   = p2c(xy0)+0.2*xyd[0] - .3j*xyd[1],
            r     = mean(abs(xyd))*0.04,
            delta = 30,
            cmap_fn = color)


def locate_opposites(peaks,maxd,starti,edges):
    '''
    To be used on the result of calling match_peaks()

    ### Algorithm: 
    
     - We have a list of edges between adjacent angles
     - Starting from a seed, follow the graph in both 
       directions half-way around
     - Hopefully, we'll come to a peak from the opposite 
       heading direction
    
    Parameters
    ----------
    peaks: list
        Length ``NANGLES`` list of ``2×NPEAKS`` ``np.float32`` 
        arrays with (x,y) locations of peaks at each 
        heading angle.
    maxd: 
        The maximum distance between peaks allowed when 
        connecting them
    starti: int ∈ {0,..,``NANGLES``-1}
        The angle (index) to start at as a "seed".
    edges: returned by match_peaks(Nφ,q,maxd)
        lenghth ``NANGLES`` list of ``2×NCONNECTED`` edge sets 
        containing indecies into the point sets (source) and
        targets in  the next adjacent direction.
    
    Returns
    -------
    iop: int (scalar)
        Index ``iop ∈ {0,..,NANGLES-1}`` opposite of the 
        direction associated with the seed index given by
        ``starti ∈ {0,..,NANGLES-1}``. 
        This should be ``( starti + NANGLES//2 ) % NANGLES``.
    op: np.int32
        Indecies of matching field in direction 
        ``peaks[iop]`` for each field ``peaks[istart]``
        (or -1 is no match exists).
    dd: float32
        Distance to matched peak, or NaN it not matched.
    
    '''
    q = [[1,1j]@pk for pk in peaks]
    Nφ = len(q)
    i = starti
    # Search forward and backward for connected peak in 
    # opposite direction
    fop = arange(len(q[i]))
    bop = copy(fop)
    iop = (Nφ//2+i)%Nφ
    for j in range(Nφ//2):
        d   = dict(edges[(i+j)%Nφ].T)
        fop = [(d[n] if n in d else NaN) for n in fop]
        d   = dict(edges[(i+Nφ-1-j)%Nφ][::-1].T)
        bop = [(d[n] if n in d else NaN) for n in bop]
        
    # Check distance to opposite match in both directions
    df = abs(q[i] - [q[iop][j] if isfinite(j) \
                     else NaN for j in fop])
    db = abs(q[i] - [q[iop][j] if isfinite(j) \
                     else NaN for j in bop])
    
    # Pick closer opposite
    # - Default to the forward search result (may be none)
    # - If the backward match is closer, use that instead
    # - Update the pair match and distance
    op          = float32(fop)
    replace     = (db<df) | ~isfinite(op)
    op[replace] = array(bop)[replace]
    dd          = nanmin([df,db,full(db.shape,inf)],axis=0)
    
    # Make sure opposite is still reasonably nearby
    bad = dd>maxd
    op[bad] = NaN
    dd[bad] = NaN
    # Check uniqueness: each target picks closest source
    targets = {*op[isfinite(op)]}
    for target in targets:
        used = where(op==target)[0]
        if len(used)<=1: continue
        #print('target',target,'used by',used)
        other = used[argmax(dd[used])]
        pick  = {fop[other],bop[other]}-targets-{nan}
        if len(pick)==1:
            pick      = int([*pick][0])
            op[other] = pick
            dd[other] = abs(q[i][other]-q[iop][pick])
            #print('Remapped',other,'to',pick)
        else:
            op[other] = dd[other] = NaN
    return iop,np.int32(nan_to_num(op,nan=-1)),dd


def plot_connection_arrow(q1,q2,op=None,**kwargs):
    '''
    Draw arrows connecting related fields from two maps. 
    
    Parameters
    ----------
    q1: np.complex64
        field centers in map 1 (as x+iy complex numbers)
    q2: np.complex64
        field centers in map 2 (as x+iy complex numbers)
    op: np.int32
        for every index in q1, corresponding index in q2
        (or -1 if no connection). 
    '''
    if op is None:
        op = arange(len(q1))
    q1 = p2c(q1)
    q2 = p2c(q2)
    a = exp(0.35j-.35) # For arrows
    lines = []
    for j,m1 in enumerate(q1):
        if op[j]>=0:
            m2 = q2[op[j]]
            v = m1-m2
            lines+=[m2+v*a,m1,m2,NaN,m1,m2+v*conj(a),NaN]
    plot(real(lines),imag(lines),**{'color':RUST,'lw':.4,**kwargs})


def fit_heading_variance(
    data,
    model,
    θ,
    NSEW,
    weight_function='cos'
    ):
    '''
    Re-optimize the prior marginal variance for the 
    given set of heading-weighted models.
    
    This is necessary for achieving interpretable posterior
    confidence intervals, since variable amounts (generally,
    less) of data are present for the different directions.
    
    Parameters
    ----------
    data: Dataset
    model: Model
    θ: np.float32
        Heading angles for every time sample in Dataset
    NSEW: np.float32
        List of reference heading angles to recompute
        
    Other Parameters
    ----------------
    weight_function: str or function; default 'cos'
        Function used to compute non-negative heading
        weights as a function of the difference between
        the animals current heading, and the reference
        orientation. Can be ``'cos'`` for 
        ``max(0,cos(Δθ))``, ``'cos2'`` for
        ``max(0,cos(Δθ))²``, or a custom function. 
        
    Returns
    -------
    models: list
        Length 4 list of [N,S,E,W] results, each containing
        the models with optimized hyperparameters.
    fits: list
        Length 4 list of [N,S,E,W] results, each containing
        the ``(posterior_mean, posterior_variance, loss)``
        tuple returned by ``coordinate_descent`` for 
        optimizing the model for each heading angle in NSEW.
    '''
    # Prepare hyperparameter grid
    rβ = 4  # Range (ratio) to search for optimal kernel height
    Nβ = 21 # Kernel height search grid resolutions
    βs = float32(exp(linspace(log(1/rβ),log(1*rβ),Nβ))[::-1])
    pargrid = [βs]

    models, fits = [],[]
    for iphi,phi in enumerate(NSEW): 
        
        # Heading-weighted data binning
        if weight_function=='cos':
            sw  = maximum(0,cos(θ-phi))
        elif weight_function=='cos2':
            sw  = maximum(0,cos(θ-phi))**2
        else:
            sw  = weight_function(θ-phi)
        
        data2 = data.reweighted(sw)
        
        # Loss function
        kv0 = model.kv
        P   = model.P
        def evaluate_ELBO(parameters,state):
            β       = parameters[0]
            μ,v,μh  = (None,)*3 if state is None else state
            model   = DiagonalFourierLowrank(kv0/β,P,data2)
            μh,v,nl = coordinate_descent(model)
            return (model.F.T@μh ,v,μh), -nl, model

        # Run grid search
        bestindex,bestpars,bestresult,allresults = grid_search(
            pargrid,evaluate_ELBO)
        kv = kv0/bestpars[0]
        print('σ0   = %f'%kv0)
        print('β    = %f'%bestpars[0])
        print('σ0/β = %f'%kv)

        # Infer posterior and save
        model = DiagonalFourierLowrank(kv,P,data2)
        fit   = coordinate_descent(model,tol=1e-8)
        models.append(model)
        fits.append(fit)
    return models, fits


def matched_cardinal_points(peaks,edges,indexNSEW):
    '''
    For each connected component from tracking peaks over
    various heading angles, 
    return the z+iy location of all NSEW-facing points
    
    Parameters
    ----------
    peaks: length ``NANGLES`` list
        List of peak locations from 
        ``get_peaks_at_heading_angles``
    edges: length ``NANGLES`` list
        List of edges between peaks returned by
        ``match_peaks`` or ``link_peaks``
    indexNSEW: list
        list of indecies into peaks and edges for the
        directions of interest (presumed to be N,S,E,W).
    
    Returns
    -------
    points: np.array
        A ``N_COMPONENTS × len(indexNSEW)`` array which 
        identifies which of the peaks at these heading
        indecies are part of the same grid field. Missing
        directions are filled in with ``NaN``. 
    '''
    # Find NSEW pairings from the peak tracking
    # List of point locations
    paths,chains = extract_as_paths(peaks,edges)
    points = []
    for i,(p,c) in enumerate(zip(paths,chains)):
        ix_angle, ix_peak = int32(c).T
        exist = np.any(indexNSEW[:,None]==ix_angle,1)
        which = argmax(indexNSEW[:,None]==ix_angle,1)
        points += [
            [NaN if ~e else p2c(peaks[j][:,ix_peak[w]]) \
             for j,w,e in zip(indexNSEW,which,exist)]
        ]
    return array(points)


def sample_heading_angles(
    data,
    models,
    fits,
    angles,
    exclusion_radius = 0.4,
    edge_radius      = 0.2,
    nsamples         = 2000,
    resolution       = 2,
    height_threshold = 0.9,
    prpeak_threshold = 0.01,
    doplot           = True,
    names            = None,
    colors           = None,
    confidence_mode  = 'quadratic'):
    '''
    Sample confidence intervals, and plot, for a collection
    of models fit to (North, South, East, West) heading 
    angles. 
    
    Apply this to the lists``models`` and ``fits`` returned 
    by ``fit_heading_variance(data,model,heading,angles)``.
    
    This routine re-optimizes the kernel parameters 
    via grid-search for each heading angle, and calculates
    posterior confidence intervals via sampling.
    
    Parameters
    ----------
    data: Dataset
    models: list
        List of N Models
    fits: list
        List of N fits returned by ``coordinate_descent``
    angles: list of floats
        List of heading directions for each model
    
    Other Parameters
    ----------------
    radius: positive float; default 0.45
        Region in which peak must be a local maximum to 
        be considered a grid field. 
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    edge_radius: positive float; Default 0.0
        Remove peaks closer than ``edge_radius`` bins
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    nsamples: positive int
        Number of posterior samples to use
    resolution: positive int; default 2
        Upsampling factor for posterior sampling
    height_threshold: positive float ∈[0,1]
        Fraction of locations a peak must be taller than
        to be included
    prpeak_threshold: positive float ∈[0,1]
        Fraction of samples a grid field must be present in
        to be included
    doplot: boolean; default True
        Whether to draw plots
    names: list of str
        Titles of each plot
    colors: lost of Matplotlib colors
        List of plotting colors for each heading angle
    confidence_mode: str; default 'quadratic'
        Can be ``'quadratic'`` or ``'voronoi'``
    
    Returns
    -------
    samples: list
        Length ``len(angles)`` list of sampled peak-density
        maps. Each element is a 
        ``lgcpspatial.posterior.SampledConfidence`` 
        object with attributes:
            ellipses: np.float32
                NaN-delimeted (x,y) coordinates for plotting 
                confidence ellipses; 
            gaussians: list
                List of mean, covariance for each peak;
                each entry is a tuple of
                    μ: shape (2,) np.float32
                        The peak mean
                    Σ: shape (2,2) np.float32
                        Covariance of gaussian convidence
                        interval.
            samples: PosteriorSample
                A PosteriorSample object with attributes:
                density:
                    Counts of total number of times a field 
                    peak appeared at each location for all 
                    samples.
                pfield:
                    Normalized (sum to 1) density of peak 
                    locations for each grid field. Grid 
                    fields are defined as a local region 
                    around each local maximum in the peak 
                    ``density`` map.
                peaks: np.float32
                    2xNPEAKS array of grid-field-peak (x,y) 
                    coordinates.
                totals:
                    Number of samples within each peak basin
                    that actually contained a peak.
                means: np.float32
                    Center of mass of each peak
                sigmas: np.float32
                    2D sampled covariance of each peak
                nearest: np.float32
                    (L*resolution)x(L*resolution) map of 
                    Voronoi regions for each peak
                kde: np.float32
                    ``(L*resolution)×(L*resolution)`` smoothed 
                    peak density map.
    '''
    
    kplot = None
    if doplot:
        kplot = int(ceil(sqrt(len(angles))))
        figure(0,(3,3),200)
        subplots_adjust(0,0,1,1,0,0.1)

    samples = []
    for i,phi in enumerate(angles): 
        model = models[i]
        fit   = fits[i]
        if doplot:
            subplot(kplot,kplot,i+1)
        if confidence_mode=='quadratic':
            samples.append(SampledConfidence(
                data,
                model,
                fit,
                radius           = exclusion_radius,
                edge_radius      = edge_radius,
                resolution       = resolution,
                nsamples         = nsamples,
                height_threshold = height_threshold,
                prpeak_threshold = prpeak_threshold,
                pct              = 95,
                doplot           = True,
                color            = (0,)*4,
                scalebar         = i==0
            ))
            QuadraticConfidence(
                data,
                model,
                fit,
                radius           = exclusion_radius,
                edge_radius      = edge_radius,
                height_threshold = height_threshold,
                pct              = 95,
                doplot           = doplot,
                color            = None if colors is None else colors[i],
            )
        elif confidence_mode=='voronoi':
            samples.append(SampledConfidence(
                data,
                model,
                fit,
                radius           = exclusion_radius,
                edge_radius      = edge_radius,
                resolution       = resolution,
                nsamples         = nsamples,
                height_threshold = height_threshold,
                prpeak_threshold = prpeak_threshold,
                pct              = 95,
                doplot           = doplot,
                color            = None if colors is None else colors[i],
                scalebar         = i==0
            ))
        else: raise ValueError((
            'confidence_mode can be quadratic or voronoi, '
            'got %s')%confidence_mode)
        
        if doplot and names is not None:
            title(names[i])
    return samples


def match_with_tracked_peaks(
    peaks, 
    edges,
    angle_index,
    target_peaks,
    target_peaks_opposite,
    opposites_maxd,
    connection_radius):
    '''
    Match "fast-tracked" peak shifts with 
    posterior-peak-density maps.    
    
    ### Details: 
    
    In this workflow, we track shifts in single
    peaks over a broad range by re-fitting a single model
    over a high-resolution sweep of heading angles
    ("fast track"). Shifted peaks that are connected by a 
    continuous range of heading angles are assumed to 
    reflect shifted copies of the same grid field.
    
    To get confidence intervals, however, we need an
    interpretable posterior, which requires re-calibrating
    the prior variance. This is slow, so we do this for 
    only a few directions (e.g. N,S,E,W). 
    
    The resulting posterior-peak-densities won't exactly
    match the "fast-tracked" peaks, since they were 
    computed using different procedures. They should, 
    however, be *very* close. 
    
    This function matches them. 
    
    ### Procedure:
    
    Fast-tracked peaks are stored as lists of peak locations
    ``peaks`` and edge-indecies ``edges``, for all heading 
    angles. These are computed using the 
    ``lgcpspatial.heading`` module as follows:
    
        ``
        peaks = get_peaks_at_heading_angles(
            data,model,angles,heading_angle)
        edges = link_peaks(peaks,maximum_point_distance)  
        ``
    
    To figure out which shifted peaks are copies of the
    same field, we: 
    
     1. Walk the ``edge`` graph half-way around to find 
        connected peaks in opposite directions. 
     2. Find nearest neighbors with ``target_peaks`` and
        ``target_peaks_opposite``
     3. Connect indecies in ``target_peaks`` to the index
        in ``target_peaks_opposite`` connected to the same
        fast-tracked component (if any).
    
        
    Parameters
    ----------
    peaks: list 
        Length ``NANGLES`` list of 2 × NPEAKS np.float32  
        arrays containing (rx,ry) peak locations at a list 
        of heading angles, as returned by 
        ``get_peaks_at_heading_angles()``.
    edges: list
        Length ``NANGLES`` list of edge sets for each pair of 
        headings, as returned by ``match_peaks`` or 
        ``link_peaks()``. Each list entry is a 2 × NEDGES  
        int32 array. This contains pairs of indecies 
        ``(a,b)``. For edge set ``i``, index ``a`` is the index 
        into ``peaks[i]`` (edge source) and index ``b`` is the 
        index  into ``peaks[(i+1)%Nφ]`` (edge target).
    angle_index: int ∈ {0,..,NANGLES-1}
        Which angle to match, represented as an index
        into the peaks/edges array.
    target_peaks: 2 × NPEAKS_A np.float32
        Reference peak-density centroids for heading
        direction ``angle_index`` to match.
    target_peaks_opposite: 2 × NPEAKS_A np.float32
        Reference peak-density centroids for heading
        direction **opposite** to ``angle_index`` to match.         
    opposites_maxd: positive float
        Tolerance for matching opposite-direction peaks 
        from the continuous tracking, in the same
        units as the points ``peaks``. (This is most likely
        **normalized [0,1]² coordinates** if you're 
        following the provided example workflows). 
        This should be similar to the grid scale, to avoid 
        associating peaks from different fields in the 
        region where the posterior is very noisy. 
        I suggest
        ``opposites_maxd = P/L*0.75``, where ``P`` is the
        grid-cell period in pixels and ``L`` is the 
        resolution of the discrete bin (again, in pixels).
    connection_radius: positive float
        Tolerance for matching fast-tracked peaks with
        posterior-peak-density centroids, in the 
        same units as ``peaks. This is most likely
        **normalized [0,1]² coordinates** if you're 
        following the provided example workflows). 
        This should be small, since we're just trying to 
        account for small changes. I suggest
        ``connection_radius = P/L/5``.
        
    Returns
    -------
    paired_indecies: 2 × NMATCHED np.int32
        For each of the ``NMATCHED`` peak pairs, 
        the first row is the index into 
        ``target_peaks`` and the second an index into
        ``target_peaks_opposite``.
    '''
    
    # Validate arguments 
    assert all([np.all(np.isfinite(p)) for p in peaks])
    assert np.all(np.isfinite(target_peaks))
    assert np.all(np.isfinite(target_peaks_opposite))
    
    # Find opposite direction and connections with it
    '''
    Match peaks based on the "continues" tracking over
    many angles. These were tracked without re-calibrating
    the prior variance, so their locations are generally
    correct but the posterior variances aren't 
    interpretable. We use them to decide which peaks are
    shifted version of the same field. 

    peaks: list
        Length ``NANGLES`` list of ``2×NPEAKS`` ``np.float32`` 
        arrays with (x,y) locations of peaks at each 
        heading angle.

    iop: int
        Index ``iop ∈ {0,..,NANGLES-1}`` opposite of the 
        direction associated with the seed index given by
        ``starti ∈ {0,..,NANGLES-1}``. 
        This should be ``( starti + NANGLES//2 ) % NANGLES``.

    op: int32
        Indecies of matching field in direction 
        ``peaks[iop]`` for each field ``peaks[istart]``
        (or -1 is no match exists).
    '''
    iop,op = locate_opposites(
        peaks,
        opposites_maxd,
        angle_index,
        edges)[:2] 
    print(iop)

    # encode peak means as a complex number
    q  = [[1,1j]@pk for pk in peaks]
    z1 = [1,1j]@target_peaks
    z2 = [1,1j]@target_peaks_opposite

    # Get list of peak locations in the reference direction
    # (q[i]) and the opposite direction (q[iop])
    q1,q2 = q[angle_index],q[iop]
    
    
    '''
    Register connected-peak data from ``locate_opposites``
    (fast approximate peak tracking) with 
    the posterior-confidence information computed by the
    (accurate by slow) posterior sampling with 
    recalibrated prior variance. 
    These should be close, but aren't exactly equal
    Match them based on proximity.

     - Get pairwise distances in each direction d1, d2
     - Get the closest matches in each direction i1, i2
     - Remove pairs too far apart
     - Check the paired-opposite-heading-peaks computed 
       earlier
     - For each (heading, opposite-heading) pair
       - Skip pairs with no match
       - Retrieve the posterior-sample peak index for each
         peak in the pair.
       - If both peaks exist in the posterior sample
         - Match the pair in the posterior samples
    '''
    d1,d2 = pdist(z1,q1),pdist(z2,q2)
    i1,i2 = np.nanargmin(d1,0),argmin(d2,0)
    i1[np.nanmin(d1,0)>=connection_radius] = -1
    i2[np.nanmin(d2,0)>=connection_radius] = -1
    op2 = int32(zeros(len(z1)))-1
    for j,k in enumerate(op):
        if k<0: continue
        l,m = i1[j],i2[k]
        if l>=0 and m>=0: op2[l]=m
    
    paired_indecies = [
        (j,k) for j,k in enumerate(op2) if k>0]
    paired_indecies = np.int32(paired_indecies).T
    return paired_indecies


def maybe_aliased(a,b,c,threshold=1.05):
    '''
    Deal with aliasing: 
    if matches look ambiguous, remove them.
    
    Fields are marked as postentially aliased if there
    is another opposite-direction match closeby, or
    if they are equally close to a different grid field
    centroid. 
    
    Parameters
    ----------
    a: 1D np.complex64
        Peak locations in direction 1 as complex numbers
    b: 1D np.complex64
        Peak locations in direction 2 as complex numbers
    c: 1D np.complex64
        Field centroids as complex numbers
        
    Returns
    -------
    bad: np.bool
        Indicator list of which fields might be aliased. 
    '''
    # Is there another match with 5% of this distance? 
    d    = abs(a-b).ravel()*threshold
    D    = pdist(a,b).squeeze()
    bad  = (np.nansum(D<=d[:,None],1)>1)
    bad |= (np.nansum(D<=d,0)>1) 
    # Is there another centroid that is closer? 
    bad |= (np.nansum(pdist(a,c)<=abs(a-c)[:,None]*threshold,1)>1)  
    bad |= (np.nansum(pdist(b,c)<=abs(b-c)[:,None]*threshold,1)>1) 
    return bad
