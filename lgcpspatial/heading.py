#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
heading.py: Subroutines used in 
`example 5: heading dependence`.
"""

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from numpy import *

from collections   import defaultdict

from lgcpspatial.util          import *
from lgcpspatial.savitskygolay import SGdifferentiate as ddt
from lgcpspatial.load_data     import bin_spikes, Arena
from lgcpspatial.lgcp2d        import DiagonalFourierLowrank
from lgcpspatial.lgcp2d        import coordinate_descent
from lgcpspatial.posterior     import interpolate_peaks
from lgcpspatial.posterior     import SampledConfidence
from lgcpspatial.plot          import *
from lgcpspatial.grid_search   import grid_search


def smoothed_heading_angle(px,py,Fs=50.0,Fl=2.0):
    '''
    Calculate smoothed estimate of heading from 
    position data. 
    
    On a standard Cartesian plane, with the y axis
    increasing from bottom to top, and the x axis 
    increasing from left to right, the heading angles
    are as follows: 

      - 0   : rightwards (eastwards)
      - ½π  : upwards    (northwards)
      - π   : leftwards  (westwards)
      - 3π/2: downwards  (southwards)
    
    Parameters
    --------------------------------------------------------
    px: float32
        List of animal's location, x-coordinate
    py: float32
        List of animal's location, y-coordinate
        
    Other Parameters
    --------------------------------------------------------
    Fs: float
        Sampling rate of (px,py) position data
    Fl: float
        Low-pass cutoff frequency in Hz
        
    Returns
    ---------------------------------------------------------------------------------------------------------
    heading_angle: float32
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
    threshold=10.,
    Fs=50.0,
    Fl=2.0
    ):
    '''
    Check for location shifts based on heading. Re-weight 
    data based on cosine similarity to target heading angle.
    
    Parameters
    --------------------------------------------------------
    data: object
        Any object with the following attributes:
            L: float
                Size of LxL spatial grid for binned data.
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
                deviation from `prior_mean`.
            arena.hull:
                Convex Hull object describing the arena 
                perimeter
    model: lgcpspatial.DiagonalFourierLowrank 
        parent model instance (fitted model without heading
        filtering)
    heading_angles: np.float32 array
        List of heading angles to check.
        Westward is 0 degrees, then rotates counterclockwise 
        through southward, eastward, northward. 
    
    Other Parameters
    --------------------------------------------------------
    threshold: float
        Percentile peaks must be above to be retained.
        Should be in [0,100).
    
    Returns
    --------------------------------------------------------
    peaks: list
        List of 2×NPEAKS float32 arrays containing (x,y)
        locations of peaks at each of the angles specified
        by the list `heading_angles`.
    '''
    L,kv,P = model.L, model.kv, model.P
    
    px,py,spikes = data.px,data.py,data.spikes
    arena = Arena(px,py,L,resolution=1)
    heading_angle = smoothed_heading_angle(px,py,Fs,Fl)
    
    peaks = []        
    μh,v  = None,None # propagate initial conditions 
    for phi in progress_bar(heading_angles): 
        
        # Heading-weighted data binning
        sw  = maximum(0,cos(heading_angle-phi))
        data2 = data.reweighted(sw)
        
        # Infer and save
        hmodel = DiagonalFourierLowrank(kv,P,data2,
            prior_mean=data2.prior_mean,
            lograte_guess=data2.lograte_guess)
        μh,v,l = coordinate_descent(hmodel,
            initialmean=μh,initialcov=v,tol=1e-3)
        rate   = exp(hmodel.F.T@μh+v/2).reshape(L,L)
        thresh = nanpercentile(rate[arena.mask],threshold)
        hpeaks = interpolate_peaks(rate,height_threshold=thresh)[:2]
        ok     = is_in_hull(hpeaks.T,data.arena.hull)
        peaks.append(hpeaks[:,ok])
    
    return peaks


def match_peaks(peaks,maxd):
    '''
    Matching algorithm:
    
     - Assume q come from an array of angles on [0,2pi)
     - Get distances between all adjacent angles
     - Build directed graph joining fields of adjacent angles
     - Greedy approach
        - If you're my closest match, and I'm yours, pair up.
        - Repeat until no more edges closer than `maxd`

    Parameters
    --------------------------------------------------------
    peaks: list
        Length NANGLES list of 2×NPEAKS float32 arrays with 
        (x,y) locations of peaks at each heading angle
    maxd: 
        The maximum distance between peaks allowed when 
        connecting them

    Returns
    --------------------------------------------------------
    edges: list
        Length NANGLES list of edge sets for each pair of 
        headings. Each list entry is a 2×NEDGES int32 array.
        This contains pairs of indecies (a,b).
        For edge set i, index a is the index into peaks[i]
        (edge source) and index b is the index into 
        peaks[(i+1)%Nφ] (edge target).
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
    point sets `z1` and `z2`, limiting matches to points closer
    than `connection_radius` apart. 2D points are encoded as
    complex numbers.
    
    Parameters
    --------------------------------------------------------
    z1: iterable
        iterable of 2D points encoded as complex numbers
    z2: iterable
        iterable of 2D points encoded as complex numbers
    connection_radius: float
        Maximum radius at which to allow connections. 
        
    Returns
    --------------------------------------------------------
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
    --------------------------------------------------------
    peaks: list 
        Length NANGLES list of 2xNPEAKS np.float32 arrays 
        containing (rx,ry) peak locations at a list of 
        heading angles, as returned  by 
        `get_peaks_at_heading_angles()`.
    edges: list
        Length NANGLES list of edge sets for each pair of 
        headings, as returned by `match_peaks` or 
        `link_peaks()`. Each list entry is a 2×NEDGES int32 
        array. This contains pairs of indecies (a,b). For 
        edge set i, index a is the index into peaks[i] (edge 
        source) and index b is the index into 
        peaks[(i+1)%Nφ] (edge target).
        
    Returns
    --------------------------------------------------------
    paths: list
        List of npoints x 2 path data for each connected
        component
    chains: list
        Chained node-information in format of (iphi,ipeak)
    
    '''
    # Get components sharing edges
    # Start by finding connected fields at nearby angles
    # `cc` stores a list of sets of connected nodes.
    # The node ID format is (angle #, point ID @ angle #)
    # The point IDs are the same as the outgoing edge info #
    # in the edges datastructure.
    Nphi = len(peaks)
    cc = []
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
    paths = []
    for c in cc:
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
        path = float32([peaks[iphi][:,ipeak] \
                        for iphi,ipeak in chain])
        paths.append(path)
        chains.append(chain)
    return paths,chains


def link_peaks(
        peaks,
        maxd,
        max_end_distance = None
    ):
    '''
    Cleans up the result from `match_peaks()`, removing any
    peaks that aren't tracked unambiguously over a range of
    heading angles. 
    
    Parameters
    --------------------------------------------------------
    peaks: list
        List of 2×NPEAKS arrays containing (x,y) locations
        of identified peaks over a range of heading angles.
    maxd: int
        Maximum distance permitted between connected peaks 
        at adjacent angles.
    
    Other Parameters
    --------------------------------------------------------
    max_end_distance: float
        Maximum distance allowed between endpoints
        of a tracked peak.
        
    Returns
    --------------------------------------------------------
    edges: list
        A graph connecting peaks putatively associated with
        the same grid field at different heading angles. 
        This has the same length as the `heading_angles` 
        argument. The format is the same as the result 
        returned by `match_peaks`, but edges from peaks that
        aren't tracked unambiguously over a range of heading 
        angles have been removed.
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
        color=riley.reversed(),
        **kwargs):
    '''
    Plot connected grid fields. Use this with 
    `get_peaks_at_heading_angles()` and `link_peaks()`.
    
    Assumptions: Orient Cartesian (x,y) plane in the 
    standard way, with +y upwards and +x rightwards.
    Then heading angles are: 
    
      - 0   : rightwards (eastwards)
      - ½π  : upwards    (northwards)
      - π   : leftwards  (westwards)
      - 3π/2: downwards  (southwards)
        
    This assumes that peaks and edges are both
    taken from angles samples on [0,2π), where the
    angles are linspace(0,2*pi,len(peaks)+1)[:-1].
    
    Parameters
    --------------------------------------------------------
    peaks: list 
        Length NANGLES list of 2xNPEAKS np.float32 arrays 
        containing (rx,ry) peak locations at a list of 
        heading angles, as returned  by 
        `get_peaks_at_heading_angles()`.
    edges: list
        Length NANGLES list of edge sets for each pair of 
        headings, as returned by `match_peaks` or 
        `link_peaks()`. Each list entry is a 2×NEDGES int32 
        array. This contains pairs of indecies (a,b). For 
        edge set i, index a is the index into peaks[i] (edge 
        source) and index b is the index into 
        peaks[(i+1)%Nφ] (edge target).
        
    Other Parameters
    --------------------------------------------------------
    perim: np.float32
        NPOINTS x 2 Array of (x,y) points of the arena 
        perimeter to add to plot. Optional, default is None.
    compass: bool
        Draw colored compass rose if true
    color: matplotlib color
        Color or colormap to use for heading angles
        defaults to the custom `riley` circular color map.
    **kwargs:
        Forwarded to `plot()`
    '''
    Nphi = len(peaks)
    q = [[1,1j]@pk for pk in peaks]
    
    if not perim is None:
        plot(*perim.T,lw=2,color='w')
        plot(*perim.T,lw=1,color='k')
        
    if isinstance(color,str):
        try:
            color = matplotlib.colormaps.get_cmap(color)
        except:
            pass # Assume it's something like 'r' 'g' 'b'
    
    for i,ee in enumerate(edges):
        if len(ee)==0: continue
        ia,ib = ee
        za,zb = q[i][ia],q[(i+1)%Nphi][ib]
        lines = ravel(array([za,zb,NaN*zeros(len(za))]).T)
        c = color(i/Nphi)\
            if isinstance(color,matplotlib.colors.Colormap)\
            else color
        plot(real(lines),imag(lines),
             **{'color':c,'lw':.6,**kwargs})
    
    axis('square');
    title('Tracked peaks',pad=0)
    axis('off')
    
    if not perim is None:
        xyd = np.max(perim,0) - np.min(perim,0)
        xy0 = np.max(perim,0)
    else:
        xy1 = np.max([np.max(p,1) for p in peaks],0)
        xy0 = np.min([np.min(p,1) for p in peaks],0)
        xyd = xy1 - xy0
        xy0 = xy1
    
    if compass:
        draw_compass(
            xy0=p2c(xy0)+0.2*xyd[0] - .3j*xyd[1],
            r=mean(abs(xyd))*0.05)


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
    --------------------------------------------------------
    peaks: list
        List of 2×NPEAKS float32 arrays containing (x,y)
        locations of peaks at each of the angles specified
        by the list of heading angles `heading_angles`.
    maxd: 
        The maximum distance between peaks allowed when 
        connecting them
    starti: 
        The angle (index) to start at as a "seed".
    edges: returned by match_peaks(Nφ,q,maxd)
        lenghth Nφ list of 2×NCONNECTED edge sets containing
        indecies into the point sets (source) and targets in 
        the next adjacent direction
        
    Returns
    --------------------------------------------------------
    iop: int
        Index between 0 and Nφ-1 into the q list of the 
        opposite-heading fields
    op: int32
        Indecies of matching field in direction 
        `q[iop]` for each field `q[istart]`.
        If no match exists, -1
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
    return iop,int32(nan_to_num(op,nan=-1)),dd


def plot_connection_arrow(q1,q2,op=None,**kwargs):
    '''
    Draw arrows connecting related fields from two maps. 
    
    Parameters
    --------------------------------------------------------
    op: for every index in q1, corresponding index in q2
        or -1 if no connection
    q1: field centers in map 1 (as x+iy complex numbers)
    q2: field centers in map 2 (as x+iy complex numbers)
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


def fit_heading_variance(data,model,θ,NSEW):
    '''
    Re-optimize the prior marginal variance for the 
    given set of heading-weighted models.
    
    This is necessary for achieving interpretable posterior
    confidence intervals, since variable amounts (generally,
    less) of data are present for the different directions.
    
    Parameters
    --------------------------------------------------------
    data: Dataset
    model: Model
    θ: np.float32
        Heading angles for every time sample in Dataset
    NSEW: np.float32
        List of reference heading angles to recompute
        
    Returns
    --------------------------------------------------------
    models: list
        A list, the same length as NSEW, containing
        the models with optimized hyperparameters for each
        heading angle in NSEW
    fits: list
        A list, the same length as NSEW, containing
        the (posterior_mean, posterior_variance, loss)
        tuple returned by `coordinate_descent` for 
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
        sw    = maximum(0,cos(θ-phi))
        data2 = data.reweighted(sw)
        
        # Loss function
        kv0   = model.kv
        P     = model.P
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
    --------------------------------------------------------
    peaks: length N_HEADING_ANGLES list
        List of peak locations from 
        `get_peaks_at_heading_angles`
    edges: length N_HEADING_ANGLES list
        List of edges between peaks returned by
        `match_peaks` or `link_peaks`
    indexNSEW: list
        list of indecies into peaks and edges for the
        directions of interest (presumed to be N,S,E,W).
        
    Returns
    --------------------------------------------------------
    points: np.array
        A N_COMPONENTS x len(indexNSEW) array which 
        identifies which of the peaks at these heading
        indecies are part of the same grid field. 
        Reach row is a list of integer indecies into 
        the corresponding list in the `peaks` variable
        for the heading-angle-indecies specified by
        `indexNSEW`.
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
    exclusion_radius = None,
    edge_radius      = None,
    nsamples         = 2000,
    resolution       = 2,
    height_threshold = 0.8,
    prpeak_threshold = 0.8,
    doplot           = True,
    names            = None,
    colors           = None):
    '''
    Performed a detailed analysis of a range of heading
    angles.
    
    Apply this to the lists`models` and `fits` returned by
    `fit_heading_variance(data,model,heading,angles)`.
    
    This routine re-optimizes the kernel parameters 
    via grid-search for each heading angle, and calculates
    posterior confidence intervals via sampling.
    
    Parameters
    --------------------------------------------------------
    data: Dataset
    models: list
        List of N Models
    fits: list
        List of N fits returned by `coordinate_descent`
    angles: list of floats
        List of heading directions for each model
        
    Other Parameters
    --------------------------------------------------------
    
    Other Parameters
    --------------------------------------------------------
    exclusion_radius: positive float
        Region a peak must clear to be a local maximum,
        In units of pixels in the L×L grid.
        If `None`, defaults to `models[0].P/2.5`.
    edge_radius: positive float
        Remove peaks closer than `edge_radius` bins
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
        In units of pixels in the L×L grid.
        If `None`, defaults to `models[0].P/2.0`.
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
    
    '''
    if exclusion_radius is None:
        exclusion_radius = models[0].P/2.5
    if edge_radius is None:
        edge_radius = models[0].P/2
    
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
            color            = colors[i],
            scalebar         = i==0
        ))
        if doplot:
            title(names[i])
            
    return samples
