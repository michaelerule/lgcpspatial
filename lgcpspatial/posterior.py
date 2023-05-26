#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
posterior.py: Subroutines for further analysis of the posterior rate map returned from Gaussian process inference. 
"""

# Load a Matlab-like namespace
from numpy         import *
from scipy.spatial import ConvexHull
from numpy.linalg  import norm,LinAlgError

from lgcpspatial.loaddata import bin_spikes, Arena
from lgcpspatial.lgcp2d   import chinv,chsolve,RI,LGCPResult
from lgcpspatial.plot     import *
from lgcpspatial.util     import *

def findpeaks(
    q,
    height_threshold = -inf,
    clearance_radius = 1):
    '''
    Find points higher than height_threshold, that are also 
    higher than all other points in a radius r circular
    neighborhood.
    
    Parameters
    ----------
    q: np.float32
        2D array of potential values
        
    Other Parameters
    ----------------
    height_threshold: float
        Peaks must be higher than this to cound.
    clearance_radius: int
        Peaks must be larger than all other pixels in radius
        ``clearance_radius`` to count.
        
    Returns
    -------
    :np.bool
        2D boolean array of the same sape as q, indicating
        which pixels are local maxima within radius ``r``.
    '''
    L  = q.shape[0]
    clearance_radius = max(1.0,clearance_radius)
    
    # Add padding
    rpad = max(1,int(np.ceil(clearance_radius)))
    Lpad = L+2*rpad;
    qpad = zeros((Lpad,Lpad)+q.shape[2:],dtype=q.dtype)
    qpad[rpad:-rpad,rpad:-rpad,...] = q[:,:,...]

    # Points to search
    Δ = range(-rpad,rpad+1)
    limit = clearance_radius**2
    search = {(i,j) 
              for i in Δ 
              for j in Δ 
              if (i!=0 or j!=0) and (i*i+j*j)<=limit}
    
    # Only points above the threshold are candidate peaks
    p = q>height_threshold
    
    # Mask away points that have a taller neighbor
    for i,j in search:
        p &= q>qpad[i+rpad:L+i+rpad,j+rpad:L+j+rpad,...]
    
    return p


def dx_op(L):
    '''
    Parameters
    --------------------------------------------------------
    L: int
        Size of L×L spatial grid
    '''
    # 2D difference operator in the 1st coordinate
    dx = zeros((L,L))
    dx[0, 1]=-.5
    dx[0,-1]= .5
    return dx


def hessian_2D(q):
    '''
    Get Hessian of discrete 2D function at all points
    
    Parameters
    --------------------------------------------------------
    q: np.complex64
        List of peak locations encoded as x+iy complex
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
    Zeros out corner frequencies
    
    Parameters
    --------------------------------------------------------
    nr: int
        number of rows in mask
    nc: int
        number of columns in mask
    '''
    r = (arange(nr)-(nr-1)/2)/nr
    c = (arange(nc)-(nc-1)/2)/nc
    z = r[:,None]+c[None,:]*1j
    return abs(z)<.5


def fft_upsample_2D(x,factor=4):
    '''
    Upsample 2D array using the FFT
    
    Parameters
    --------------------------------------------------------
    x: 2D np.float32
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


def interpolate_peaks(
    z,
    clearance_radius = 1,
    height_threshold = None,
    return_heights   = False
    ):
    '''
    Obtain peak locations by quadratic interpolation
    
    Parameters
    ----------
    z: ndarray, L×L×NSAMPLES
        A 3D array of sampled 2D grid-fields,
        where the LAST axis is the sample numer. 
        
    Other Parameters
    ----------------
    clearance_radius: integer
        Radius over which point must be local maximum to 
        include. 
        Defaults to 1 (nearest neighbors).
    height_threshold: float
        Threshold (in height) for peak inclusion. 
        Defaults to the 25th percentil of z
    return_heights: boolean; default False
        Whether to return heights as a second return
        valude
        
    Returns
    -------
    peaks: tuple
        either ``(ix,iy)`` coordinates of peaks if
        ``q`` is a 2D array, or ``(ix,iy,iz)`` coordinates
        if ``q`` is a 3D array. The ``iz`` coordinate
        will label the sample of ``q`` each peak belongs to.
    heights: list
        **Returned only if ``return_heights=True``**; 
        The height of each peak
    '''
    z = np.array(z)
    L = z.shape[0]
    is3d = True
    if len(z.shape)==2:
        z = z.reshape(L,L,1)
        is3d = False
    
    # Peaks are defined as local maxima that are larger than
    # all other points within radius ``r``, and also higher
    # than the bottom 25% of log-rate values. 
    if height_threshold is None:
        height_threshold=nanpercentile(z,25)

    # Local indecies of local maxima    
    peaks = findpeaks(
        z,height_threshold,clearance_radius)
    rx,ry,rz = where(peaks)
    
    # Get heights at peaks
    heights = z[peaks]
    
    # Use quadratic interpolation to localize peaks
    clip = lambda i:np.clip(i,0,L-1)
    rx0 = clip(rx-1)
    rx2 = clip(rx+1)
    ry0 = clip(ry-1)
    ry2 = clip(ry+1)
    s00 = z[rx0,ry0,rz]
    s01 = z[rx0,ry ,rz]
    s02 = z[rx0,ry2,rz]
    s10 = z[rx ,ry0,rz]
    s11 = z[rx ,ry ,rz]
    s12 = z[rx ,ry2,rz]
    s20 = z[rx2,ry0,rz]
    s21 = z[rx2,ry ,rz]
    s22 = z[rx2,ry2,rz]
    dx  = (s21 - s01)/2
    dy  = (s12 - s10)/2
    dxx = s21+s01-2*s11
    dyy = s12+s10-2*s11
    dxy = (s22+s00-s20-s02)/4
    det = 1/(dxx*dyy-dxy*dxy)
    ix  = (rx-( dx*dyy-dy*dxy)*det + 0.5)/L
    iy  = (ry-(-dx*dxy+dy*dxx)*det + 0.5)/L
    # Rarely, ill-conditioning leads to inappropriate 
    # interpolation. We remove these cases. 
    bad = (ix<0) | (ix>1-1/L) | (iy<0) | (iy>1-1/L)
    ix  = ix[~bad]
    iy  = iy[~bad]
    
    peaks = float32((iy,ix,rz[~bad]) if is3d else (iy,ix))
    if return_heights:
        heights = heights[~bad]
        return peaks, heights
    else:
        return peaks


def get_peak_density(z,resolution,r=1,height_threshold=None):
    '''
    Obtain peaks by quadratic interpolation, then bin
    the results to a spatial grid with linear interpolation.
    
    Parameters
    ----------
    z: ndarray, L×L×NSAMPLES
        A 3D array of 2D grid field samples,
        where the LAST axis is the sample numer. 
    resolution: int>1
        Upsampling factor for binned peal locations
        
    Other Parameters
    ----------------
    r: int (default 1)
        Radius over which point must be local maximum to include. 
    height_threshold: float
        Threshold (in height) for peak inclusion. 
        Defaults to the 25th percentile of z
    '''
    L = z.shape[0]
    # Get list of peak locations
    iy,ix = interpolate_peaks(z,
        clearance_radius = r,
        height_threshold = height_threshold)[:2]
    # Bin peaks on a (possibly finer) spatial grid with
    # linear interpolation. 
    return bin_spikes(iy,ix,0*iy,L*resolution)[0]


def sample_posterior_lograte(model,posterior_mean,v,nsamples=200):
    '''
    Sample from the Gaussian log-rate posterior. 
    
    Parameters
    --------------------------------------------------------
    model: 
        a ``diagonal_fourier_lowrank`` model object 
    posterior_mean: float32
        low-rank posterior mean returned by 
        ``coordinate_descent(model)``
    v: float32
        marginal variances returned by 
        ``coordinate_descent(model)``
        
    Other Parameters
    ----------------
    nsamples: int>1
        Number of samples to draw; 
        Default is 200
        
    Returns
    -------
    z: L×L×nsamples float32 array
    '''
    
    # Get saved state from the model then 
    # Retrieve the posterior mean log-rate and mean-rate
    n,y,μ0,L,R,use2d,F,Λ,h2e,M = model.cached
    μ,λ = model.rates_from_lowrank(posterior_mean,v)

    # Get Cholesky factor of low-rank posterior covariance
    # Σq = inv(Λq) = Cq' Cq and Λq = CΛq CΛq'
    x = sqrt(n*λ,dtype='f')[None,:]*h2e # ⎷ of precisions
    C = chinv(float32(diag(Λ) + x@x.T)) # Σ Cholesky
    
    # Draw samples from Gaussian posterior
    # Δ: diplacement from low-d mean 
    # zh: low-d samples
    Δ  = C@float32(np.random.randn(R,nsamples)) 
    zh = Δ + posterior_mean[:,None]      
    
    # Convert log-rate samples to position space and return
    return (h2e.T@zh).reshape(L,L,nsamples)


class PosteriorSample:
    '''
    Sample the distribution of grid-field peaks from 
    a log-Gaussian posterior distribution.

     1. Draw samples in the low-rank space
     2. Convert them to spatial coordinates
     3. Get probable location of each grid field peak

    (Treat this class simply as a function with named
    return values)

    Attributes
    ----------
    density:
        Counts of total number of times a field peak
        appeared at each location for all samples.
    pfield:
        Normalized (sum to 1) density of peak locations
        for each grid field. Grid fields are defined as 
        a local region around each local maximum in the 
        peak ``density`` map.
    peaks: np.float32
        2xNPEAKS array of grid-field-peak (x,y) 
        coordinates.
    totals:
        Number of samples within each peak basin that
        actually contained a peak.
    means: np.float32
        Center of mass of each peak
    sigmas: np.float32
        2D sampled covariance of each peak
    nearest: np.float32
        (L*resolution)x(L*resolution) map of Voronoi 
        regions for each peak
    kde: np.float32
        (L*resolution)x(L*resolution) smoothed peak 
        density map.
    
    Parameters
    ----------
    model: 
        a ``diagonal_fourier_lowrank`` model object 
    posterior_mean: 
        low-rank posterior mean returned by 
        ``coordinate_descent(model)``
    posterior_variance: 
        marginal variances returned by 
        ``coordinate_descent(model)``

    Other Parameters
    ----------------
    Arena: object
        An object with an atrribute
        hull: scipy.spatial._qhull.ConvexHull
            Convex Hull object describing the arena 
            perimeter. Points will be clipped to this if
            it is not None. Ensure this convex hull has 
            been scaled up by the same factor if 
            ``resolution`` is >1.
    nsamples: int>1
        Number of samples to draw; 
        Default is 200
    resolution: int>0
        Upsampling factor compared to the bin-grid 
        resolution; Default is 2
    radius: float
        Local region (in units of pixels on the L×L
        grid) that peaks should clear to be included.
        I suggest P/2.5
    edge_radius: float; Default 0
        Remove peaks closer than ``edge_radius`` bins
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
    height_threshold: float
        Peak height threshold
    prpeak_threshold: float
        Probability threshold for including a peak,
        in [0,1], based on the fractions of samples
        that contain a given peak.
    '''
    def __init__(self,
        model,
        posterior_mean,
        posterior_variance,
        arena            = None,
        nsamples         = 200,
        resolution       = 2,
        radius           = 0.5,
        edge_radius      = 0.0,
        height_threshold = 0,
        prpeak_threshold = 0):
        # Get log-rate samples
        z = sample_posterior_lograte(
            model,
            posterior_mean,
            posterior_variance,
            nsamples)
        L = model.L
        
        # Count peaks at each location with linear interp.
        density = get_peak_density(z,resolution,
            height_threshold=percentile(
                z,100*height_threshold))/nsamples
        
        # Normalize the region around each of the  
        # identified grid-field centers
        # Focus on a local radius of r=P/2.5
        # Gaussian blur (σ=r/8) density to find fields
        pfield  = zeros(shape(density))
        
        # Get density-peak centroids with interpolation
        # Peak locations are returned in [0,1] coordinates
        kde   = blur(density,radius*resolution/7)
        peaks = interpolate_peaks(
            kde,
            clearance_radius = int(radius*resolution))
        
        # Remove points outside hull or too close to edge
        keep  = is_in_hull(peaks.T,arena.hull)
        if edge_radius>0. and not arena is None:
            keep = keep & ~arena.close_to_boundary(
                peaks, 
                edge_radius)
        
        # Calculate Voronoi region around each peak.
        # It is useful to retain out-of-bounds peaks
        # here since these limit the domain of peaks
        # close to the edge of the arena.
        rz      = [1,1j]@peaks
        lr      = L*resolution
        grid    = zgrid(lr)/lr+.5*(1+1j)
        D       = abs(grid[:,:,None]-rz[None,None,:])
        nearest = float32(argmin(D,axis=2))
        nearest[nanmin(D,axis=2)>model.P/L]=NaN

        field_ids = unique(nearest)
        field_ids = int32(field_ids[isfinite(field_ids)])

        # Iterate over all grid fields
        # Normalize density within each field
        npeaks = sum(keep)
        totals = full(npeaks,NaN,'f')
        means  = full((npeaks,2),NaN,'f')
        sigmas = full((npeaks,2,2),NaN,'f')

        if np.sum(keep)<=0:
            raise RuntimeError(
                'No valid peaks were found')
        
        peaks = peaks[:,keep]
        for i,id in enumerate(find(keep)):
            match         = nearest==id
            regional      = kde[match]
            totals[i]     = sum(regional)
            if totals[i] <= 0.0: continue
            pr            = regional/totals[i]
            pfield[match] = pr
            # Calculate moments
            mu        = sum(grid[match]*pr)
            delta     = c2p(grid[match]-mu)
            sigmas[i] = einsum('zN,ZN,N->zZ',delta,delta,pr)
            means[i]  = c2p(mu)
        
        keep   = (totals > prpeak_threshold*nsamples) & \
            is_in_hull(means,arena.hull)
        peaks  = peaks[:,keep]
        totals = totals[keep]
        means  = means [keep]
        sigmas = sigmas[keep]
        
        self.density = density
        self.pfield  = pfield
        self.peaks   = peaks
        self.totals  = totals
        self.means   = means
        self.sigmas  = sigmas
        self.nearest = nearest
        self.kde     = kde


class SampledConfidence:
    '''
    Construct confidence intervals for grid-field peaks
    using sampling, and (optionally) plot them.

    **For small posterior variance:** The resulting
    confidence intervals may be too large, since 
    spurious peaks near the edges of a field's Voronoi 
    cell are counted toward the posterior peak-location
    variance. This is conservative.

    **For large posterior variance:** We estimate the
    uncertainty in a peak's location by measuring
    the variance of each peak in the peak-density map.
    These samples are limited to each peak's Voronoi 
    cell, which creates an upper bound on the 
    uncertainty. I.e. this will always report that
    grid-field peaks are localized to within one 
    grid-cell period. 
    
    Attributes
    ----------
    samples: PosteriorSample
        A ``PosteriorSample`` object.
    ellipses: 
        NaN-delimeted (x,y) coordinates for plotting 
        confidence ellipses; 
    gaussians: List
        (μ,Σ) 2D confidence Gaussians for all peaks.   
    
    Parameters
    ----------
    result: LGCPResult
    
    Other Parameters
    ----------------
    radius: positive float; default 0.45
        Region in which peak must be a local maximum to 
        be considered a grid field. 
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    edge_radius: positive float; Default 0.0
        Remove peaks closer than ``edge_radius`` 
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    resolution: positive integer
        Grid upsampling factor, defaults to 2
    nsamples: positive integer
        Number of samples to draw, defaults to 4000
    height_threshold: float in (0,1)
        Inclusion threshold for peak height in each 
        sample and probability peak over all samples,
        default:0
    prpeak_threshold: float
        Probability threshold for including a peak,
        in [0,1], based on the fractions of samples
        that contain a given peak. Default is 0
    pct: float in (0,100)
        Percentile to use for confidence bounds,
        default is 95%.
    doplot: boolean
        Plot resulting grid map? Default is False.
    cmap: matplotlib.colormap
        Forwarded to ``imshow()`` if ``doplot=True``
    draw_scalebar: boolean; default True
        Draw 1 meter scale bar?
    draw_colorbar: boolean; default True
        Add a color bar axis?
    **kwargs:
        Forwarded to ``plot()`` if ``doplot`` is True
    
    Returns
    -------
    SampledConfidence object with these attributes:
        samples: PosteriorSample
            A PosteriorSample object.
        ellipses: 
            (x,y) coordinates for confidence ellipses,
            NaN delimeted.
        gaussians: 
            List of (μ,Σ) 2D confidence Gaussians.
        arena: 
            An Arena object representing the experiment
            arena at the upsampled resolution.
    
    '''
    def __init__(self,
        data,
        model = None,
        fit   = None,
        radius           = 0.45,
        edge_radius      = 0.0,
        resolution       = 2,
        nsamples         = 4000,
        height_threshold = 0.95,
        prpeak_threshold = 0.,
        pct              = 95,
        doplot           = False,
        cmap             = 'bone_r',
        draw_scalebar    = True,
        draw_border      = True,
        draw_colorbar    = True,
        **kwargs):
        
        if isinstance(data,LGCPResult):
            model = data.model
            fit   = data.fit
            data  = data.data
        
        L = data.L

        # Retrieve the posterior mode (in low-D frequency)
        # and variance (in space)
        posterior_mean,posterior_variance = fit[:2]
        
        radius      *= model.P
        edge_radius *= model.P
        
        # Sample from the GP posterior (see posterior.py)
        result = PosteriorSample(
            model,
            posterior_mean,
            posterior_variance,
            arena            = data.arena,
            nsamples         = nsamples,
            resolution       = resolution,
            radius           = radius,
            edge_radius      = edge_radius,
            height_threshold = height_threshold,
            prpeak_threshold = prpeak_threshold)
        
        # peaks: 2 × NPEAKS (x,y) peak locations
        # prpeak: fractions of samples with peak here
        # pfield:
        #     normalized distribution of peak location for
        #     each Voronoi-segmented grid field
        # means: estimated centroid of each field
        # sigmas: 2x2 covariance ellipse of each field
        peaks  = result.peaks
        prpeak = result.density
        pfield = float32(result.pfield)
        means  = result.means
        sigmas = result.sigmas
        
        # Convert scale
        m1 = data.scale*(L+1)*resolution
        dx = (100/m1)**2 # (cm/pixel)²
        pfield /= dx
        nfields = peaks.shape[1]    

        # Collect all ellipses to plot 
        ellipses,gaussians = [],[]
        for i,(mu,sigma) in enumerate(zip(means,sigmas)):
            if all(isfinite(sigma)) and all(isfinite(mu)):
                # Prepare covariance ellipse for plotting
                z = covariance_crosshairs(
                    sigma,
                    p=.95,
                    draw_cross=False) + mu[:,None]
                ellipses +=[*z.T]+[(NaN,NaN)]
                gaussians+=[(mu,sigma)]
        ellipses = array(ellipses).T

        # Get higher resolution masks for upsampled data
        arena = Arena.from_mask(
            data.arena.mask,
            resolution=resolution)
        
        # Create a plot of sampled confidence intervals
        if doplot:
            # Plot peak densities
            vmax = ceil(percentile(
                result.kde[arena.mask],99.9)*100)/100
            imshow(result.kde*arena.nanmask,
                extent=(0,1)*2,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                origin='lower')
            if draw_border:
                plot(*arena.perimeter.T,lw=3,color='w')

            truepct = r'\%' if \
                plt.rcParams['text.usetex'] else '%'
            title('Probable centers of each grid field'\
                  '\n Ellipses: 95'\
                  +truepct+' confidence')

            if draw_scalebar:
                # Add a scale bar
                y0 = where(any(arena.mask,axis=1)
                    )[0][-1]/(L*resolution)
                y1 = y0 - m1/(L*resolution)
                x0 = (where(any(arena.mask,axis=0)
                    )[0][0]-L/20*resolution)/(L*resolution)
                yscalebar((y0+y1)/2,y1-y0,'1 m',x=x0,
                          fontsize=8)

            # Add color bar
            if draw_colorbar:
                good_colorbar(0,vmax,
                    cmap,'$\\Pr($peak$)$ / cm²',
                    fontsize=7,sideways=0,vscale=0.7)
            plot(*ellipses,**{
                'lw':0.5,
                'color':OCHRE,
                **kwargs})
            axis('off')
            ylim(1,0)

        self.arena     = arena
        self.samples   = result
        self.ellipses  = ellipses
        self.gaussians = gaussians


class QuadraticConfidence:
    '''
    Locally-quadratic approximation of peak-location
    confidence intervals.

    **For small posterior variance:** The locally-
    quadratic intervals tend to be smaller than the
    peak-density map sampled intervals, because they
    exclude spurious peaks at intermediate locations.
    This has been spot-checked and found true using
    shuffle tests on the Krupic data, but should be
    verified again when using new data. 

    **For large posterior variance:** The locally-
    quadratic intervals are not interpretable, since
    they do not account for the fact that grid fields
    are confined to a region related to the grid perid. 
    Incorrect, large values will be returned. These
    should be discarded by setting the 
    ``localization_radius`` parameter.       

    Attributes
    ----------
    ellipses: list
        List of x,y coordinates for plotting confidence 
        ellipses; NaN delimeted.
    gaussians: list 
        List of (μ,Σ) 2D confidence Gaussians for 
        all peaks.

    Parameters
    ----------
    result: LGCPResult

    Other Parameters
    ----------------
    radius: float; Default 0.45
        Region in which a peak must be a local maximum 
        to count as a grid field. 
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    localization_radius: float; Default 0.8
        Drop peaks with confidence outside this radius.
        Set to ``inf`` to retain all peaks. 
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    edge_radius: float; Default 0
        Remove peaks closer than ``edge_radius`` bins
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
    height_threshold: float in (0,1); default 0.5
        Inclusion threshold for peak height in each 
        sample and probability peak over all samples,
    prpeak_threshold: float
        Probability threshold for including a peak,
        iinterpolate_peaksn [0,1], based on the 
        fractions of samples that contain a given peak.
    pct: float in (0,100); default 95.0
        percentile to use for confidence bounds,
    doplot: boolean, default True
        Render a plot of the grid map? Default:False. 
    draw_border: boolean, default True
        Whether to render the arena border if plotting.
    '''
    def __init__(
        self,
        data,
        model = None,
        fit   = None,
        radius              = 0.45,
        localization_radius = 0.8,
        edge_radius         = 0,
        height_threshold    = 0.95,
        pct                 = 95,
        doplot              = False,
        draw_border         = True,
        **kwargs):
        
        if isinstance(data,LGCPResult):
            model = data.model
            fit   = data.fit
            data  = data.data
        
        _,_,_,L,R,use2d,F,_,h2e,_ = model.cached
        P = model.P
        radius              = float(radius)*P
        localization_radius = float(localization_radius)*P
        edge_radius         = float(edge_radius)*P

        # Retrieve the posterior mode (in low-D space) and 
        # variance (in regular space)
        # Sample from the GP posterior (see posterior.py)
        posterior_mean,v = fit[:2]
        μ ,λ = model.rates_from_lowrank(posterior_mean,v)
        μ    = μ.reshape(L,L)

        # Get peaks in arena
        # Note that these are peaks in the log-rate (μ)
        # not the rate, λ. 
        peaks = interpolate_peaks(μ,
            clearance_radius = int(round(radius)),
            height_threshold =
            percentile(μ,100*height_threshold))[:2]
        peaks = peaks[:,is_in_hull(peaks.T,data.arena.hull)]
        if edge_radius>0.:
            peaks = data.arena.remove_near_boundary(
                peaks, 
                edge_radius)
        
        # Get negative of 2×2 Hessians at all points
        pidx = (*int32(peaks[::-1]*L+0.5),)
        H    = -hessian_2D(μ)[pidx]

        # Get post.r mean Hessian and covariance gradients
        # - Get low-rank Cholesky factor of covariance
        # - Construct discrete d/dx in low-D Hartley space
        # - Obtain low-rank derivatives (left-multiply)
        # - Project back to spatial domain only at peaks
        #   by keeping only those columns in the inverse  
        #   transform model.h2e that correspond to a peak 
        #   location.
        Q   = model.low_rank_cholesky(posterior_mean,v).T
        dx  = dx_op(L)
        fx  = fft2(dx  ,norm='ortho')[use2d]
        fy  = fft2(dx.T,norm='ortho')[use2d]
        fQ  = h2f_2d_truncated(Q,L,use2d)
        dxQ = RI((fx.T*fQ.T).T)
        dyQ = RI((fy.T*fQ.T).T)

        # At this point, dxQ and dyQ are the x and y
        # derivatives applied on the left to the Cholesky
        # factor of the posterior covariance. Together, they
        # describe the derivatives of the posterior 
        # covariance at all locations. In lgcp2d, h2e is 
        # defined as the R×L² matrix of 2D fourier
        # components. Cutting out only the [pidx] locations
        # from this operator corresponds to inverting the
        # Hartley transform only at the locations where
        # peaks exist (saves time). 
        ispk = h2e.reshape(R,L,L).T[pidx]
        J    = array([ispk@dxQ,ispk@dyQ]).transpose(1,0,2)

        # Calculate covariances, prepare crosshairs 
        # - Get the expected peak shift covariance Σx0
        # - If peak is localized, plot confidence ellipse
        # - Collect all ellipse to plot at once (faster)
        ellipses,gaussians,bad = [],[],[]
        retained_peaks = []
        for (mx,h,j) in zip(peaks.T,H,J):
            try:
                ΣxJD = chsolve(h,j)
                Σx0  = ΣxJD@ΣxJD.T
            except LinAlgError:
                bad.append(mx)
                continue
            cxy = covariance_crosshairs(
                Σx0,
                p=pct/100,
                draw_cross=False).T
            if np.max(norm(cxy,axis=1))<localization_radius/L:
                # Peak is acceptably localized
                ellipses.extend(cxy + mx)
                ellipses += [(nan,)*2]
                gaussians.append((mx,Σx0))
                retained_peaks.append(mx)
            else:
                # Peak is not acceptably localized
                bad.append(mx)
        retained_peaks = array(retained_peaks)
        ellipses = array(ellipses).T

        if doplot:
            if draw_border:
                plot(*data.arena.perimeter.T,lw=3,color='k')
                plot(*data.arena.perimeter.T,lw=1,color='w')
            if plt.rcParams['text.usetex']:
                title('Ellipses: %d'%(pct)+
                      r'\% confidence',pad=0)
            else:
                title('Ellipses: %d'%(pct)+
                      r'% confidence',pad=0)
            plot(*ellipses,
                 label='Quadratic',
                 **{**{'lw':.6,'color':MAUVE},**kwargs})
            if len(bad):
                scatter(*array(bad).T,color=RUST,
                        marker='x',lw=.5,s=6,zorder=6)
            axis('off')
            axis('square')
            ylim(1,0)

        if not len(gaussians):
            raise UserWarning((
                'No peaks were localized within '
                'localization_radius = %f')
                %localization_radius)
            
        self.peaks     = retained_peaks
        self.ellipses  = ellipses
        self.gaussians = gaussians
        self.bad       = bad


class QuadraticConfidenceJoint:
    '''
    Similar to QuadraticConfidence, but modified to 
    return the joint covariance to address correlations 
    between fields in the posterior introduced by the 
    GP prior. 

    Parameters
    ----------
    result: LGCPResult

    Other Parameters
    ----------------
    radius: float
        Region in which a peak must be a local maximum 
        to count as a grid field. This is in units of 
        "bins" on the L×L grid.
        **fraction of the grid-cell period ``model.P``**.
    localization_radius: float
        Drop peaks with confidence outside this radius.
        **fraction of the grid-cell period ``model.P``**.
    height_threshold: float in (0,1), default:.8
        Inclusion threshold for peak height in each 
        sample  and probability peak over all samples, 
    pct: float in (0,100), default:95
        percentile to use for confidence bounds
    edge_radius: float; Default 0
        Remove peaks closer than ``edge_radius`` bins
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
    doplot: boolean, default:False. 
        Plot resulting the grid map? 
    **kwargs:
        Forwarded to plot()

    Returns
    -------
    (rx,ry):
        Field locations
    ellipses: 
        NaN-delimeted x,y coordinates for plotting 
        confidence ellipses; 
    gaussians: list
        (μ,Σ) 2D confidence Gaussians for all peaks.
    ok: list<int>
        Which of the ``(rx,ry)`` points were included 
        in Σx0
    Σx0: 
        Joint covariance of all included points, packed
        as ``(x,y)``
    '''
    def __init__(self,
        data,
        model = None,
        fit   = None,
        radius              = None,
        localization_radius = None,
        edge_radius         = 0,
        height_threshold    = 0.8,
        pct                 = 95,
        regularization      = 1e-5,
        doplot              = False,
        **kwargs):
        
        if isinstance(data,LGCPResult):
            model = data.model
            fit   = data.fit
            data  = data.data
        
        _,_,_,L,R,use2d,F,_,h2e,_ = model.cached
        P = model.P
        radius = float(radius)*P
        localization_radius = float(localization_radius)*P

        # Retrieve the posterior mode (in low-D space) and 
        # variance (in regular space)
        # Sample from the GP posterior (see posterior.py)
        posterior_mean,v = fit[:2]
        μ ,λ = model.rates_from_lowrank(posterior_mean,v)
        μ    = μ.reshape(L,L)

        # Get peaks in arena
        peaks = interpolate_peaks(μ,
            clearance_radius = int(round(radius)),
            height_threshold = percentile(
                μ,100*height_threshold))[:2]
        peaks = peaks[:,is_in_hull(peaks.T,data.arena.hull)]

        # Get negative of 2×2 Hessians at all points
        pidx = (*int64(peaks[::-1]*L+0.5),)
        H    = -hessian_2D(μ)[pidx]

        # Obtain posterior derivatives
        # - Get low-rank Cholesky factor of covariance
        # - Get discrete derivatives in low-D Hartley space
        # - Get low-rank derivatives (left-multiply)
        # - Project back to spatial domain only at peaks
        #   by keeping only those columns in the inverse 
        #   transform model.h2e that correspond to a peak 
        #   location.
        Q   = model.low_rank_cholesky(posterior_mean,v).T
        dx  = dx_op(L)
        fx  = fft2(dx  ,norm='ortho')[use2d]
        fy  = fft2(dx.T,norm='ortho')[use2d]
        fQ  = h2f_2d_truncated(Q,L,use2d)
        dxQ = RI((fx.T*fQ.T).T)
        dyQ = RI((fy.T*fQ.T).T)

        # At this point, dxQ and dyQ are the x and y 
        # derivatives applied on the left to the Cholesky 
        # factor of the posterior covariance. Together, they
        # describe the derivatives of the posterior 
        # covariance at all locations. In lgcp2d, h2e is 
        # defined as the R×L² matrix of 2D fourier 
        # components. Cutting out only the [pidx] locations 
        # from this operator corresponds to inverting the 
        # Hartley transform only at the locations where
        # peaks exist (saves time). 
        ispk = h2e.reshape(R,L,L).T[pidx]
        J    = array([ispk@dxQ,ispk@dyQ]).transpose(1,0,2)

        # Calculate covariances and crosshairs 
        # - Calculated expected peak shift covariance Σx0
        # - If peak is localized, plot 95% ellipse
        # - Collect all ellipse to plot at once (faster)
        bad = []
        for ii,(mx,h,j) in enumerate(zip(peaks.T,H,J)):
            try:
                ΣxJD = chsolve(h,j)
                Σx0  = ΣxJD@ΣxJD.T
            except LinAlgError:
                bad.append(ii)
                continue
            # Use if peak is acceptably localized
            cxy  = covariance_crosshairs(
                Σx0,
                p=pct/100,
                draw_cross=False).T
            if np.max(norm(cxy,axis=1))>localization_radius/L:
                bad.append(ii)

        # Reapeat on the good fields jointly
        nfields,_,R = J.shape
        ok    = int32(sorted([*({*arange(nfields)}-{*bad})]))
        nkeep = len(ok)
        JJ    = J[ok].transpose(1,0,2).reshape(2*nkeep,R)
        HH    = scipy.linalg.block_diag(*H[ok])
        ΣxJD  = chsolve(float64(HH),float64(JJ))
        Σx0   = ΣxJD@ΣxJD.T
        Σx0  += regularization*eye(Σx0.shape[0])
        
        # Overwrite with updated confidence intervals
        gaussians = [(peaks[:,iok],Σx0[i*2:i*2+2,i*2:i*2+2]) 
            for i,iok in enumerate(ok)]
        
        ellipses = []
        for mx,S in gaussians:
            try:
                cxy  = covariance_crosshairs(
                    S,p=pct/100,draw_cross=False).T
                ellipses.extend(cxy + mx)
                ellipses += [(nan,)*2]
            except:
                pass
        ellipses = array(ellipses).T

        if doplot:
            plot(*data.arena.perimeter.T,lw=3,color='k')
            plot(*data.arena.perimeter.T,lw=1,color='w')
            if plt.rcParams['text.usetex']:
                title('Ellipses: %d'%(pct)+\
                      r'\% confidence',pad=0)
            else:
                title('Ellipses: %d'%(pct)+\
                      r'% confidence',pad=0)
            plot(*ellipses,
                 label='Quadratic',
                 **{**{'lw':.6,'color':MAUVE},**kwargs})
            if len(bad):
                scatter(
                    *peaks[:,bad],
                    color=BLACK,
                    marker='x',
                    lw=.5,s=3,zorder=6)
            axis('off')
            axis('square')
            ylim(1,0)
            '''
            figure(figsize=(10,10),dpi=120)
            imshow(Σx0,vmin=-.0001,vmax=.0001,
                extent=(0.5,nkeep+0.5)*2)
            xticks(arange(nkeep)+1,map(str,ok));
            yticks(arange(nkeep)+1,map(str,ok));
            for i in range(1,nkeep):
                axvline(i+0.5,color='w',lw=0.3)
                axhline(i+0.5,color='w',lw=0.3)
            '''
            
        self.ellipses=ellipses
        self.gaussians=gaussians
        self.ok=ok
        self.joint_covariance=Σx0
        self.peaks=peaks


class QuadraticExponentialConfidence:
    '''
    Locally-quadratic approximation based on the 
    posterior firing-rate moments. 

    Attributes
    ----------
    ellipses: list
        List of x,y coordinates for plotting confidence 
        ellipses; NaN delimeted.
    gaussians: list 
        List of (μ,Σ) 2D confidence Gaussians for 
        all peaks.

    Parameters
    ----------
    result: LGCPResult

    Other Parameters
    ----------------
    radius: float; Default 0.45
        Region in which a peak must be a local maximum 
        to count as a grid field. 
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    localization_radius: float; Default 0.8
        Drop peaks with confidence outside this radius.
        Set to ``inf`` to retain all peaks. 
        This is in units of 
        **fraction of the grid-cell period ``model.P``**.
    edge_radius: float; Default 0
        Remove peaks closer than ``edge_radius`` bins
        to the arena's edge. The default value of 0
        does not remove points within a margin of
        the boundary.
    height_threshold: float in (0,1); default 0.5
        Inclusion threshold for peak height in each 
        sample and probability peak over all samples,
    prpeak_threshold: float
        Probability threshold for including a peak,
        iinterpolate_peaksn [0,1], based on the 
        fractions of samples that contain a given peak.
    pct: float in (0,100); default 95.0
        percentile to use for confidence bounds,
    doplot: boolean, default True
        Render a plot of the grid map? Default:False. 
    draw_border: boolean, default True
        Whether to render the arena border if plotting.
    '''
    def __init__(
        self,
        data,
        model = None,
        fit   = None,
        radius              = 0.45,
        localization_radius = 0.8,
        edge_radius         = 0,
        height_threshold    = 0.5,
        pct                 = 95,
        doplot              = False,
        draw_border         = True,
        **kwargs):
        
        if isinstance(data,LGCPResult):
            model = data.model
            fit   = data.fit
            data  = data.data
        
        _,_,_,L,R,use2d,F,_,h2e,_ = model.cached
        P = model.P
        radius = float(radius)*P
        localization_radius = float(localization_radius)*P

        # Retrieve the posterior mode (in low-D space) and 
        # variance (in regular space)
        # Sample from the GP posterior (see posterior.py)
        posterior_mean,v = fit[:2]
        μ ,λ = model.rates_from_lowrank(posterior_mean,v)
        μ    = μ.reshape(L,L)
        λ    = λ.reshape(L,L)
        
        # Get peaks in arena
        peaks = interpolate_peaks(λ,
            clearance_radius = int(round(radius)),
            height_threshold =
            percentile(μ,100*height_threshold))[:2]
        peaks = peaks[:,is_in_hull(peaks.T,data.arena.hull)]
        if edge_radius>0.:
            peaks = data.arena.remove_near_boundary(
                peaks, 
                edge_radius)
        
        # Get negative of 2×2 Hessians at all points
        pidx = (*int32(peaks[::-1]*L+0.5),)
        H    = -hessian_2D(λ)[pidx]
        

        # Get post.r mean Hessian and covariance gradients
        # - Get low-rank Cholesky factor of covariance
        # - Construct discrete d/dx in low-D Hartley space
        # - Obtain low-rank derivatives (left-multiply)
        # - Project back to spatial domain only at peaks
        #   by keeping only those columns in the inverse  
        #   transform model.h2e that correspond to a peak 
        #   location.
        Q   = model.low_rank_cholesky(posterior_mean,v).T
        dx  = dx_op(L)
        fx  = fft2(dx  ,norm='ortho')[use2d]
        fy  = fft2(dx.T,norm='ortho')[use2d]
        fQ  = h2f_2d_truncated(Q,L,use2d)
        dxQ = RI((fx.T*fQ.T).T)
        dyQ = RI((fy.T*fQ.T).T)

        # At this point, dxQ and dyQ are the x and y
        # derivatives applied on the left to the Cholesky
        # factor of the posterior covariance. Together, they
        # describe the derivatives of the posterior 
        # covariance at all locations. In lgcp2d, h2e is 
        # defined as the R×L² matrix of 2D fourier
        # components. Cutting out only the [pidx] locations
        # from this operator corresponds to inverting the
        # Hartley transform only at the locations where
        # peaks exist (saves time). 
        ispk = h2e.reshape(R,L,L).T[pidx]
        J    = np.float32([ispk@dxQ,ispk@dyQ]).transpose(1,0,2)

        # We will need the posterior second moment of λ
        # This is λ²exp(v)
        M2 = (λ**2 * exp(v.reshape(L,L)))[pidx]
        
        # Calculate covariances, prepare crosshairs 
        # - Get the expected peak shift covariance Σx0
        # - If peak is localized, plot confidence ellipse
        # - Collect all ellipse to plot at once (faster)
        ellipses,gaussians,bad,good = [],[],[],[]
        retained_peaks = []
        for (mx,h,j,m2) in zip(peaks.T,H,J,M2):
            try:
                ΣxJD = chsolve(h,j)
                Σx0  = (ΣxJD@ΣxJD.T)*m2
            except LinAlgError:
                bad.append(mx)
                continue
            cxy = covariance_crosshairs(
                Σx0,
                p=pct/100,
                draw_cross=False).T
            if np.max(norm(cxy,axis=1))<localization_radius/L:
                # Peak is acceptably localized
                ellipses.extend(cxy + mx)
                ellipses += [(nan,)*2]
                good.append(mx)
                gaussians.append((mx,Σx0))
                retained_peaks.append(mx)
            else:
                # Peak is not acceptably localized
                bad.append(mx)
        retained_peaks = array(retained_peaks)
        ellipses = array(ellipses).T

        if doplot:
            if draw_border:
                plot(*data.arena.perimeter.T,lw=3,color='k')
                plot(*data.arena.perimeter.T,lw=1,color='w')
            if plt.rcParams['text.usetex']:
                title('Ellipses: %d'%(pct)+
                      r'\% confidence',pad=0)
            else:
                title('Ellipses: %d'%(pct)+
                      r'% confidence',pad=0)
            plot(*ellipses,
                 label='Quadratic',
                 **{**{'lw':.6,'color':MAUVE},**kwargs})
            if len(bad):
                scatter(*array(bad).T,color=RUST,
                        marker='x',lw=.5,s=6,zorder=6)
            axis('off')
            axis('square')
            ylim(1,0)

        if not len(gaussians):
            raise UserWarning((
                'No peaks were localized within '
                'localization_radius = %f')
                %localization_radius)
            
        self.peaks     = retained_peaks
        self.ellipses  = ellipses
        self.gaussians = gaussians

        
def fraction_within_arena(data, gaussians, N=1000):
    '''
    This function is depricated. Testing whether 
    peaks are close to the edge is now handled by
    ``load_data.Arena.remove_near_boundary()``
    
    For a list of 2D Gaussians given as (μ,Σ) tuples, 
    estimate the fraction of probability mass within 
    data.arena.hull. 
    
    This uses fairly crude numerical integration; It is
    sufficiently accurate for testing whether fields
    lie too close to the edge of the arena.
    
    Parameters
    ----------
    data: Dataset
    gaussians list
        List of (μ,Σ) 2D Gaussians
    
    Returns
    -------
    np.float32:
        Fraction of each Gaussian within the dataset's
        arena's convex hull. 
    '''
    zg = np.random.randn(2,200)
    results = []
    for μ,Σ in gaussians:
        e,v = scipy.linalg.eigh(Σ)
        zt  = v.T.dot(zg*(e**0.5)[:,None])+μ[:,None]
        results.append(mean(is_in_hull(zt,data.arena.hull)))
    return float32(results)
