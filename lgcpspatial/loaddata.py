#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Routines for loading and preparing Krupic lab datasets

Resolution for the camera used to record R1 is 350px/m;
resolution for other rats (R11 and R18) was 338px/m
Fs     = 50.0      # Sample rate of data (samples/second)
"""

import os
import warnings
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from scipy.io   import loadmat

from lgcpspatial.plot       import *
from lgcpspatial.util       import *
from lgcpspatial.estimators import kde

# Used to find peaks and truncate radial autocorrelation
from scipy.special import jn_zeros

# Used to get subset of 2D space containing the arena
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def patch_position_data(px,py,delta_threshold=0.01):
    '''
    Linearly interpolate across noise/glitches in the 
    position tracking.
    
    Parameters
    ----------
    px: np.float32
    '''
    pz = px + 1j*py
    bad = where(abs(diff(pz))>delta_threshold)[0]
    pz[bad]=NaN
    z = interpolate_NaN(pz)
    assert all(isfinite(z))
    return z.real,z.imag


def bin_spikes(
    px,
    py,
    s,
    L,
    w=None,
    method='linear'
    ):
    '''
    Bin spikes, using linear interpolation to distribute 
    point mass to four nearest pixels, weighted by distance. 
    
    Parameters
    ----------
    px: np.flaot32
        ``x`` location of points
    py: np.float32
        ``y`` location of points
    s: np.ndarray
        Spike count at each point
    L: int
        Number of spatial bins for the LxL grid
    w: np.float32
        Weights to apply to each point 
        (default is None for no weigting)
    method: str
        Method to use; Can be ``"linear"`` or ``"nearest"``.
    '''
    
    if method=='nearest':
        # Bin spike counts simple version
        N  = np.histogram2d(y,x,(bins,bins),density=0,weights=w)[0]
        ws = s if w is None else np.array(s)*np.array(w)
        K  = np.histogram2d(y,x,(bins,bins),density=0,weights=ws)[0]
        return N,K
    if method=='linear':
        if w is None: w = ones(len(px))
        # The 2D coordinates (px,py) should be normalized to
        # lie with the unit square [0,1)²
        assert np.max(px)<1 and np.max(py)<1 \
           and np.min(px)>=0 and np.min(py)>=0
        # The integer parts of the coordinate tells us the 
        # top-right bin location of the 2×2 neighborhood. 
        # The fractional parts will tell us how to 
        # distribute the point mass within this 
        # neighborhood.
        ix,fx = divmod(px*L,1)
        iy,fy = divmod(py*L,1)
        assert np.max(ix)<L and np.max(iy)<L
        w11 = fx*fy*w
        w10 = fx*(1-fy)*w
        w01 = (1-fx)*fy*w
        w00 = (1-fx)*(1-fy)*w
        qx  = np.concatenate([ix,ix+1,ix,ix+1])
        qy  = np.concatenate([iy,iy,iy+1,iy+1])
        z   = np.concatenate([w00,w10,w01,w11])
        ibins = np.arange(L+1)
        N   = np.histogram2d(qy,qx,(ibins,ibins),
                          density=False,weights=z)[0]
        ws  = z*np.concatenate((s,)*4)
        K   = np.histogram2d(qy,qx,(ibins,ibins),
                          density=0,weights=ws)[0]
        return np.float32(N),np.float32(K)
    raise ValueError(
        '`method` should be "linear" or "nearest"'

class Arena:
    '''
    Object describing the area visited by an animal within 
    an experiment.
    
    Attributes
    ----------
    mask: np.bool
        L × L boolean matrices of grid locations within the 
        experimetnal arena
    nanmask: np.float32
        L × L float32 array with 1 for in-bounds points and 
        NaN for out-of-bounds points. Multiple array data by
        this mask before 
        sending to ``imshow()`` to render out-of-bounds points
        as transparent.
    perimeter: np.float32
        NPOINTS × 2 list of points in the convex hull of the 
        experimental arena
        in [0,1]² normalied coordinates        
    hull: scipy.spatial._qhull.ConvexHull
        Convex Hull object describing the arena perimeter
    '''
    def __init__(self,*args,**kwargs):
        if '__from_mask' in kwargs:
            Arena._init_from_mask(self,*args,**kwargs)
        elif '__from_perimeter' in kwargs:
            Arena._init_from_perimeter(self,*args,**kwargs)
        else:
            Arena._init_from_points(self,*args,**kwargs)
    
    def _init_from_points(self,px,py,L,
                 resolution=1,
                 radius=1.5,
                 thr=0.1):
        '''
        Prepare a mask from the convex hull of visited
        locations

        Parameters
        ----------
        px: np.float32
            NPOINTS ndarray
            X coordinates of points in [0,1]
        py: np.float32
            NPOINTS ndarray
            Y coordinates of points in [01]
        L: int
            positive integer
            Number of spatial bins (e.g. 128)

        Other Parameters
        ----------------
        resolution: positive integer
            Upsampling factor
            Defaults to 1
        radius: positive float
            Additional padding to add to mask outside
            perimeter in pixels. This defines the blur
            radius for extending the boundary.
        thr: positive float: 
            Thresholding to use after blur to extend
            the boundary.

        Returns
        -------
        arena: Arena
            An Arena object containing a LxL boolean array 
            ``mask``, a NPOINTSx2 array of perimeter points
            ``perimenter``, and a 
            ``scipy.spatial._qhull.ConvexHull`` object ``hull``; 
        '''
        
        # Encircle points in a convex hull
        points = array([px,py]).T
        hull   = ConvexHull(points)
        verts  = concatenate([hull.vertices,hull.vertices[:1]])
        perim  = points[verts]
        
        # Generate discrete grid at desired resolution
        Lr     = L*resolution
        grdp   = linspace(0,1,Lr+1+Lr)[1::2]
        grdp   = array([
            grdp[None,:]*ones((Lr,Lr)),
            grdp[:,None]*ones((Lr,Lr))])
        grdp  = grdp.reshape(2,(Lr)**2).T
        
        # Find points within this grid inside convex hull
        mask  = is_in_hull(grdp,hull).reshape(Lr,Lr)
        
        # Extend the mask to add a margin
        mask  = blur(mask,radius)>thr
        
        # Recalculate hull to include extended mask
        py,px = array(where(mask))/(Lr)
        points= array([px,py]).T
        hull  = ConvexHull(points)
        verts = concatenate(
            [hull.vertices,hull.vertices[:1]])
        
        self.mask = mask
        self.perimeter = points[verts]
        self.hull = hull
        self.nanmask = float32(mask)
        self.nanmask[self.nanmask<1] = NaN
    
    def _init_from_mask(self,mask,resolution=None,radius=None,**kwargs):
        
        self.mask = mask
        L = mask.shape[0]
        
        # Recalculate hull to include extended mask
        py,px = np.array(np.where(mask))/L
        points= np.array([px,py]).T
        hull  = ConvexHull(points)
        verts = np.concatenate(
            [hull.vertices,hull.vertices[:1]])
        
        self.mask = mask
        self.perimeter = points[verts]
        self.hull = hull
        self.nanmask = float32(mask)
        self.nanmask[self.nanmask<1] = NaN
        
        if not resolution is None:
            # Generate discrete grid at desired resolution
            Lr   = L*resolution
            grdp = linspace(0,1,Lr+1+Lr)[1::2]
            grdp = array([
                grdp[None,:]*ones((Lr,Lr)),
                grdp[:,None]*ones((Lr,Lr))])
            grdp  = grdp.reshape(2,(Lr)**2).T

            # Find points within this grid inside convex hull
            mask  = is_in_hull(grdp,hull).reshape(Lr,Lr)

            if not radius is None:
                # Extend the mask to add a margin
                mask  = blur(mask,radius)>thr

            # Recalculate hull to include extended mask
            py,px = array(where(mask))/(Lr)
            points= array([px,py]).T
            hull  = ConvexHull(points)
            verts = concatenate(
                [hull.vertices,hull.vertices[:1]])

            self.mask = mask
            self.perimeter = points[verts]
            self.hull      = hull
            self.nanmask   = float32(mask)
            self.nanmask[self.nanmask<1] = NaN
    
    def from_mask(mask,**kwargs):
        return Arena(mask,__from_mask=True,**kwargs)
        
    def distance_to_boundary(self, peaks):
        '''
        Measure the distance of a point within this
        Arena's convex hull, and the boundary (the
        convex hull). 
        
        This is calculated in a fairly crude way: 
        brute force search for the closest out of
        bounds point on the L × L grid. 

        Parameters
        ----------
        peaks: 2 × NPOINTS np.float32  
            (x,y) positions of points to test.
        
        Returns
        -------
        is_close: Length NPOINTS 1D np.bool
            Boolean array indicating 
        
        '''
        peaks = np.float32(peaks)
        
        if np.any(peaks<0) or np.any(peaks>1):
            raise ValueError(
                'Expected peaks to be 2×NPOINTS np.float32 '
                'array of (x,y) points in normalized [0,1]²'
                'coordinates; Got values outside [0,1]².')
        L  = self.mask.shape[0]
        outside = (zgrid(L)/L+.5+.5j)[~self.mask]
        zpeaks  = [1,1j]@peaks
        D  = abs(zpeaks[:,None]-outside[None,:])
        D  = np.min(D,1)
        return D
    
    def close_to_boundary(self, peaks, radius):
        '''
        Detect peaks within distance ``radius`` of the 
        Arena boundary. 

        Parameters
        ----------
        peaks: 2 × NPOINTS np.float32  
            (x,y) positions of peaks to trim 
            in normalized [0,1]² coordinates.
        radius: float
            Distance from edge *in pixels*.

        Returns
        -------
        is_close: Length NPOINTS 1D np.bool
            Boolean array indicating 
        '''
        L  = self.mask.shape[0]
        return self.distance_to_boundary(peaks)<radius/L
    
    def remove_near_boundary(self, peaks, radius):
        '''
        Delete peaks that are too close to the edgs of
        an experimental arena.

        Parameters
        ----------
        peaks: 2 × NPOINTS np.float32  
            (x,y) positions of peaks to trim 
            in normalized [0,1]² coordinates.
        radius: float
            Distance from edge to trim
            in pixels

        Returns
        -------
        peaks: 2 × NPOINTS np.float32  
            (x,y) positions of peaks further than ``radius``
            pixels from the boundary.      
        '''
        ok = ~self.close_to_boundary(peaks,radius)
        return peaks[:,ok]
    
    def contains(self,P):
        return lgcpspatial.util.is_in_hull(P,self.hull)
        
class Dataset:
    '''
    Class to load and extract experimental data
    from Krupic Lab data format.     
    
    Attributes
    --------------------------------------------------------
    spikes: np.float32
        Length NSAMPLES array of spike counts at each time 
        point.
    extent: tuple
        (xstart,xstop,ystart,ystop) coordinates of 2D arena 
        in meters. Pass this as the ``extent`` argument to 
        ``imshow()`` to correctly position L × L grid in 
        physical coordinates.  
    scale: float, units of 1/meters 
        Scaling factor between (px,py)∈[0,1]²
        coordinates and meters. 
        px/scale = position in meters 
    px: np.float32
        Normalized x-position in [0,1]
    py: np.float32
        Normalized y-position in [0,1]
    margin: float
        Margin padding argument passed to constructor. 
        Fraction of empty space around binned data to avoid 
        circular FFT wraparound. Ensure your prior kernel
        width is smaller than margin*L/2.
    '''
    def __init__(self,x,y,spikes,margin=.1):
        '''
        Parameters
        ----------------------------------------------------
        x: np.float32
            1D array of animal's x position.
        y: np.float32
            1D array of animal's y position.
        spikes: np.float32
            1D array of binned spike counts at the same
            sample rate as (x,y). 
        '''
        x = float32(x).ravel()
        y = float32(y).ravel()
        spikes = float32(spikes).ravel()
        if not len(x)==len(y)==len(spikes):
            raise ValueError('x,y, and spikes should be'
                             'float32 arrays with the same'
                             'length,')
        
        # Define normalized coordinates to bin data to grid
        minx,maxx = nanmin(x),nanmax(x)
        miny,maxy = nanmin(y),nanmax(y)
        midx,midy = (minx+maxx)/2.0, (miny+maxy)/2.0;
        delta = nanmax([maxx-minx,maxy-miny])
        pad   = .5 + margin
        edge  = delta*pad
        scale = (1-1e-6)/(edge*2)
        px    = (x-midx+edge)*scale
        py    = (y-midy+edge)*scale
        x0    = midx-edge
        y0    = midy-edge
        x1    = 1/scale+x0
        y1    = 1/scale+y0

        self.extent = (x0,x1,y0,y1)
        self.px=px
        self.py=py
        self.margin=margin
        self.spikes=spikes
        self.scale=scale
    
    def from_file(dataset,margin=.1):
        '''
        Load a dataset, rescaling the spatial locations.
        
        We correct the head direction angles to align
        with the usual definitions of polar angles
        for (x,y) position data. 

        Parameters
        ----------------------------------------------------
        dataset: str
            Which dataset file to load
        margin: float
            Pading for circular convolution; default .1

        Returns
        ----------------------------------------------------
        Dataset: 
            A Dataset object prepared from the given 
            grid-cell data file. 
        '''
        # Retrieve data from file
        data    = loadmat(dataset,squeeze_me=True)
        dataset = dataset
        
        for varname in ('xy dir pos_sample_rate '
                        'pixels_per_m spikes_times '
                        'spk_sample_rate').split():
            if not varname in data:
                raise ValueError('Expected %s to contain a'
                    ' variable named %s'%(dataset,varname))
        
        xy_position_px       = data['xy']
        head_direction_deg   = data['dir']
        position_sample_rate = data['pos_sample_rate']
        px_per_meter         = data['pixels_per_m']
        spike_times_samples  = data['spikes_times']
        spike_sample_rate    = data['spk_sample_rate']
        
        # Fix head direction to line up with px,py
        # for usual definitions of Cartesian and polar
        # coordinates
        head_direction_deg = (-head_direction_deg + 180)%360
        
        if len(spike_times_samples)==0:
            warnings.warn('The `spikes_times` variable '
                'for file %s appears to be empty.'%dataset)

        # Convert units
        dt                  = 1 / position_sample_rate
        xy_position_meters  = xy_position_px / px_per_meter 
        spike_times_seconds = spike_times_samples/ spike_sample_rate
        NSPIKES             = len(spike_times_samples)
        NSAMPLES            = len(head_direction_deg)

        # Downspample spikes to position sampling rate
        it,ft  = divmod(spike_times_seconds/dt,1)
        w      = ones(NSPIKES)
        wt     = concatenate([1-ft,ft])
        qt     = concatenate([it,it+1])
        spikes = float32(histogram(qt,
           arange(NSAMPLES+1),density=0,weights=wt)[0])
        
        # Repair defects in position tracking
        px,py = xy_position_meters.T
        zx,zy = patch_position_data(px,py,
                                    delta_threshold=dt)
        
        # Build and return Dataset object
        data = Dataset(zx,zy,spikes,margin=margin)
        data.dataset              = dataset
        data.xy_position_px       = xy_position_px
        data.head_direction_deg   = head_direction_deg
        data.position_sample_rate = position_sample_rate
        data.px_per_meter         = px_per_meter
        data.spike_times_samples  = spike_times_samples
        data.spike_sample_rate    = spike_sample_rate
        data.dt                   = dt
        data.xy_position_meters   = xy_position_meters
        data.spike_times_seconds  = spike_times_seconds
        data.NSPIKES              = NSPIKES
        data.NSAMPLES             = NSAMPLES
        return data

    def prepare(
        self,
        L,
        P=None,
        spike_weights=None,
        blur_radius=2,
        doplot=False,
        arena_only=False
        ):
        '''
        Prepare dataset for further inference.
        This function bins spikes and position data, and 
        heuristically estimates grid scale. It adds the 
        following attributes to the Dataset object:

        Attributes
        ----------
        L: int
            Size of L × L spatial grid for binned data.
            I've found that 128² grids work well and are a
            good compromise between speed, resolution, and 
            numerical stability.
        arena: Arena
            Object describing binned 2D experimental arena 
            shape. See the documentation of the Arena class 
            for more details. 
        n: np.float32
            Length L² 1D array of total visits to each 
            location. This is interpolated, so may contain 
            non-integer values.
        y: np.float32
            Length L² 1D array of total spikes at each 
            location. This is interpolated, so may contain 
            non-integer values.
        P: float
            Heuristic grid period from autocorrelogram
        prior_variance: float
            Heuristic estimate of Gaussian Process prior 
            kernel marginal variance. If this is bigger, the 
            prior will be weighted less during Baysian
            inference. This heuristic estimate is biased;
            you should optimize the kernel hyperparameters
            using the ELBO if you want the posterior
            variance to be interpretable. See
            ``grid_search.py``. 
        bg_blur_radius: float
            Low-frequency spatial scale below which rate
            variations are considered background
            fluctuations and removed. 
        kde_blur_radius:
            Grid-period scale; Use this as the standard
            deviation of a Gaussian blur for a Kernel
            Density Estimator (KDE) of firing rate as a
            function of location.
        prior_mean: np.float32
            L × L array of estimated background log-rate. 
            Use this as a prior during Gaussian Process 
            inference so that the calculated rate maps are
            defined in terms of grid-like structure above
            these background rate variations. This is
            calculated by Gaussian blurring the visits ``n``
            and spike counts ``y`` with a kernel with
            standard-deviation ``bg_blur_radius``, and taking
            the logarithm of their ratio.
        lograte_guess: np.float32
            L × L array of KDE-estimated log-firing rate.
            This is calculated by blurring visits ``n`` and
            spike counts ``y`` with a Gaussian kernel with
            standard deviation ``kde_blur_radius``, taking 
            the logarithm of their ratio, and subtracting
            prior_mean.

        Parameters
        ----------
        fn: str
            File path to load
        L: int
            Number of bins in L × L spatial grid

        Other Parameters
        ----------------
        P: int; default None
            Grid period; If not provided we will estimate
            this from the autocorrelogram,
        blur_radius: float; default 2
            Initial KDE blur radius for heuristic period
            estimate.
        spike_weights: np.float32; default None
            Array of sample weightings of the same length
            as Dataset.spikes to use when binning.
        doplot: boolean; default True
            Whether to show a summary plot
        arena_only: boolean; default False
            Prepare binned position and arena information
            only. Useful if you want to use this code
            for something that is not a grid cell. The
            following attributes will remain absent:
            ``P``, ``bg_blur_radius``, ``prior_mean``, 
            ``kde_blur_radius``, ``lograte_guess``, 
            ``prior_variance``.

        Returns
        -------
        A Dataset object containing spatially-binned data,
        Heurstic estimates of the grid scale, and initial 
        firing-rate maps estimated using a kernel density 
        estimator. See the ``PreparedDataset`` docstring for 
        more detail. 
        '''
        # Bin spikes
        N,K = bin_spikes(self.px,
                         self.py,
                         self.spikes,
                         L,
                         w=spike_weights) 
        zeros    = N<=0.0
        N[zeros] = 1
        Y        = float32(K/N)
        Y[zeros] = 0
        n        = N.ravel()                  # s/bin
        y        = Y.ravel()                  # spikes/s/bin
        self.L=L
        self.n=n
        self.k=K.ravel()
        self.y=y

        # Prepare mask using the convex hull
        arena = Arena(self.px,self.py,L)
        self.arena=arena
        
        # This is helpful
        perimeter_meters = arena.perimeter/self.scale \
            + [self.extent[0],self.extent[2]]
        arena.perimeter_meters = perimeter_meters
        
        # Don't bother
        if arena_only: 
            return self

        # Calibrate grid scale
        λhat   = blur(Y,blur_radius)      # spikes/second
        acorr2 = fft_acorr(λhat,arena.mask) # 2D autocorr.
        acorrR = radial_average(acorr2)   # radial autocorr.
        res    = 50                       # Subsample res.
        if P is None:
            # № bins to 1st peak
            P,acup = acorr_peak(acorrR,res) 
            # Get period from peak assuming bessel Jo
            P  *= 2*np.pi/jn_zeros(1,2)[-1]
            
            if isnan(P):
                warnings.warn(
                    'Could not detect grid period using '
                    'autocorrelogram peaks. Is the period '
                    'too large? Is this a place cell? '
                    'I will try to use the trough instead.'
                )
                P,acup = acorr_trough(acorrR,res) 
                # Get period from peak assuming bessel Jo
                P *= 2*np.pi/jn_zeros(1,1)[-1]
                if isnan(P):
                    warnings.warn(
                        'Heuristic detection of grid-cell '
                        'period using peaks/troughs in the '
                        'autocorrelogram failed; Check '
                        'that this is really a grid cell '
                        'and that the data can resolve '
                        'its periodicity. For high grid '
                        'resolutions (large ``L``) relative'
                        ' to the grid scale, it may also '
                        'be useful to increase the '
                        'argument ``blur_radius`` to '
                        '``Dataset.prepare()`` (the default'
                        ' is 2).'
                    )
        elif doplot:
            _,acup = acorr_peak(acorrR,res)
            
        # Precompute variables
        kde_blur_radius = P/pi              # bins
        bg_blur_radius  = kde_blur_radius*5 # bins
        λhat = kde(N,K,kde_blur_radius)     # KDE rate
        λbg  = kde(N,K,bg_blur_radius )     # Bkgnd rate
        lλh  = slog(λhat)                   # Log rate
        prior_mean    = slog(λbg)           # Log bkgnd
        lograte_guess = lλh - prior_mean    # Fgnd lograte
        # Heuristic guess for prior kernel variance
        prior_variance = var(lograte_guess[arena.mask]) 
        
        if doplot:
            figure(figsize=(6,4.5),dpi=200)
            subplots_adjust(left=0,right=0.95,
                            top=.88,bottom=0.1,
                            wspace=0.1,hspace=0.6)
            ax = {}
            ax[1]=subplot2grid((3,3),(0,0))
            imshow(N,vmin=percentile(N,5),
                   vmax=percentile(N,98));
            title('Visits'); axis('off')
            ax[2]=subplot2grid((3,3),(0,1))
            imshow(y.reshape(L,L),vmin=percentile(y,1),
                   vmax=percentile(y,98));
            title('Activity'); axis('off')
            ax[3]=subplot2grid((3,3),(0,2))
            showim(arena.mask,'Mask');
            ax[4]=subplot2grid((3,3),(1,0))
            imshow(λhat*arena.mask,vmin=0,
                   vmax=percentile(λhat,99));
            title('KDE-smoothed rate'); axis('off')
            ax[5]=subplot2grid((3,3),(1,1))
            showim(lograte_guess,'Foreground log-rate',
                   mask=arena.mask);
            ax[6]=subplot2grid((3,3),(1,2))
            showim(prior_mean,'Background log-rate',
                   mask=arena.mask);
            if hasattr(self,'dataset'): 
                suptitle(self.dataset.split(os.sep)[-1])
            ax[7]=subplot2grid((3,3),(2,0))
            imshow(acorr2,vmax=percentile(acorr2,99.9))
            title('Autocorrelation'); axis('off')
            ax[8]=subplot2grid((3,3),(2,1),colspan=2)
            plot((linspace(-L/2,L/2,L*res)),acup)
            xlim(-L/2,L/2)
            xlabel('Δbins')
            ylabel('spike²/sample²')
            title('Radial Autocorrelation')
            axvline(P,color='k',lw=0.5)
            simpleaxis()
            nudge_axis_left(240)
            nudge_axis_right(35)
            sca(ax[1]); colorbar(label='sample/bin')
            sca(ax[2]); colorbar(label='spike/sample')
            sca(ax[3]); colorbar(label='spike/sample'
                                ).remove()# spacing hack
            sca(ax[4]); colorbar(label='spike/sample')
            sca(ax[5]); colorbar(label='log-rate')
            sca(ax[6]); colorbar(label='log-rate')
            sca(ax[7]); colorbar(label='spike²/sample²')
            figurebox(color='w')

        if not all(isfinite(N)):
            warnings.warn(
                'Some visit counts (N) are not finite')
        if not all(isfinite(K)):
            warnings.warn(
                'Some spike counts (K) are not finite')
        if not all(isfinite(λhat)):
            warnings.warn(
                'Some rates (λhat=K/N) are not finite')
        if not all(isfinite(λbg)):
            warnings.warn(
                'Some background rate (λbg) values '
                'are not finite')
        
        self.P=P
        self.bg_blur_radius  = bg_blur_radius
        self.prior_mean      = prior_mean
        self.kde_blur_radius = kde_blur_radius
        self.lograte_guess   = lograte_guess
        self.prior_variance  = prior_variance
        
        return self
    
    def create_posthoc(
        L,n,y,
        prior_mean=0,
        lograte_guess=None
        ):
        '''
        Bypass normal initialization and Dataset holding 
        only binned positions and rate. 
        This is only used by 
        ``example 0 hyperparameter groundtruth test.ipynb``
        
        The resulting object has the prepared values
        ``(n,y,prior_mean,lograte_guess,L)`` only, 
        and lacks the ``(px,py,spikes)`` raw data fields. 
        '''
        n = float32(n).ravel()
        y = float32(y).ravel()
        assert len(n)==L*L
        assert len(y)==L*L
        data = Dataset.__new__(Dataset)
        data.n = n
        data.y = y
        data.prior_mean    = prior_mean
        data.lograte_guess = lograte_guess
        data.L = L
        return data
    
    def reweighted(self,weights):
        '''
        Re-bin spike histogram using the provided weights
        
        Parameters
        ----------
        weights: 1D np.array
            1D array of spike weights of the same length
            as data.spikes.
            
        Returns
        -------
        Dataset:
            New dataset with spike histogram binned using
            the provided weights.
        '''
        weights = float32(weights).ravel()
        if not weights.shape==self.spikes.shape:
            raise ValueError('list of weights should be '
                'the same length as Dataset.spikes.')
        N,K = map(ravel,bin_spikes(
            self.px,
            self.py,
            self.spikes,
            self.L,weights))
        Y   = float32(K/maximum(1e-9,N))
        Y[N<=1e-9] = 0
        data = Dataset.create_posthoc(
            self.L,N,Y,
            self.prior_mean,
            self.lograte_guess)
        return data
    
    def normalized_to_physical_units(data,p):
        '''
        Convert a list of points from [0,1]² units to
        physical units (meters)

        Parameters
        ----------
        p: np.array
            array of 2D point data

        Returns
        -------
        np.array: 
            2×NPOINTS array of points in physical units
        '''
        p = array(p)
        if np.any(iscomplex(p)):
            p = float32([p.real,p.imag])
        if len(p.shape)!=2:
            raise ValueError('Expected a list of 2D points')
        if p.shape[1]!=2:
            p = p.T
        if p.shape[1]!=2:
            raise ValueError(('Could not figure out which '+
                'direction contains (x,y) points for '+
                'shape %s')%(p.shape,))
        if np.any(p<0) or np.any(p>1):
            raise ValueError('Expected points in [0,1]²; '+
                'point data contained values outside this.')
        p = p/data.scale + [data.extent[0],data.extent[2]]
        return p.T
