function result = makegridcellkernel(L,P,kv,opt)
% MAKEGRIDCELLKERNEL  Initialize a low-rank spatial frequency space kernel
% for gric cells. 
%   result = MAKEGRIDCELLKERNEL(L,P,V,OPT) Will prepare a periodic
%   grid-cell Gaussian-process kernel for an L×L size grid, with period P
%   (in pixels) and prior marginal variance V. 
% 
%   OPT should be a struct specifying the following options:
%    - |opt.component_threshold_percent|: Threshold for retaining a spatial
%    frequency component in the low-rank subspace. The default is 1%,
%    corresponding to discarding all components with eigenvalues less than
%    1% of the largest eigenvalue (variance of direction of largest
%    variance) in the kernel. 
%    - |opt.dc|: DC (constant) variance of the prior kernel. This
%    determines how much we penalize the posterior mean log-rate to remain 
%    close to the mean-log-rate specified by the prior. I tend to set this
%    to a large value, to leave this direction largely unconstrained. It
%    defaults to |1e3|.
%    - |opt.keep|: KEEP is a L×W boolean array, with each entry 
%    corresponding to a 2D spatial Fourier coefficient as returned by 
%    MATLAB's FFT2() function. If |opt.keep| is specified,
%    |opt.component_threshold_percent| is ignored, and all components set
%    to |1| in |opt.keep| are retained instead. 
%   
%   This function will return a |RESULT| struct with the following
%   fields/members:
%    - |result.kern|: The 2D covariance kernel in the spatial domain.
%    - |result.Kf|: The covariance kernel in the 2D spatial-frequency
%    domain, BEFORE discarding components. 
%    - |result.keep|: A boolean array indicating which frequency components
%    were retained (see |opt.keep|). 
%    - |result.K|: The convariance in the 2D spatial-frequency domain,
%    AFTER discarding unused components (removed components are set to
%    zero). 
%    - |result.R|: The total number of components retained
%    - |result.window|: The windowing function used to reduce ringing on
%    finite domains. 
%    - |result.clip|: The variance threshold used to clip and remove
%    low-variance components in the kernel. 
%    - |result.V|: A saved copy of the parameter V passed to this function.
%    - |result.P|: A saved copy of the parameter P passed to this function.


if nargin<4
    opt.matlabs_syntax_is_woefully_deficient = true;
end

if ~isfield(opt,'dc')
    opt.dc = 1e3; % DC kernel variance; Something large. 
end

if ~isfield(opt,'component_threshold_percent')
    % 1% keeps a lot of components, but gives accuracy enough to compare
    % the ELBO loss function across different periods
    % 10% is more than accurate enough for qualitative insignts, but 
    % can lead to jittery / non-smooth loos functions as you change the grid
    % period. 
    opt.component_threshold_percent = 1.0;
end

if ~isfield(opt,'keep')
    keep = 0;
else
    keep = opt.keep;
end


% The radial average of an ideal grid is the zeroith order bessel function.
[cx,cy] = meshgrid( ((0:L-1)-floor(L*.5)).*(2*pi/P) );
cr      = hypot(cx,cy);
kern    = besselj(0,fftshift(cr));

% Spatial windowing of the kernel reduces ringing in the frequency domain.
window = hanning(L);
window = fftshift(window);
window = window'*window;
kern   = kern.*window;

% Limit the kernel to nearest neighbor interactions
thezeros = besselzero(0,3);
cutoff   = thezeros(end);
clip     = fftshift(cr<cutoff);
kern     = kern.*clip;

% Attenuate high frequencies
kern = blur2d(kern,P/pi);

% Normalize
kern = kern ./ max(kern(:));

% Switch to Fourier spectrum
Kf = max(0.0,real(fft2(kern)));
eigvals = abs(Kf);

% Identify small-variance spatial frequency components
if keep==0
    seigvals = sort(eigvals(:));
    thr  = seigvals(end-1)*(opt.component_threshold_percent/100);
    keep = abs(Kf)>thr;
    keep = keep | keep';
end

% Repair too-small eigenvalues
emin = 1e-5.*max(eigvals(:));
zero = eigvals<emin;
Kf(zero) = emin;
kern = real(ifft2(Kf));

% Scale prior variance
Kf = Kf .* kv;

% Create truncated representation
K  = Kf(keep);

% Let the DC component large variance (we don't constrain the DC mean)
Kf(1,1) = Kf(1,1) + opt.dc;

%kern = real(ifft2(Kf.*keep));

% R is the dimensionality of our low-rank space
R = sum(keep(:));

% Build return struct
result.kern = kern;
result.Kf   = Kf;
result.keep = keep;
result.K    = K;
result.R    = R; 
result.window = window;
result.clip = clip;
result.V    = kv;
result.P    = P;
end



function w = hanning(M)
    % Hanning window (in case signal processing toolbox not installed)
    % This matches numpy.hanning and not Matlab's hanning()
    n = 1-M:2:M;
    w = 0.5 + 0.5*cos(pi*n/(M-1));
end

