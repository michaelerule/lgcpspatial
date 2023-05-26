function result = lgcpregress(n,k,kernel)
%{
LGCPREGRESS  Log-Gaussian Cox Process regression
  RESULT = lgcpregress(N,K,KERNEL) infers log-rates from binned visit
  counts K and spike counts N using kernel KERNEL returned by 
  MAKEGRIDCELLKERNEL(L,P,V), where L is the size of the L×L 2D spatial
  grid, P is the grid-cell period (in units of bins), and V determines the
  overall scaling of the prior variance. 
%}

% Options struct
opt.niter     = 10;        % Maximum coordinate descent iterations 
opt.nmeaniter = 5;         % Maximum NR iterations for mean update
opt.nvariter  = 1;         % Maximum fixed point iters for variance update
opt.tol       = 1e-7;      % Absolute tolerance for model fit
opt.gamma     = 1.0;       % Dispersion correction, (<1 overdispersed)

% Argument validation
assert(isequal(size(n),size(k)));
[W,H] = size(n);
N = W*H;

% Precompute variables
kde_blur_radius = kernel.P/pi;          % bins
bg_blur_radius  = kde_blur_radius*5; % bins
rhat = kderatemap(n,k,kde_blur_radius); % KDE rate
rbg  = kderatemap(n,k,bg_blur_radius ); % Bkgnd rate
lrh  = slog(rhat);                   % Log rate
mu0  = slog(rbg);                    % Log bkgnd
lr0  = lrh - mu0;                    % Fgnd lograte

% Define the preconditioner operator "M" as a function.
% This multiplies a vector by our prior covariance. Our prior is 
% translation-symmetric in space and can be implemented as convolution.
% In low-rank Hartley space, convolution becomes pointwise multiplication.
% M = @(u)(kernel.K.*u);

% Flatten 2D arena arrays into vectors
n = n(:);
k = k(:);

% Evidence as per-bin rates as pseudodata; 
% Missing data contribute no evidence. 
y = k./n;
y(~isfinite(y)) = 0.0;

% Reweighting for dispersion correction
gn = n.*opt.gamma; 

% Prior precision
PRIORPREC = (1.0./kernel.K); 

% Tall thin matrix of the 2D fourier components we actually use
% This is a semi-orthogonal operator that projects from the spatial
% domain into the low-rank frequency domain. R×L²

Q = eye(W*H);
Q = Q(1:end,kernel.keep);
Q = reshape(Q,W,H,kernel.R);
F = fft2(Q);
F = real(F) + imag(F);
F = reshape(F,W*H,kernel.R);
F = F';
orthonorm = 1.0/sqrt(W*H);
F = F*orthonorm; % Make pseudo-unitary

% Initialize estimated variables
v = zeros(N,1); % Initial marginal variances set to zero
z = F*lr0(:);      % Initial delta-log-rates 

assert(allfinite(v));
assert(allfinite(z));
assert(allfinite(mu0));
assert(allfinite(gn));
assert(allfinite(y));

% Coordinate descent model fitting
for i=1:opt.niter
    % Update the mean as a Baysian log-Gaussian point-process regression.
    for j=1:opt.nmeaniter
        % Calculate the mean rate λ of the log-Gaussian posterior.
        lambda = sexp(F'*z + mu0(:) + v.*0.5);
        assert(allfinite(lambda));

        % Multiply the evidence at each location (λ) by the number of
        % visits to each location "n" as well as any dispersion correction
        % "γ". Compute this in advance to avoid recalculation within the
        % inner loop of `minres()`.
        q = gn.*lambda;
        assert(allfinite(q));

        % The loss gradient for the low-rank frequency-space 
        % representation of the posterior mean-log-rate. This has two
        % contributions, one from the prior and one from the likeilhood. 
        % The prior contribution multiplies the current mean-log-rate
        % with the prior precision (inverse covariance). The likelihood
        % is the negative log Poisson likelihood in the spatial domain. 
        J = PRIORPREC.*z + F*(q-gn.*y);
        assert(allfinite(J));

        % The hessian-vector product operator for minres.
        % This has two parts, one from the prior, one from the likelihood.
        % The prior contribution is a convolution, which becomes pointwise
        % multiplication in our low-rank frequency space. The likelihood
        % contribution is an integral over the spatial domain. We need to
        % convert back to spatial coordinates, integrate against the
        % projected pseudopoints, then convert back to our low-rank space.
        % The semi-orthogonal operator F handles this. 
      
        % Minres is actually sloper than mldivide in Matlab
        Hv = F*(q.*F') + diag(PRIORPREC);
        dz = -Hv\J;
        assert(allfinite(dz));
        
        % Updates and convergence tests
        ez = max(abs(dz));
        z  = z + dz;
        if ez<opt.tol; break; end;
    end
    % Update the variance as if a Laplace approximation
    for j=1:opt.nvariter
        % Calculate the mean rate λ of the log-Gaussian posterior.
        lambda = sexp(F'*z + mu0(:) + v.*0.5);

        % Inverting the low-rank posterior covariance
        q  = gn.*lambda;
        x  = sqrt(q)'.*F;
        X  = diag(PRIORPREC) + x*x';
        C  = chol(X,'lower');
        A  = inv(C); % TODO: does matlab have a triangular inverse fn?
        DF = F'*A; % Remember F is R×L²
        v2 = sum(DF.^2,2);

        % Updates and convergence tests
        ev = max(abs(v2-v));
        v  = v2;
        if ev<opt.tol; break; end;
    end
end

% Calculate final loss
lograte = F'*z + mu0(:);
lambda = sexp(lograte + v.*0.5);
q  = gn.*lambda;
x  = sqrt(q)'.*F;
X  = diag(PRIORPREC) + x*x';
C  = chol(X,'lower');
C  = inv(C);
nyg  =  gn'*(lambda-y.*lograte);    %  n'(λ-y∘μ)
mTm  =  sum(z.^2.*PRIORPREC,"all"); %  μ'Λ₀μ
trTS =  sum(C.^2.*PRIORPREC,"all"); %  tr[Λ₀Σ]
ldSz = -sum(log(PRIORPREC),"all");  %  ln|Σ₀|
ldSq =  2*sum(log(diag(C)),"all");  % -ln|Σ|
%loss = nyg + .5*(ldSz);
loss = nyg + .5*(mTm + trTS + ldSz - ldSq - kernel.R);

% Build return struct
result.rhat = rhat;
result.rbg  = rbg ;
result.lrh  = lrh ;
result.mu0  = mu0 ;
result.lr0  = lr0 ;
result.gn   = gn  ; 
result.z    = z   ;
result.v    = v   ;
result.F    = F   ;
result.loss = loss;
result.meanlograte = reshape(lograte,W,H);
result.meanrate = reshape(lambda,W,H);

end




function y = slog(x)
    % Safe log function; Avoids numeric overflow by clipping
    y = log(max(x,1e-10));
end

function y = sexp(x)
    % Safe exponential function: Clip to avoid under/overflow
    y = exp(min(max(x,-10),10));
end

%{
function u = Fu(u,kernel)
    % Project vector down to low-rank space using FFT
    % This is MUCH slower in matlab
    [W,H] = size(kernel.kern);
    u = reshape(u,W,H);
    u = fft2(u);
    u = RI(u);
    u = u(kernel.keep)/sqrt(W*H);
end

function x = Ftu(u,kernel)
    % Project vector up from low-rank space using FFT
    % This is MUCH slower in matlab
    [W,H] = size(kernel.kern);
    x = zeros(W*H,1);
    x(kernel.keep(:)) = u*sqrt(W*H);
    x = reshape(x,W,H);
    x = RI(ifft2(x));
    x = reshape(x,W*H,1);
end

function x = RI(z)
    x = real(z) + imag(z);
end


opt.P         = 23.58975;  % Grid period in pixels
opt.kv        = 0.4109905; % Log-rate prior kernel marginal variance
%}