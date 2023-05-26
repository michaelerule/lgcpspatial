
%{
This script contains example commands for demonstrating the log-Gaussian
inference routines implemented in this folder. This implementation is 
slightly slower and less complete than the Python implementation. 

Key files: 

 - NORMALIZEPOSITION(XY,MARGIN): This converts XY path data into normalized
 [0,1)² coordinates with a bit of padding, in preparation for binning data
 to a grid. 
 - TIMES2COUNTS(ST,FACTOR): This bins spikes in time, with linear
 interpolation. Use this to downsample a spike train to the same rate as
 your position data. 
 - BINSPIKES(PXY,S,L,W): Bin spikes S onto occuring at positions XY onto a
 L×W spatial grid (with linear interpolation). 
 - MAKEGRIDCELLKERNEL(L,P,V,OPT): Prepare a 2D grid cell covariance kernel,
 its 2D Fourier transform, and truncate this Fourier transform to make a 
 low-rank representation.
 - LGCPREGRESS(N,K,KERNEL): This function computes the variational
 posterior for a given grid of visit counts N, spike counts K, and kernel
 function KERNEL.

%}

% Load a dataset and collapse data onto basis functions
clear all; 
load([pwd '/../example data/r2405_051216b_cell1816.mat']);
pxy    = normalizeposition(xy);
s      = times2counts(spikes_times,pos_sample_rate./spk_sample_rate);
L      = 128;
[n,k]  = binspikes(pxy,s,L);

% Estimate model with known parameters
P      = 24.0;
kv     = 0.30;
kernel = makegridcellkernel(L,P,kv);
result = lgcpregress(n,k,kernel);

% Extract arena boundary
qhull   = pxy(convhull(pxy,'simplify',true),:);
points  = (0:L-1)/L;
[qx,qy] = meshgrid(points,points);
inside  = inpolygon(qx(:),qy(:),qhull(1:end,1),qhull(1:end,2));
mask    = double(inside);
mask(~inside) = NaN;
mask    = reshape(mask,L,L);

% Plot inferred map
close all;
im = imagesc(result.meanrate.*mask, interpolation='bilinear');
colormap(flipud(bone));
set(im, 'AlphaData', mask)
hold all;
plot(qhull(1:end,1)*L+1.0, qhull(1:end,2)*L+1.0, 'w', "linewidth", 5);
plot(qhull(1:end,1)*L+1.0, qhull(1:end,2)*L+1.0, 'k', "linewidth", 1 );
axis equal;

% Code to optimize kernel via brute-force grid search.
NGRID   = 10;
TODO    = NGRID*NGRID;
scan_P  = exp(linspace(log(15)  ,log(40),NGRID));
scan_kv = exp(linspace(log(0.05),log(7.5),NGRID));
[iP,iV] = meshgrid(scan_P, scan_kv);
iP = iP(:);
iV = iV(:);
lambda = @(i)(lgcpregress(n,k,makegridcellkernel(L,iP(i),iV(i))).loss);
losses = parforprogress(TODO,lambda,sprintf("Searching %d×%d parameter grid",NGRID,NGRID),true);

% Plot loss surface
[~,p] = sort(losses,'descend');
r = 1:length(losses);
r(p) = r;
r = reshape(r./TODO,NGRID,NGRID)
imagesc(r);
colormap(bone);

% Pick best parameters
[best,i] = max(r(:));
[ix,iy]=ind2sub(size(r),i);
bestp=scan_P(iy)
bestv=scan_kv(ix)

% Estimate model with inferred parameters
kernel = makegridcellkernel(L,bestp,bestv);
result = lgcpregress(n,k,kernel);

% Plot inferred map
close all;
im = imagesc(result.meanrate.*mask, interpolation='bilinear');
colormap(flipud(bone));
set(im, 'AlphaData', mask)
hold all;
plot(qhull(1:end,1)*L+1.0, qhull(1:end,2)*L+1.0, 'w', "linewidth", 5);
plot(qhull(1:end,1)*L+1.0, qhull(1:end,2)*L+1.0, 'k', "linewidth", 1 );
axis equal;









