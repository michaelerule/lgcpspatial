function [N,K] = binspikes(pxy,s,L,w)
% BINSPIKES bins spike and visit counts to a 2D grid with linear 
% interpolation. 
%   [N,K] = BINSPIKES(PXY,S,L,W) Accepts a 2D trajectory as a list of x
%   and y points PXY, as well as a list of non-negative spike counts 
%   S of the same length as PXY. These 2D coordinates PXY should be
%   normalized to lie with the unit square [0,1)². This function will then
%   construct a L×L (or L×W, if W is provided) grid and assign each spike 
%   count in S to a 2×2 neighborhood, distribution its point mass via 
%   linear interpolation. N is the number of visits to each bin, and K is 
%   the total number of spikes in that bin. 
    
    % Flatten input arrays, default weights to 1. 
    px = double(pxy(1:end,1));
    py = double(pxy(1:end,2));
    s  = double(s(:));
    if nargin==3; w = ones(numel(px),1); else w = w(:); end
    
    % Require all arguments to have same number of points
    T = numel(px);
    s = padout(s,T);
    assert( numel(py)==T && numel(w)==T && numel(s)==T);
    
    % Check that positions are in [0,1)²
    assert(max(px)< 1 && max(py)< 1 && min(px)>=0 && min(py)>=0);
    
    % The integer parts of the coordinate tells us the top-right bin location
    % of the 2×2 neighborhood. The fractional parts will tell us how to 
    % distribute the point mass within this neighborhood.
    px = px(:).*L;
    ix = floor(px);
    fx = px - ix;
    py = py(:).*L;
    iy = floor(py);
    fy = py - iy;
    
    % Sanity check: this assertion should never fail.
    assert(max(ix)<L && max(iy)<L);
    
    % Weights for each cell in the 2×2 neighborhood for each spike count.
    w00 = (1-fx) .* (1-fy) .* w;
    w01 = (1-fx) .*  fy    .* w;
    w10 =  fx    .* (1-fy) .* w;
    w11 =  fx    .*  fy    .* w;
    
    qx  = cat(1, ix , ix+1, ix  , ix+1);
    qy  = cat(1, iy , iy  , iy+1, iy+1);
    z   = cat(1, w00, w10 , w01 , w11 );
    
    q = qx.*L + qy;          % Pack as zero-indexed COLUMN MAJOR order. 
    N = accumarray(q,z);     % Tally visits to each bin
    u = z .* cat(1,s,s,s,s); % Reweight visits by spike count
    K = accumarray(q,u);     % Tally spikes in each bin
    
    % Restore to square shape
    N = reshape(padout(N,L*L),L,L);
    K = reshape(padout(K,L*L),L,L);
end

function x = padout(x,T)
    if numel(x)<T
        x(numel(x)+1:T)=0.0; 
    end
end


