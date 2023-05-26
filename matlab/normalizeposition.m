function [pxy,info] = normalizeposition(xy,margin)
%NORMALIZEPOSITION Rescale an NPOINTS×2 array of (X,Y) points to [0,1)²
%   [PXY,INFO] = NORMALIZEPOSITION(XY,MARGIN) Expects an NPOINTS×2 array of 
%   (X,Y) point data as input, and rescales all points to lie in [0,1)² 
%   with padding fraction MARGIN (If MARGIN is omitted, it defaults to 
%   0.1, which will add 10% padding to all edges). PXY contains the
%   rescaled points, and INFO is a struct containing INFO.extent (the
%   (x0,x1,y0,y1) coordinates in the original units) and INFO.scale (the 
%   number you should divide by to convert from [0,1)² units to the 
%   original unitS).

% Provide default value pad = 0.1
if nargin==1; margin = 0.1; end
info.margin = margin;
pad = 0.5 + margin;

% Extract point data
x = xy(1:end,1);
y = xy(1:end,2);

% Fill in NaNs
x = inerpolateNaN(x);
y = inerpolateNaN(y);

% Get range of point data
minx = min(x);
maxx = max(x);
miny = min(y);
maxy = max(y);

% Determine largest axis span
dx = maxx - minx;
dy = maxy - miny;
delta = max(dx,dy);

% Spacing to add
edge = delta*pad;

% Get middle of point data (used for centering)
midx = (maxx+minx)./2;
midy = (maxy+miny)./2;

% Multiplying by scale will send you to [0,1)² normalized units
% Dividing by scale will get you back to the original units of (X,Y).
scale = (1-sqrt(eps))./(edge*2);
info.scale  = scale;

% Extent of [0,1)² patch in original (X,Y) units.
info.x0 = midx-edge;
info.y0 = midy-edge;
info.x1 = info.x0 + 1./scale;
info.y1 = info.y0 + 1./scale;
info.extent = [info.x0,info.x1,info.y0,info.y1];

% Center data with padding scaled to [0,1)²
px = (x-info.x0)*scale;
py = (y-info.y0)*scale;
pxy = [px,py];
end


function x = inerpolateNaN(x)
    t = linspace(0,1,numel(x));
    nans = isnan(x);
    x(nans) = interp1(t(~nans), x(~nans), t(nans), 'pchip');
end








        
