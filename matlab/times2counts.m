function binned = times2counts(st, factor)
%TIMES2COUNTS Bin spike times to counts with linear interpolation
%   BINNED = TIMES2COUNTS(ST,FACTOR) will sum up spike-time timestamps ST 
%   into bins of length 1.0/FACTOR samples, distributing the point mass of 
%   each spike across two adjacent bins via linear interpolation.

    % Convert spike time stamps to downsampled-bin time stamps
    st = double(st(:)).*factor;
    it = floor(st); % Integer part: left bin index
    ft = st - it;   % Fractional part: right bin weight
    binned = accumarray(cat(1,it,it+1),cat(1,1-ft,ft));
end

