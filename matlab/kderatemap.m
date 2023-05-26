function ratemap = kde(n,k,sigma)
    %{
    Estimate rate using Gaussian KDE smoothing. This is 
    better than estimating the rate using a binned 
    histogram, but worse than a Gaussian-Process estimator. 
    
    Parameters
    ----------
    n: 2D np.array
        Number of visits to each location
    k: 2D np.array
        Number of spikes observed at each location
    sigma: float
        kernel radius exp(-x²/sigma) (standard deviation 
        in x and y ×⎷2)
    
    Returns
    -------
    ratemap: 2D array
        KDE rate estimate of firing rate in each bin
    %}
    
    % Blur the visit and spike counts separately
    n = blur2d(n,sigma);
    k = blur2d(k,sigma);

    % Get the cell's mean rate in spikes/visit
    alpha=sum(k)/sum(n);
    ratemap = (k+(alpha-0.5)*0.9+0.5)./(n+0.9);
end
