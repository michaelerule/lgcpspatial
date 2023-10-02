
*This manuscript has been peer-reviewed and accepted, but the reference implementation contained in this repository has not been tested on all platforms; Please report any mistakes, bugs, or difficulties with installation that you encounter, including places were the documentation is inadequate.*

## Variational Log-Gaussian Point-Process Methods for Grid Cells

Rule, M. E., Chaudhuri-Vayalambrone, P., Krstulovic, M., Bauza, M., Krupic, J., & O'Leary, T. (2023). Variational log-Gaussian point-process methods for grid cells. Hippocampus, 1–17. [doi: https://doi.org/10.1002/hipo.23577](https://doi.org/10.1002/hipo.23577).

![](https://github.com/michaelerule/lgcpspatial/blob/main/f5v1.png)

### Abstract

We present practical solutions to applying Gaussian-process methods to calculate spatial statistics for grid cells in large environments. Gaussian processes are a data efficient approach to inferring neural tuning as a function of time, space, and other variables. We discuss how to design appropriate kernels for grid cells, and show that a variational Bayesian approach to log-Gaussian Poisson models can be calculated quickly. This class of models has closed-form expressions for the evidence lower-bound, and can be estimated rapidly for certain parameterizations of the posterior covariance. We provide an implementation that operates in a low-rank spatial frequency subspace for further acceleration. We demonstrate these methods on a recording from grid cells in a large environment.

### Repository structure

This repository contains example implementations of log-Gaussian Cox process regressions for analyzing grid cells (and perhaps other periodic, densely sampled 2D spatial datasets).

 - `example data/`: An example grid cell from [Krupic et al. (2018)](https://doi.org/10.1126/science.aao4960).
 - `python/lgcp/`: An example implementation as a python package 
 - `python/notebooks/`: IPython notebooks to reproduce the figures in the manuscript
 - `matlab/`: A (less-complete) provisional Matlab implementation
 - `old_versions/`: Previous iterations




