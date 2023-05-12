***Under construction; please check back later!*** 
This manuscript has not yet been peer-reviewed. Do get in touch if you find errors, or if you notice places where I've failed to cite relevant prior work. 

# Variational Log-Gaussian Point-Process Methods for Grid Cells

Rule, M., Chaudhuri-Vayalambrone, P., Krstulovic, M., Bauza, M., Krupic, J., & O’Leary, T. (2023). Variational Log-Gaussian Point-Process Methods for Grid Cells. [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.18.533177v2.abstract), 2023-03. [doi: https://doi.org/10.1101/2023.03.18.533177](https://doi.org/10.1101/2023.03.18.533177).

![](https://github.com/michaelerule/lgcpspatial/blob/main/F4.large.png)

### Abstract

We present practical solutions to applying Gaussian-process methods to calculate spatial statistics for grid cells in large environments. Gaussian processes are a data efficient approach to inferring neural tuning as a function of time, space, and other variables. We discuss how to design appropriate kernels for grid cells, and show that a variational Bayesian approach to log-Gaussian Poisson models can be calculated quickly. This class of models has closed-form expressions for the evidence lower-bound, and can be estimated rapidly for certain parameterizations of the posterior covariance. We provide an implementation that operates in a low-rank spatial frequency subspace for further acceleration. We demonstrate these methods on a recording from grid cells in a large environment.

### Repository structure

This repository contains example implementations of log-Gaussian Cox process regressions for analyzing grid cells (and perhaps other periodic, densely sampled 2D spatial datasets).

 - `docs/`: Documentation for the python `lgcpspatial` package, [browsable as HTML here.](https://michaelerule.github.io/lgcpspatial/index.html). 
 - `example data/`: An example grid cell from [Krupic et al. (2018)](https://doi.org/10.1126/science.aao4960).
 - `lgcpspatial/`: An example implementation as a python package 
 - `tutorials /`: IPython notebooks demonstrating use of the `lgcpspatial` package. 




