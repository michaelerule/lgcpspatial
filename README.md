# lgcpspatial
Log-Gaussian Cox process Python library for spiking neural activity in hippocampus and surrounding structures.

# Python files

- `__init__.py`: Package header; Not used at the moment

- `basics.py`: Routines for Gaussian blurs, FFT-based convolutions and autocorrelation, constructing radially-symmetric kernels, and common types of linear operators.

- `config.py`: Matplotlib and numpy configuration

- `datasets.py`: Routines for loading and preparing Krupic lab datasets

- `estimators.py`: Histogram and Kernel density estimators. These provide a comparison for Gaussian process methods as well as heuristic initializers

- `plot.py`: A subset of plotting helpers copied from `neurotools/graphics/plot`, and some new helper routines to reduce notebook clutter

- `posterior.py`: Routines for further analysis of the posterior rate map returned from Gaussian process inference. 

- `simulate_data.py`: Routines to simulate spiking observations from grid fields. Used for ground-truth benchmarking of inference routines.

- `lgcp2d.py`: Generic routines for optimizing a range of Gaussian process paramterizations. Ultimately, this is not very useful, since the fixe-point-diagonal iteration is so much faster than the other parameterizations that there's no point to trying the other ones. 

- `util.py`: Miscellaneous utility functions used in the notebooks

- `variational.py`: Currently empty

# Notebooks

- `00_preliminaries.ipynb`: Simulate grid cell data. Demonstrate histogram and kernel density rate maps. Estimate period from these. 

- `04_test_heuristic_LGCP.ipynb`: Fit a linear and log-Gaussian-Poisson Gaussian process model with heuristic kernels.

- `01_gradient_checks.ipynb`: Use Jax's automatic differentiation to check the validity of gradients for variational inference. 

- `02_gradient_checks_2_v6.ipynb`: Prepare analytic gradient routines in the form that they will be used for GP inference. Prepare code to implement FFT-based algorithms using the Hartley transform.

- `10_vtest_variational_local.ipynb`: Demonstrate variational inference parameterizing the posterior covariance as $\boldsymbol\Sigma &\approx \mathbf A^\top \operatorname{diag}[\mathbf v] \mathbf A$

- `11_test_variational_lowrank.ipynb`: Demonstrate variational inference parameterizing the posterior covariance as $\boldsymbol\Sigma = \mathbf X \mathbf X^\top$. It turns out this doesn't work very well. The components have many parameters that need to be optimized. Matrix operations are slow. The number of components in $\mathbf X$ required for a reasonable approximation is large. The decomposition has some symmetries as well, so the optimization problem is underconstrained. 

- `09_test_variational.ipynb`: Most up-to-date notebook which uses the low-rank diagional precision form for the posterior covariance. 

# Notes

This folder contain notes and derivations and will likely be removed in the final release. 

# Obsolete files

dct.py: Implements a custom flavor of the Discrete Cosine Trasform. This is no longer used at present, and will be replaced with the simpler Hartley transform on release.
