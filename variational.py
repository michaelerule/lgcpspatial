#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
"""
--------------------------------------------------------------------------------
Finding a compact yet expressive parameterization of the posterior covariance
matrix is one challenge in designing a variational Bayesian implementation for 
Gaussian process inference. This file explores four approaches:

1. Parameterization as Σ=ADA' where D is a diagonal matrix and A is a fixed, 
   local Gaussian convolution kernel
2. Parameterization as Σ=[Λ+D]¯¹, where Λ is the prior convolution kernel and
   D is a diagonal matrix
3. Parameterization as Σ=XX' where X is low rank
4. Parameterization as Σ=QD'DQ' where Q is a unitary discrete cosine transform
   operator, implemented via FFT, which discards high-frequency components. 
   D is a triangular Cholesky factor of a low-rank covariance expressed in this
   reduced frequency space, U=DD'.
"""



