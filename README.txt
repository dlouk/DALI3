Dimension-Adaptive Leja Interpolation (DALI)
(alternatively: DArmstadt's Leja Interpolation)

Adjusted to Python3 
--------------------------------------------------------------------------------

Development/maintenance: Dimitrios Loukrezis (loukrezis@temf.tu-darmstadt.de)
--------------------------------------------------------------------------------

DALI is a Python software for multivariate approximation, using a 
dimension-adaptive stochastic collocation algorithm based on univariate Leja 
interpolation rules. The software has been developed during my PhD studies at 
the Institute for Theory of Electromagnetic Fields (TEMF) of TU Darmstadt, under 
the supervision of Prof. Dr.-Ing. Herbert De Gersem (TU Darmstadt) and 
Jun.-Prof. Dr.-Ing. Ulrich Römer (TU Braunschweig).
--------------------------------------------------------------------------------

The DALI software has been used in the studies presented in the following papers:

@article{loukrezis2019assessing,
author  = {Dimitrios Loukrezis and Ulrich  Römer and Herbert  De Gersem},
title   = {Assessing the performance of Leja and Clenshaw-Curtis collocation for 
           computational electromagnetics with random input data},
journal = {International Journal for Uncertainty Quantification},
issn    = {2152-5080},
year    = {2019},
volume  = {9},
number  = {1},
pages   = {33--57}
}

@article{georg2018uncertainty,
       author = {{Georg}, Niklas and {Loukrezis}, Dimitrios and {R{\"o}mer}, Ulrich and
         {Sch{\"o}ps}, Sebastian},
        title = "{Uncertainty quantification for an optical grating coupler with an adjoint-based Leja adaptive collocation method}",
      journal = {arXiv e-prints},
         year = "2018",
          eid = {arXiv:1807.07485},
}

@article{loukrezis2019approximation,
       author = {{Loukrezis}, Dimitrios and {De Gersem}, Herbert},
        title = "{Approximation and Uncertainty Quantification of Stochastic Systems with Arbitrary Input Distributions using Weighted Leja Interpolation}",
      journal = {arXiv e-prints},
         year = "2019",
          eid = {arXiv:1904.07709},
}

In accordance to ethical scientific practice, we kindly ask to cite at least one 
of these works, in case you use DALI for your own research.
--------------------------------------------------------------------------------

Regarding the mathematical background of the algorithm implemented in DALI, we 
suggest the following papers:
- "Dimension-Adaptive Tensor-Product Quadrature", Gerstner and Griebel, 
Computing, 2003
- "High-Dimensional Adaptive Sparse Polynomial Interpolation and Applications 
to Parametric PDEs", Chkifa, Cohen, and Schwab, Found. Comput. Math., 2014
- "Adaptive Leja Sparse Grid Constructions for Stochastic Collocation and 
High-Dimensional Approximation", Narayan and Jakeman, SIAM Sci. Comput., 2014
--------------------------------------------------------------------------------

The present software and the related examples rely partially on the Chaospy 
Python toolbox.
- https://github.com/jonathf/chaospy 
- "Chaospy: An open source tool for designing methods of uncertainty 
quantification", Feinberg and Langtangen, J. Comput. Sci., 2015

Please note that using DALI in combination with Chaospy implies that the user 
respects the corresponding copyright notices and license's disclaimers of 
warranty.  
--------------------------------------------------------------------------------
