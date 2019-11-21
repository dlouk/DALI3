# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:51:15 2019

@author: loukrezis

Probability density function for the meromorphic function using Chaospy
"""

import chaospy as cp

# num RVs
N = 50
# marginal pdfs
jpdf = cp.Iid(cp.Uniform(-1,1), N)
#jpdf = cp.Iid(cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1), N)
