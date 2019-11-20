# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:59:31 2019

@author: loukrezis

Probability density function for Meromorphic Function using OpenTURNS
"""

import openturns as ot

# construct joint pdf
N = 16
z = []
for i in range(N):
    if i%2==0:
        z.append(ot.TruncatedNormal(0,1,0,3))
    else:
        z.append(ot.TruncatedNormal(0,1,-3,0))
jpdf_ot = ot.ComposedDistribution(z)