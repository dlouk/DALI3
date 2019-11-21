# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:59:31 2019

@author: loukrezis

Probability density function for the meromorphic function using OpenTURNS
"""

import openturns as ot

# construct joint pdf
N = 50
z = []
for i in range(N):
    z.append(ot.Uniform(-1,1))
    #z.append(ot.TruncatedNormal(0,1,0,3))
jpdf_ot = ot.ComposedDistribution(z)