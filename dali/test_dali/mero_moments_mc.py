# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:29:23 2019

@author: loukrezis

Reference moment values for meroND
"""

import numpy as np
from meroND import meroND
from mero_pdf import jpdf


meanz = []
varz = []
nsamples = 1000000
np.random.seed(42)
for i in range(100):
    print(i)
    mc_sample_in = jpdf.sample(nsamples).T
    mc_sample_out = [meroND(si) for si in mc_sample_in]
    meanz.append(np.mean(mc_sample_out))
    varz.append(np.var(mc_sample_out))
mc_mean = np.mean(meanz)
mc_var = np.mean(varz)
