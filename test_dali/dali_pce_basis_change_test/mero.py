# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:49:03 2019

@author: loukrezis

Meromorphic function
"""

import numpy as np

def qoi_mero(yvec):
    """Meromorphic function"""
    gvec_tilde = np.array([1e0, 5*1e-1, 1e-1, 5*1e-2, 1e-2, 5*1e-3, 1e-3,
                           5*1e-4, 1e-4, 5*1e-5, 1e-5, 5*1e-6, 1e-6, 5*1e-7,
                           1e-7, 5*1e-8])
    coeff = 1.0 / (2.0*np.linalg.norm(gvec_tilde, ord=1))
    gvec = gvec_tilde * coeff
    dotprod = np.dot(gvec, yvec)
    return 1.0/(1 + dotprod)