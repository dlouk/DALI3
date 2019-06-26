# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:20:20 2018

@author: Dimitris Loukrezis

Post-process the information of a DALI interpolation dictionary.
"""

import chaospy as cp
import numpy as np
from interpolation import interpolate_multiple
from leja1d import seq_lj_1d


def compute_cv_error(interp_dict, jpdf, func, Nsamples=1000, seed=165,
                     which='all'):
    """Compute maximum discrete error from a cross-validation set with Nsamples
    random realizations of the joint PDF."""
    np.random.seed(seed)
    test_points = jpdf.sample(Nsamples).T
    fevals = np.array([func(tp) for tp in test_points])
    if which == 'all':
        idx = interp_dict['idx_act'] + interp_dict['idx_adm']
        hs = interp_dict['hs_act'] + interp_dict['hs_adm']
    else:
        idx = interp_dict['idx_act']
        hs = interp_dict['hs_act']
    ievals = interpolate_multiple(idx, hs, jpdf, test_points)
    errs = [fevals[ns] - ievals[ns] for ns in range(Nsamples)]
    cv_err = np.max(np.abs(errs))
    return cv_err


def compute_moments(interp_dict, jpdf, which='all'):
    """Compute expected value and variance."""
    if which=='all':
        idx = np.array(interp_dict['idx_act'] + interp_dict['idx_adm'])
        hs = interp_dict['hs_act'] + interp_dict['hs_adm']
        hs2 = interp_dict['hs2_act'] + interp_dict['hs2_adm']
    else:
        idx = np.array(interp_dict['idx_act'])
        hs = interp_dict['hs_act']
        hs2 = interp_dict['hs2_act']
    max_idx_per_dim = np.max(idx, axis=0)
    M = len(hs) # approx. terms
    N = len(max_idx_per_dim) # dimensions
    # weights per dimension
    weights_per_dim = {}
    for n in range(N):
        # get knots per dimension based on maximum index
        kk, ww = seq_lj_1d(order=max_idx_per_dim[n], dist=jpdf[n])
        weights_per_dim[n] = ww
    # multi-dimensional weights
    weights_md = [[weights_per_dim[n][idx[m,n]] for m in range(M)]
                   for n in range(N)]
    weights_md = np.prod(weights_md, axis=0)
    # moments (!variance computation is unstable!)
    expected = np.dot(weights_md, hs)
    variance = np.dot(weights_md, hs2) - expected*expected
    return expected, variance

