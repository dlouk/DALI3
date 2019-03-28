# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:18:26 2018

@author: Dimitris Loukrezis

Dimension-Adaptive Leja Interpolation (DALI) algorithm.
"""


import numpy as np
from idx_admissibility import admissible_neighbors
from leja1d import seq_lj_1d
from lagrange1d import Hierarchical1d
#from itertools import izip
from interpolation import interpolate_single


def dali(func, N, jpdf, tol=1e-12, max_fcalls=1000, verbose=True,
         interp_dict={}):
    """Dimension-Adaptive Leja Interpolation (DALI) algorithm.
    FUNC: function to be approximated.
    N: number of parameters.
    JPDF: joint probability density function.
    TOL, MAX_FCALLS: exit criteria, self-explanatory.

    'ACT': activated, i.e. already part of the approximation.
    'ADM': admissible, i.e. candidates for the approximation's expansion."""

    if not interp_dict: # if dictionary is empty --> cold-start
        idx_act = []    # M_activated x N
        idx_adm = []    # M_admissible x N
        hs_act  = []    # M_activated x 1
        hs_adm  = []    # M_admissible x 1
        hs2_act = []    # M_activated x 1
        hs2_adm = []    # M_admissible x 1
        fevals_act = [] # M_activated x 1
        fevals_adm = [] # M_admissible x 1


        # start with 0 multi-index
        knot0 = []
        for n in range(N):
        # get knots per dimension based on maximum index
            kk, ww = seq_lj_1d(order=0, dist=jpdf[n])
            knot0.append(kk[0])
        feval = func(knot0)

        # update activated sets
        idx_act.append([0]*N)
        hs_act.append(feval)
        hs2_act.append(feval*feval)
        fevals_act.append(feval)

        # local error indicators
        local_error_indicators = np.abs(hs_act)

    else: # get data from dictionary
        idx_act = interp_dict['idx_act']
        idx_adm = interp_dict['idx_adm']
        hs_act  = interp_dict['hs_act']
        hs_adm  = interp_dict['hs_adm']
        hs2_act = interp_dict['hs2_act']
        hs2_adm = interp_dict['hs2_adm']
        fevals_act = interp_dict['fevals_act']
        fevals_adm = interp_dict['fevals_adm']

        # local error indicators
        local_error_indicators = np.abs(hs_adm)

    # compute global error indicator
    global_error_indicator = local_error_indicators.sum() # max or sum

    # fcalls / M approx. terms up to now
    fcalls = len(idx_act) + len(idx_adm) # fcalls = M --> approx. terms

    # maximum index per dimension
    max_idx_per_dim = np.max(idx_act + idx_adm, axis=0)

    # univariate knots and polynomials per dimension
    knots_per_dim = {}
    polys_per_dim = {}
    for n in range(N):
        kk, ww = seq_lj_1d(order=max_idx_per_dim[n], dist=jpdf[n])
        knots_per_dim[n] = kk
        P = len(kk) # no. of knots = no. of polynomials = P_n+1
        polys_per_dim[n] = [Hierarchical1d(kk[:p+1]) for p in range(P)]

    # start iterations
    while global_error_indicator > tol and fcalls < max_fcalls:
        if verbose:
            print(fcalls)
            print(global_error_indicator)

        # the index added last to the activated set is the one to be refined
        last_act_idx = idx_act[-1][:]
        # compute the knot corresponding to the lastly added index
        last_knot = [knots_per_dim[n][i]
                     for n, i in zip(range(N), last_act_idx)]
        # get admissible neighbors of the lastly added index
        adm_neighbors = admissible_neighbors(last_act_idx, idx_act)

        for an in adm_neighbors:
            # update admissible index set
            idx_adm.append(an)
            # find which parameter/direction n (n=1,2,...,N) gets refined
            n_ref = np.argmin([idx1 == idx2
                                 for idx1, idx2 in zip(an, last_act_idx)])
            # sequence of 1d Leja nodes/weights for the given refinement
            knots_n, weights_n = seq_lj_1d(an[n_ref], jpdf[int(n_ref)])

            # update max_idx_per_dim, knots_per_dim, if necessary
            if an[n_ref] > max_idx_per_dim[n_ref]:
                max_idx_per_dim[n_ref] = an[n_ref]
                knots_per_dim[n_ref] = knots_n
                polys_per_dim[n_ref].append(Hierarchical1d(knots_n))

            # find new_knot and compute hierarchical surpluses
            new_knot = last_knot[:]
            new_knot[n_ref] = knots_n[-1]
            feval = func(new_knot)
            feval2 = feval*feval
            fevals_adm.append(feval)
            ieval = interpolate_single(idx_act, hs_act, polys_per_dim, new_knot)
            ieval2 = interpolate_single(idx_act, hs2_act, polys_per_dim, new_knot)
            HS = feval - ieval
            HS2 = feval2 - ieval2
            hs_adm.append(HS)
            hs2_adm.append(HS2)
            fcalls += 1 # update function calls

        # update error indicators
        local_error_indicators = np.abs(hs_adm)
        global_error_indicator = local_error_indicators.sum() # max or sum?

        # find index from idx_adm with maximum local error indicator
        help_idx = np.argmax(local_error_indicators)
        # remove index, hs and hs2 from admissible sets
        idx_add = idx_adm.pop(help_idx)
        hs_add = hs_adm.pop(help_idx)
        hs2_add = hs2_adm.pop(help_idx)
        feval_add = fevals_adm.pop(help_idx)
        # update activated sets
        idx_act.append(idx_add)
        hs_act.append(hs_add)
        hs2_act.append(hs2_add)
        fevals_act.append(feval_add)

    #store data to dictionary
    interp_dict = {}
    interp_dict['idx_act'] = idx_act
    interp_dict['idx_adm'] = idx_adm
    interp_dict['hs_act']  = hs_act
    interp_dict['hs2_act'] = hs2_act
    interp_dict['hs_adm']  = hs_adm
    interp_dict['hs2_adm'] = hs2_adm
    interp_dict['fevals_act'] = fevals_act
    interp_dict['fevals_adm'] = fevals_adm
    return interp_dict

