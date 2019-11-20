# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:11:33 2019

@author: loukrezis

Polynomial Chaos Expansions based on Dimension-Adaptive Leja Interpolation
"""

import chaospy as cp
import numpy as np
import openturns as ot
import scipy.linalg as scl
from idx_admissibility import admissible_neighbors
from leja1d import seq_lj_1d
from pce_tools import transform_multi_index_set, get_design_matrix

def dali_pce(func, N, jpdf_cp, jpdf_ot, tol=1e-12, max_fcalls=1000, verbose=True,
            interp_dict={}):

    if not interp_dict: # if dictionary is empty --> cold-start
        idx_act = []    # M_activated x N
        idx_adm = []    # M_admissible x N
        fevals_act = [] # M_activated x 1
        fevals_adm = [] # M_admissible x 1
        coeff_act = [] # M_activated x 1
        coeff_adm = [] #  M_admissible x 1

        # start with 0 multi-index
        knot0 = []
        for n in range(N):
        # get knots per dimension based on maximum index
            kk, ww = seq_lj_1d(order=0, dist=jpdf_cp[n])
            knot0.append(kk[0])
        feval = func(knot0)

        # update activated sets
        idx_act.append([0]*N)
        coeff_act.append(feval)
        fevals_act.append(feval)

        # local error indicators
        local_error_indicators = np.abs(coeff_act)

        # get the OT distribution type of each random variable
        dist_types = []
        for i in range(N):
            dist_type = jpdf_ot.getMarginal(i).getName()
            dist_types.append(dist_type)

        # create orthogonal univariate bases
        poly_collection = ot.PolynomialFamilyCollection(N)
        for i in range(N):
            if dist_types[i] == 'Uniform':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.LegendreFactory())
            elif dist_types[i] == 'Normal':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.HermiteFactory())
            elif dist_types[i] == 'Beta':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.JacobiFactory())
            elif dist_types[i] == 'Gamma':
                poly_collection[i] = ot.OrthogonalUniVariatePolynomialFamily(ot.LaguerreFactory())
            else:
                pdf = jpdf_ot.getDistributionCollection()[i]
                algo = ot.AdaptiveStieltjesAlgorithm(pdf)
                poly_collection[i] = ot.StandardDistributionPolynomialFactory(algo)

        # create multivariate basis
        mv_basis = ot.OrthogonalProductPolynomialFactory(poly_collection,
                                                        ot.EnumerateFunction(N))
        # get enumerate function (multi-index handling)
        enum_func = mv_basis.getEnumerateFunction()

    else:
        idx_act = interp_dict['idx_act']
        idx_adm = interp_dict['idx_adm']
        coeff_act  = interp_dict['coeff_act']
        coeff_adm  = interp_dict['coeff_adm']
        fevals_act = interp_dict['fevals_act']
        fevals_adm = interp_dict['fevals_adm']
        mv_basis = interp_dict['mv_basis']
        enum_func = interp_dict['enum_func']
        # local error indicators
        local_error_indicators = np.abs(coeff_adm)

    # compute global error indicator
    global_error_indicator = local_error_indicators.sum() # max or sum

    # fcalls / M approx. terms up to now
    fcalls = len(idx_act) + len(idx_adm) # fcalls = M --> approx. terms

    # maximum index per dimension
    max_idx_per_dim = np.max(idx_act + idx_adm, axis=0)

    # univariate knots and polynomials per dimension
    knots_per_dim = {}
    for n in range(N):
        kk, ww = seq_lj_1d(order=max_idx_per_dim[n], dist=jpdf_cp[n])
        knots_per_dim[n] = kk

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
            knots_n, weights_n = seq_lj_1d(an[n_ref], jpdf_cp[int(n_ref)])

            # update max_idx_per_dim, knots_per_dim, if necessary
            if an[n_ref] > max_idx_per_dim[n_ref]:
                max_idx_per_dim[n_ref] = an[n_ref]
                knots_per_dim[n_ref] = knots_n

            # find new_knot and compute function on new_knot
            new_knot = last_knot[:]
            new_knot[n_ref] = knots_n[-1]
            feval = func(new_knot)
            fevals_adm.append(feval)
            fcalls += 1 # update function calls

        # create PCE basis
        idx_system = idx_act + idx_adm
        idx_system_single = transform_multi_index_set(idx_system, enum_func)
        system_basis = mv_basis.getSubBasis(idx_system_single)
        # get corresponding evaluations
        fevals_system = fevals_act + fevals_adm
        # multi-dimensional knots
        M = len(idx_system) # equations terms
        knots_md = [[knots_per_dim[n][idx_system[m][n]] for m in range(M)] for n in range(N)]
        knots_md = np.array(knots_md).T
        # design matrix
        D = get_design_matrix(system_basis, knots_md)
        # solve system of equaations
        Q, R = scl.qr(D, mode='economic')
        c = Q.T.dot(fevals_system)
        coeff_system = scl.solve_triangular(R, c)

        # find the multi-index with the largest contribution, add it to idx_act
        # and delete it from idx_adm
        coeff_act = coeff_system[:len(idx_act)].tolist()
        coeff_adm = coeff_system[-len(idx_adm):].tolist()
        help_idx = np.argmax(np.abs(coeff_adm))
        idx_add = idx_adm.pop(help_idx)
        pce_coeff_add = coeff_adm.pop(help_idx)
        fevals_add = fevals_adm.pop(help_idx)
        idx_act.append(idx_add)
        coeff_act.append(pce_coeff_add)
        fevals_act.append(fevals_add)
        # re-compute coefficients of admissible multi-indices

        # local error indicators
        local_error_indicators = np.abs(coeff_adm)
        # compute global error indicator
        global_error_indicator = local_error_indicators.sum() # max or sum

    # store expansion data in dictionary
    interp_dict = {}
    interp_dict['idx_act'] = idx_act
    interp_dict['idx_adm'] = idx_adm
    interp_dict['coeff_act'] = coeff_act
    interp_dict['coeff_adm'] = coeff_adm
    interp_dict['fevals_act'] = fevals_act
    interp_dict['fevals_adm'] = fevals_adm
    interp_dict['enum_func'] = enum_func
    interp_dict['mv_basis'] = mv_basis
    return interp_dict
