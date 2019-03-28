# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:05:35 2017

@author: Dimitris Loukrezis

Leja interpolation on adaptively constructed sparse grid.
"""

import numpy as np
from leja1d import seq_lj_1d
from lagrange1d import Hierarchical1d


def interpolate_single(indices, coeffs, polys_per_dim, non_grid_knot):
    """Leja interpolation on adaptively constructed sparse grid."""
    non_grid_knot = np.array(non_grid_knot)
    indices = np.array(indices)
    N = len(non_grid_knot)
    M = len(indices)
    # per dimension evaluations
    evals_per_dim = {}
    for n in range(N):
        P = len(polys_per_dim[n])
        evals_per_dim[n] = np.ones(P)
        for p in range(1,P): # column 0 --> order 0 --> = 1.0
            evals_per_dim[n][p] = polys_per_dim[n][p].evaluate(non_grid_knot[n])[0]
    # loop over M approximation terms (i.e., indices)
    ievals = [[evals_per_dim[n][indices[m,n]] for m in range(M)]
               for n in range(N)]
    ievals = np.prod(ievals, axis=0)

    return np.dot(ievals, coeffs)


def interpolate_multiple(indices, coeffs, jpdf, non_grid_knots):
    """Leja interpolation on adaptively constructed sparse grid."""

    # get shape of non_grid_points array --> K knots, N parameters
    non_grid_knots = np.array(non_grid_knots)
    try: # case: 2d array, K x N
        K, N = np.shape(non_grid_knots)
    except: # case: 1d array, 1 x N
        K = 1
        N = len(non_grid_knots)
        non_grid_knots = non_grid_knots.reshape(K,N)

    # get shape of indices array --> M approx. terms, NN parameters
    indices = np.array(indices)
    try: # case: 2d array, M x NN
        M, NN = np.shape(indices)
    except: # case: 1d array, 1 x NN
        M = 1
        NN = len(indices)
        indices = indices.reshape(M,NN)

    # check if dimensions agree
    if N != NN:
        return "Error! Knot and multi-index dimensions do not agree!"

    # get maximum index per dimension (P_1, P_2,..., P_N)
    max_idx_per_dim = np.max(indices, axis=0)

    # get knots, polynomials and polynomial evaluations per dimension
    knots_per_dim = {} # should hold N 1D arrays, [1 x (P_n+1)]
    polys_per_dim = {} # should hold N 1D lists, [1 x (P_n+1)]
    evals_per_dim = {} # should hold N 2D arrays, [K x (P_n+1)]
    for n in range(N):
        # get knots per dimension based on maximum index
        kk, ww = seq_lj_1d(order=max_idx_per_dim[n], dist=jpdf[n])
        knots_per_dim[n] = kk
        # get polynomials per dimension based on knots
        P = len(kk) # no. of knots = no. of polynomials = P_n+1
        polys_per_dim[n] = [Hierarchical1d(kk[:p+1]) for p in range(P)]
        # univariate polynomial evaluations
        evals_per_dim[n] = np.ones([K, P])
        for p in range(1,P): # column 0 --> pol. order 0 --> = 1.0
            evals_per_dim[n][:,p] = polys_per_dim[n][p].evaluate(
                                                          non_grid_knots[:, n])
    # loop over M approximation terms (i.e., indices)
    evals_multidim = np.zeros(K)
    for m in range(M):
        # start with 1st dimension
        ievals_m = evals_per_dim[0][:, indices[m,0]]
        # multiply with the rest of the dimensions
        for n in range(1,N):
            ievals_m = np.multiply(ievals_m, evals_per_dim[n][:, indices[m,n]])
        # add m-th term
        evals_multidim = evals_multidim + ievals_m*coeffs[m]
    return evals_multidim
