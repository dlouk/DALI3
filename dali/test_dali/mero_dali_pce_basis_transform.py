# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:27:16 2019

@author: loukrezis

Compute PCE with a basis transform from hierarchical polynomials to orthogonal ones
"""

import numpy as np
import openturns as ot
import pickle
import scipy.linalg as scl
import sys
#import time
from meroND import meroND
from mero_pdf import jpdf
from mero_pdf_ot import jpdf_ot
sys.path.append("../")
from pce_tools import transform_multi_index_set, get_design_matrix, PCE_Surrogate


# construct joint pdf
N = len(jpdf)

# get the distribution type of each random variable
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


max_fcalls = np.linspace(100, 1000, 10).tolist()
meanz = []
varz = []
cv_errz_rms = []
cv_errz_max = []
fcallz = []
# cross validation sample
np.random.seed(42)
nsamples = 1000
cv_samples_in = jpdf.sample(nsamples).T
cv_samples_out = [meroND(sample) for sample in cv_samples_in]
for mfc in max_fcalls:
    # load interpolation data
    mfc = int(mfc)
    print(mfc)
    interp_dict = pickle.load(open('mero_dicts/mero_dict_'+str(mfc)+'.pkl', 'rb'))
    idx = interp_dict['idx_act'] + interp_dict['idx_adm']
    hs = interp_dict['hs_act'] + interp_dict['hs_adm']
    fevals = interp_dict['fevals_act'] + interp_dict['fevals_adm']
    leja_nodes = np.loadtxt('mero_leja_ED/leja_ed_in_' + str(mfc) + '.txt')
    fcallz.append(len(idx))
    #
    # create orthogonal basis
    idx_single = transform_multi_index_set(idx, enum_func)
    orth_basis = mv_basis.getSubBasis(idx_single)
    orth_basis_size = len(orth_basis)
    # design matrix
    D = get_design_matrix(orth_basis, leja_nodes)
    Q, R = scl.qr(D, mode='economic')
    #invD = np.linalg.inv(D)
    #pce_coeffs = np.dot(invD, fevals)
    c = Q.T.dot(fevals)
    pce_coeffs = scl.solve_triangular(R, c)
    # moments
    expected = pce_coeffs[0]
    meanz.append(expected)
    var_per_term = pce_coeffs[1:]**2
    variance = np.sum(var_per_term)
    varz.append(variance)
    #
    # create PCE surrogate model
    pce_sur = PCE_Surrogate(beta_coeff=pce_coeffs, idx_set=idx, jpdf=jpdf_ot)
    ievals = pce_sur.evaluate(cv_samples_in)
    # compute cross-validation errors
    errs = np.abs(cv_samples_out-np.reshape(np.asarray(ievals), np.asarray(ievals).shape[0]))
    cv_errz_max.append(np.max(errs))
    cv_errz_rms.append(np.sqrt(np.sum(errs**2)/len(errs)))


approx_results = np.array([fcallz, cv_errz_max, cv_errz_rms, meanz, varz]).T
np.savetxt('approx_results_mero_dali_pce_bt.txt', approx_results)