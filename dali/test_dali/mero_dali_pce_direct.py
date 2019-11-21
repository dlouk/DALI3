# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:55:45 2018

@author: Dimitris Loukrezis

Test DALI-PCE algorithm on the N-dimensional meromorphic function
"""


import numpy as np
import sys
from meroND import meroND
from mero_pdf import jpdf
from mero_pdf_ot import jpdf_ot
sys.path.append("../")
from dali_pce import dali_pce
from pce_tools import PCE_Surrogate


# num RVs
N = len(jpdf)

# function to be approximated
f = meroND

# maximum function calls
max_fcalls = np.linspace(100, 1000, 10).tolist()

# cross validation sample
np.random.seed(42)
cv_sample_in = jpdf.sample(1000).T
cv_sample_out = np.array([f(cvi) for cvi in cv_sample_in])

# arrays for results storage
cv_errz_max = []
cv_errz_rms = []
meanz = []
varz = []
fcallz = []
# approximate
interp_dict = {}
for mfc in max_fcalls:
    mfc = int(mfc)
    interp_dict = dali_pce(f, N, jpdf, jpdf_ot, tol=1e-16, max_fcalls=mfc,
                           interp_dict=interp_dict, verbose=False)
    idx = interp_dict['idx_act'] + interp_dict['idx_adm']
    coeff = interp_dict['coeff_act'] + interp_dict['coeff_adm']
    pce_sur = PCE_Surrogate(beta_coeff=coeff, idx_set=idx, jpdf=jpdf_ot)
    ievals = pce_sur.evaluate(cv_sample_in)
    # compute cross-validation errors
    errs = np.abs(cv_sample_out-np.reshape(np.asarray(ievals), np.asarray(ievals).shape[0]))
    cv_errz_max.append(np.max(errs))
    cv_errz_rms.append(np.sqrt(np.sum(errs**2)/len(errs)))
    fcallz.append(len(idx))
    # moments
    expected = coeff[0]
    meanz.append(expected)
    var_per_term = np.array(coeff)[1:]**2
    variance = np.sum(var_per_term)
    varz.append(variance)
    print("")
    print("Max. fcalls = ", mfc)
    print("Cross-validation error, MAX = ", cv_errz_max[-1])
    print("Cross-validation error, RMS = ", cv_errz_rms[-1])
    print("Expected value = ", expected)
    print("Variance = ", variance)
    print("Function calls = ", fcallz[-1])
    print("")

approx_results = np.array([fcallz, cv_errz_max, cv_errz_rms, meanz, varz]).T
np.savetxt('approx_results_mero_dali_pce_direct.txt', approx_results)

print("This is the end, beautiful friend.")