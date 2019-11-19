# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:19:01 2019

@author: loukrezis

Compute Cross Validation error for DALI metamodel
"""

import numpy as np
import pickle
import sys
from mero import qoi_mero
from mero_pdf import jpdf
sys.path.append("../../DALI_Python3/dali")
from interpolation import interpolate_multiple


# num RVs
N = 16

# cross validation sample
np.random.seed(42)
nsamples = 100000
cv_samples_in = jpdf.sample(nsamples).T
cv_samples_out = [qoi_mero(sample) for sample in cv_samples_in]

max_fcalls = np.linspace(10, 90, 9).tolist()
max_fcalls = max_fcalls + np.linspace(100, 1000, 10).tolist()
cv_errz_max = []
cv_errz_rms = []
fcallz = []
for mfc in max_fcalls:
    mfc = int(mfc)
    print(mfc)
    interp_dict = pickle.load(open('mero_dicts/mero_dict_'+str(mfc)+'.pkl', 'rb'))
    idx = interp_dict['idx_act'] + interp_dict['idx_adm']
    hs = interp_dict['hs_act'] + interp_dict['hs_adm']
    ievals = interpolate_multiple(idx, hs, jpdf, cv_samples_in)
    cv_errors = np.array([np.abs(ievals[ns] - cv_samples_out[ns]) for ns in range(nsamples)])
    cv_errz_max.append(np.max(cv_errors))
    cv_errz_rms.append(np.sqrt(np.mean(cv_errors**2)))
    fcallz.append(len(idx))

cv_results = np.array([fcallz, cv_errz_max, cv_errz_rms]).T
np.savetxt('cv_results_mero_leja.txt', cv_results)
