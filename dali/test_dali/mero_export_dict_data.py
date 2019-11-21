# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:43:55 2019

@author: loukrezis

Export Leja nodes, weights, and model evaluations from the DALI surrogate model
"""

import numpy as np
import pickle
from mero_pdf import jpdf
import sys
sys.path.append("../")
from leja1d import seq_lj_1d


max_fcalls = np.linspace(100, 1000, 10).tolist()
for mfc in max_fcalls:
    mfc = int(mfc)
    print(mfc)

    # load dictionary
    mero_dict = pickle.load(open('mero_dicts/mero_dict_'+ str(mfc) +'.pkl', 'rb'))

    # export dictionary data
    idx = mero_dict['idx_act'] + mero_dict['idx_adm']
    hs = mero_dict['hs_act'] + mero_dict['hs_adm']
    fevals = mero_dict['fevals_act'] + mero_dict['fevals_adm']

    # find knots/weights
    max_idx_per_dim = np.max(idx, axis=0)
    M = len(hs) # approx. terms
    N = len(max_idx_per_dim) # dimensions
    # weights per dimension
    weights_per_dim = {}
    knots_per_dim = {}
    for n in range(N):
        # get knots per dimension based on maximum index
        kk, ww = seq_lj_1d(order=max_idx_per_dim[n], dist=jpdf[n])
        weights_per_dim[n] = ww
        knots_per_dim[n] = kk
    # multi-dimensional knots
    knots_md = [[knots_per_dim[n][idx[m][n]] for m in range(M)] for n in range(N)]
    # multi-dimensional weights
    weights_md = [[weights_per_dim[n][idx[m][n]] for m in range(M)] for n in range(N)]
    weights_md = np.prod(weights_md, axis=0)

    np.savetxt('mero_leja_ED/leja_ed_in_' + str(mfc) + '.txt', np.array(knots_md).T)
    np.savetxt('mero_leja_ED/leja_ed_out_' + str(mfc) + '.txt', fevals)
    np.savetxt('mero_leja_ED/leja_quad_weights_' + str(mfc) + '.txt', weights_md)