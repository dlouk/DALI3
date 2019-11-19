# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:55:45 2018

@author: Dimitris Loukrezis

Test DALI algorithm on meromorphic function 1/(G*Y), where
G = [g1, g2, ..., g16 = ][1, 0.5, 0.1, 0.05, 0.01, ..., 5*1e-8]
Y = [Y1, Y2, ..., Y16], Y_n ~ U[-1,1]
"""

import numpy as np
import pickle
from mero import qoi_mero
from mero_pdf import jpdf
import sys
sys.path.append("../../dali")
from dali import dali


# num RVs
N = 16
# function to be approximated
f = qoi_mero
# maximum function calls
max_fcalls = np.linspace(10, 90, 9).tolist()
#max_fcalls = max_fcalls + np.linspace(100, 500, 5).tolist()
max_fcalls = np.linspace(100, 1000, 10).tolist()

# approximate
interp_dict = {}
for mfc in max_fcalls:
    mfc = int(mfc)
    interp_dict = dali(f, N, jpdf, tol=1e-16, max_fcalls=mfc,
                       interp_dict=interp_dict, verbose=True)

    savename = 'mero_dicts/mero_dict_' + str(mfc) + '.pkl'
    pickle.dump(interp_dict, open(savename, 'wb'))

print("This is the end, beautiful friend.")