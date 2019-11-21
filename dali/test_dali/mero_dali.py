# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:55:45 2018

@author: Dimitris Loukrezis

Test DALI algorithm on the N-dimensional meromorphic function
"""

import numpy as np
import pickle
from meroND import meroND
from mero_pdf import jpdf
import sys
sys.path.append("../")
from dali import dali


# num RVs
N = len(jpdf)

# function to be approximated
f = meroND

# maximum function calls
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
