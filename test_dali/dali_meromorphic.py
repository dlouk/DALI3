# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:55:45 2018

@author: Dimitris Loukrezis

Test DALI algorithm on meromorphic function 1/(G*Y), where
G = [g1, g2, ..., g16 = ][1, 0.5, 0.1, 0.05, 0.01, ..., 5*1e-8]
Y = [Y1, Y2, ..., Y16], Y_n ~ U[-1,1]
"""

import chaospy as cp
import numpy as np
import sys
sys.path.append("../dali")
from dali import dali
from postprocess import compute_cv_error, compute_moments


def qoi_mero(yvec):
    """Meromorphic function"""
    gvec_tilde = np.array([1e0, 5*1e-1, 1e-1, 5*1e-2, 1e-2, 5*1e-3, 1e-3,
                           5*1e-4, 1e-4, 5*1e-5, 1e-5, 5*1e-6, 1e-6, 5*1e-7,
                           1e-7, 5*1e-8])
    coeff = 1.0 / (2.0*np.linalg.norm(gvec_tilde, ord=1))
    gvec = gvec_tilde * coeff
    dotprod = np.dot(gvec, yvec)
    return 1.0/(1 + dotprod)


# num RVs
N = 16
# joint PDF
jpdf = cp.Iid(cp.Uniform(-1,1), N)
#jpdf = cp.Iid(cp.Normal(0,1./3.), N)
# function to be approximated
f = qoi_mero
# maximum function calls
max_fcalls = np.linspace(10, 90, 9).tolist()
max_fcalls = max_fcalls + np.linspace(100, 500, 5).tolist()

# arrays for results storage
cv_errz = []
meanz = []
varz = []
fcallz = []
ratioz = []
# approximate
interp_dict = {}
for mfc in max_fcalls:
    mfc = int(mfc)
    interp_dict = dali(f, N, jpdf, tol=1e-16, max_fcalls=mfc,
                       interp_dict=interp_dict, verbose=False)
    # cross-validation error
    cv_err = compute_cv_error(interp_dict, jpdf, f, Nsamples=1000)
    cv_errz.append(cv_err)
    mu, sigma2 = compute_moments(interp_dict, jpdf)
    meanz.append(mu)
    varz.append(sigma2)
    fcalls = len(interp_dict['idx_act'] + interp_dict['idx_adm'])
    fcallz.append(fcalls)
    ratio = float(len(interp_dict['idx_act'])) / float(fcalls)
    ratioz.append(ratio)
    print("")
    print("Max. fcalls = ", mfc)
    print("Cross-validation error = ", cv_err)
    print("Expected value = ", mu)
    print("Function calls = ", fcalls)
    print("Act/Adm ratio = ", ratio)
    print("")

print("This is the end, beautiful friend.")
