
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:18:16 2018

@author: Dimitris Loukrezis

Test the DALI algorithm on a random harmonic oscillator model.
"""

import chaospy as cp
import numpy as np
import sys
sys.path.append("../dali")
from dali import dali
from postprocess import compute_cv_error, compute_moments


def harm_osc(x):
    """Analytical model of harmonic oscillator"""
    w0 = np.sqrt( (x[1] + x[2]) / x[0] )
    response = 3*x[3] - np.abs( 2*x[4]/(x[0]*w0*w0) * np.sin(0.5*w0*x[5]))
    return response

# parameter PDFs
pdf1 = cp.Uniform(1.0 - 3*0.05, 1.0 + 3*0.05)
pdf2 = cp.Uniform(1.0 - 3*0.01, 1.0 + 3*0.1)
pdf3 = cp.Uniform(0.1 - 3*0.01, 0.1 + 3*0.01)
pdf4 = cp.Uniform(0.5 - 3*0.05, 0.5 + 3*0.05)
pdf5 = cp.Uniform(0.45 - 3*0.075, 0.45 + 3*0.075)
pdf6 = cp.Uniform(1.0 - 3*0.2, 1.0 + 3*0.2)
# joint PDF
jpdf = cp.J(pdf1, pdf2, pdf3, pdf4, pdf5, pdf6)
# num RVs
N = 6
# function to be approximated
f = harm_osc
# results storage
max_fcalls = np.linspace(10,90,9).tolist()
max_fcalls = max_fcalls + np.linspace(100, 500, 5).tolist()
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
    print("Variance = ", sigma2)
    print("Function calls = ", fcalls)
    print("Act/Adm ratio = ", ratio)
    print("")

print("This is the end, beautiful friend.")