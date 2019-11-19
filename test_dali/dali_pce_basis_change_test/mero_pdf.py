# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:51:15 2019

@author: loukrezis

Probability density function for Meromorphic Function using Chaospy
"""

import chaospy as cp

# marginal pdfs
pdf1  = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf2  = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf3  = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf4  = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf5  = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf6  = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf7  = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf8  = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf9  = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf10 = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf11 = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf12 = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf13 = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf14 = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
pdf15 = cp.TruncNormal(lower=0, upper=3, mu=0, sigma=1)
pdf16 = cp.TruncNormal(lower=-3, upper=0, mu=0, sigma=1)
# joint pdf
jpdf = cp.J(pdf1, pdf2, pdf3, pdf4, pdf5, pdf6, pdf7, pdf8, pdf9, pdf10, pdf11,
            pdf12, pdf13, pdf14, pdf15, pdf16)