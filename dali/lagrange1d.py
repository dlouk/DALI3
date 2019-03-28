# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:09:10 2018

@author: Dimitris Loukrezis

Univariate Lagrange and hierarchical polynomials.
"""

import numpy as np


class Lagrange1d:
    """Univariate Lagrange nodal basis polynomial"""
    def __init__(self, current_knot, knots):
        self.current_knot = current_knot
        self.knots = np.array(knots)
        self.other_knots = np.setdiff1d(knots, current_knot)
        # compute denominator once and re-use
        self.denoms_prod = (self.current_knot - self.other_knots).prod()

    def evaluate(self, non_grid_knots):
        """Evaluate polynomial on specific non-grid knots"""
        non_grid_knots = np.array(non_grid_knots).flatten()
        L = list(map(lambda x: np.prod(x-self.other_knots)/self.denoms_prod,
                 non_grid_knots))
        return L


class Hierarchical1d(Lagrange1d):
    """Univariate Lagrange hierarchical basis polynomial"""
    def __init__(self, knots):
        self.knots = np.array(knots)
        self.current_knot = self.knots[-1]
        self.other_knots = self.knots[:-1]
        # compute denominator once and re-use
        self.denoms_prod = (self.current_knot - self.other_knots).prod()


def lagrange1d_eval(current_knot, other_knots, non_grid_knots):
    """Evaluate on NON_GRID_KNOTS a univariate Lagrange polynomial, defined
    for CURRENT_KNOT and OTHER_KNOTS"""
    other_knots = np.array(other_knots)
    non_grid_knots = np.array(non_grid_knots)
    denoms = current_knot - other_knots
    denoms_prod = denoms.prod()
    L = list(map(lambda x: np.prod(x - other_knots) / denoms_prod, non_grid_knots))
    return L