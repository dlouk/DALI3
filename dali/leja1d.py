# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:47:35 2018

@author: Dimitris Loukrezis

Unidimensional Leja sequences.
"""

import chaospy as cp
import numpy as np


def new_lj_1d(order, dist=cp.Uniform(-1, 1), old_knots=[]):
    """New Leja interpolation knots and weights in 1d."""
    knots, weights = cp.quad_leja(order, dist)
    knots = knots.flatten()
    is_in = np.in1d(knots, old_knots, invert=True)
    new_knots = knots[is_in]
    new_weights = weights[is_in]
    return new_knots, new_weights


def seq_lj_1d(order, dist=cp.Uniform(-1, 1)):
    """Create 1d sequences of nodes/weights given an order."""
    knots = []
    weights = []
    for k in range(order+1):
        new_knots, new_weights = new_lj_1d(k, dist, knots)
        knots = np.append(knots, new_knots)
        weights = np.append(weights, new_weights)
    return knots, weights