# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:51:07 2018

@authors: Armin Galetzka, Dimitris Loukrezis

Functions and classes used for Polynomial Chaos Expansions based on Leja
interpolation
"""


import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

def get_ed(func, jpdf, no_samples, sample_type='R', knots=[], values=[],
           ed_file=None, ed_fevals_file=None):
    """Returns randomly drawn knots and the corresponding evaluations.
    If knots!=0 in handle and the selected sequence is pseudo-random,
    or nested, only the additional knots to reach no_samples are computed
    func: function to be evaluated
    M: Number of samples
    jpdf: joint probabilty density function
    sample_type: choose sequence samples are drawn from
                 R - random (Monte Carlo) sampling
                 L - latin hypercube sampling
    knots: already excisting knots
    values: already excisting values"""

    if ed_file != None:
        knots_pool=np.genfromtxt(ed_file, delimiter=',')
        fvalues_pool=np.genfromtxt(ed_fevals_file)
    if knots == []:
        if ed_file != None:
            if no_samples<=knots_pool.shape[0]:
                knots=knots_pool[:no_samples,:]
                values=fvalues_pool[:no_samples]
                return knots, values
            else:
                knots=knots_pool
                values=fvalues_pool
                knots, values=get_ed(func,no_samples,jpdf,sample_type,knots,values)
                return knots, values
        else:
            if sample_type == 'R':
                knots=np.array(jpdf.getSample(no_samples))
            elif sample_type == 'L':
                knots=np.array(ot.LHSExperiment(jpdf,no_samples).generate())
            else:
                print('sample type not implemented')
                return
            values=np.asarray([func(knot) for knot in knots])
            return (knots, values)
    elif no_samples==knots.shape[0]:
        return (knots, values)
    else:
        if ed_file != None:
            if no_samples<=knots_pool.shape[0]:
                knots=knots_pool[:no_samples,:]
                values=fvalues_pool[:no_samples]
                return knots, values
            else:
                knots, values=get_ed(func,no_samples,jpdf,sample_type,knots,values)
                return knots, values
        else:
            if sample_type == 'R':
                knots_to_comp=np.array(jpdf.getSample(no_samples-knots.shape[0]))
            elif sample_type == 'L':
                knots_to_comp=np.array(ot.LHSExperiment(jpdf,no_samples-knots.shape[0]).generate())
            else:
                print('sample type not implemented')
                return
            knots_all=np.zeros((no_samples,knots.shape[1]))
            knots_all[:knots.shape[0],:]=knots
            knots_all[knots.shape[0]:,:]=knots_to_comp
            knots=knots_all
            values_new=np.asarray([func(knot) for knot in knots_to_comp])
            values=np.append(values,values_new)
    return (knots, values)


def get_design_matrix(exp,nodes):
    """Returns the design matrix corresponding to the pc basis and the nodes
    exp: polynomial chaos basis
    nodes: nodes where basis has to be evaluated"""
    D = np.zeros((nodes.shape[0],len(exp)))
    for i in range(len(exp)):
        D[:,i]=np.asarray(exp[i](nodes.tolist())).reshape(nodes.shape[0])
    return D

def power_iteration(A, num_simulations, b_k=None):
    if b_k==None:
        b_k = np.random.rand(A.shape[0])
    for _ in range(num_simulations ):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    eigenvalue = np.matmul(np.matmul(b_k.T, A),b_k)/np.matmul(b_k.T, b_k)
    return eigenvalue


def compute_moments(beta_coeff):
    """Compute the stochastic moments MEAN and VARIANCE
    beta_coeff: chaos coefficients"""
    m=beta_coeff[0]
    v=np.sum(np.array(beta_coeff[1:])**2)
    return m,v


def compute_sobol_index(i,beta_coeff,multi_index_set,sobol_type='M'):
    """Compute the main and total Sobol indices.
    i: index of Sobol index to be computed, possible handles: int,list,array
    beta_coef: chaos coefficients
    multi_index: active index set wih multi-indices
    sobol_type: either Main order Total Sobol indices
                M - Main
                T - Total """
    m,v=compute_moments(beta_coeff)
    m_set=np.asarray(multi_index_set)
    if isinstance(i,list) or isinstance(i,np.ndarray):
        S=[]
        for item in i:
            if sobol_type=='M':
                # get columns that are zero excpet column with index i
                idx=np.where(~m_set[:,np.arange(m_set.shape[1])!=item].any(axis=1))[0][1:]
            else:
                idx=np.where(m_set[:,np.arange(m_set.shape[1])==item].any(axis=1))[0][:]

            if idx.shape==(0,):
                S.append(0.0)
            else:
                S.append(sum(beta_coeff[idx]**2)/v)
        return S
    else:
        if sobol_type=='M':
            # get columns that are zero excpet column with index i
            idx=np.where(~m_set[:,np.arange(m_set.shape[1])!=i].any(axis=1))[0][1:]
        else:
            idx=np.where(m_set[:,np.arange(m_set.shape[1])==i].any(axis=1))[0][:]

        if idx.shape==(0,):
            return 0.0
        else:
            return sum(beta_coeff[idx]**2)/v


def transform_multi_index_set(idx_set, enum_func):
    """Transforms a multi-index set to a linear set (integer)
    idx_set: multi-index set (e.g. [[0,0],[1,0]])
    enum_func: enumerate function"""
    idx_linear=[]
    for idx in idx_set:
        idx_linear.append(enum_func.inverse(idx))
    return idx_linear


class PCE_Surrogate():
    """Creates a PCE surrogate model"""
    def __init__(self, beta_coeff, idx_set, jpdf):
        self.beta_coeff=beta_coeff
        self.idx_set=idx_set
        self.jpdf=jpdf
        self.N=jpdf.getDimension()

        # get the distribution type of each random variable
        dist_types = []
        for i in range(self.N):
            dist_type = self.jpdf.getMarginal(i).getName()
            dist_types.append(dist_type)

        # create orthogonal univariate bases
        poly_collection = ot.PolynomialFamilyCollection(self.N)
        for i in range(self.N):
            pdf = jpdf.getDistributionCollection()[i]
            algo = ot.AdaptiveStieltjesAlgorithm(pdf)
            poly_collection[i] = ot.StandardDistributionPolynomialFactory(algo)

        # create multivariate basis
        multivariate_basis = ot.OrthogonalProductPolynomialFactory(
                              poly_collection, ot.EnumerateFunction(self.N))
        # get enumerate function (multi-index handling)
        enum_func = multivariate_basis.getEnumerateFunction()
        # get epansion
        self.expansion = multivariate_basis.getSubBasis(
                                   transform_multi_index_set(idx_set, enum_func)
                                                       )
        # create openturns surrogate model
        sur_model=ot.FunctionCollection()
        for i in range(len(self.expansion)):
            multi = str(beta_coeff[i])+'*x'
            help_function=ot.SymbolicFunction(['x'],[multi])
            sur_model.add(ot.ComposedFunction(help_function,self.expansion[i]))
        self.surrogate_model=np.sum(sur_model)

    def evaluate(self, X):
        if isinstance(X,np.ndarray):
            X=X.tolist()
        return self.surrogate_model(X)

    def get_expansion(self):
        return self.expansion

    def failure_probability(self, bound, M, upper_bound,seed=999):
        ot.RandomGenerator.SetSeed(seed)
        samples = np.array(self.jpdf.getSample(M))
        fvalues = self.evaluate(samples)
        if upper_bound:
            count = fvalues[np.where(fvalues < bound)].shape[0]
        else:
            count = fvalues[np.where(fvalues > bound)].shape[0]
        return count/(M*1.0)

    def plot_pdf(self,M, seed):
        ot.RandomGenerator.SetSeed(seed)
        samples = np.array(self.jpdf.getSample(M))
        fvalues = self.evaluate(samples)
        plt.figure()
        plt.hist(fvalues,100)

    def get_pdf(self,M,h):
        sample_ot=self.evaluate(np.asarray(self.jpdf.getSample(M)))
        sample=np.asarray(sample_ot).copy()
        #correct_pdf=np.where(sample > 1)
        ###sample[correct_pdf] = 1.0 # correct samples to 1 if samples are larger 1 (S-Parameter max 1!)
        #sample=np.delete(sample,correct_pdf)
        sample=ot.Sample(np.reshape(sample,sample.shape[0]),1)
        return ot.KernelMixture(ot.Epanechnikov(), [h], sample)

    def get_surrogate_model(self):
        return self.surrogate_model