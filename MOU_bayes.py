#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:31:18 2017

@author: andrea
"""

import scipy as sp
import numpy as np
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt


def MOU_sample(M=2, duration=1000, lam='rand'):
    # this sample from the distribution of x_t+1 | x_t = N(mu, SIGMA)
    if lam == 'rand':
        tau = 1
        tmp = np.random.randn(M, M) * 1/M
        tmp[np.eye(M, dtype=bool)] = 1/tau
        # reduce rank of lam matrix
        U, S, V = np.linalg.svd(tmp)
        k = 5  # the rank
        lam = np.dot(U[:, :k], np.dot(np.diag(S)[:k, :k], V[:k, :]))
    dt = 0.01
    steps = int(duration/dt)
    ini = np.random.randn(M)
    LAM = sp.linalg.expm(-lam*dt)
    SIG = np.eye(M)  # we fix SIGMA here
    SIG_lt = np.linalg.cholesky(SIG, lower=True)
    X = np.zeros([M, steps])
    X[:, 0] = ini
    for i in range(1, steps):
        mu = np.dot(LAM, X[:, i-1:i])
        X[:, i:i+1] = np.dot(SIG_lt, np.random.randn(M, 1)) + mu
    return X, LAM


def map_MOU(X, tau=1, verbose=0):
    """
    X has variables on the rows and time on the columns
    tau is the time between time samples (not the time constant of the system!)
    twoD is the matrix of diffusion coefficients (SIGMA in Matt paper)
    """

    N = X.shape[1]  # number of time steps
    M = X.shape[0]  # dimensionality of the system
    X -= np.outer(X.mean(axis=1), np.ones(N))  # center the time series
    T1 = [np.dot(X[:, i:i+1], X[:, i:i+1].T) for i in range(1, N)]
    T1 = np.sum(T1, axis=0)
    T2 = [np.dot(X[:, i+1:i+2], X[:, i:i+1].T) for i in range(0, N-1)]
    T2 = np.sum(T2, axis=0)
    T3 = [np.dot(X[:, i:i+1], X[:, i:i+1].T) for i in range(0, N-1)]
    T3 = np.sum(T3, axis=0)
#    T4 = np.dot(X[:, 0:1], X[:, 0:1].T)  # this is actually not used
    LAM_star = np.dot(T2, np.linalg.inv(T3))
    # SIGMA_star can be useful for generating samples using
    # x_(n+1) = dot(LAM, x_n) + dot(sqrt(SIGMA), Xi_n)
    SIGMA_star = (T1-np.dot(np.dot(T2, np.linalg.inv(T3)), T2.T)) / N
    lam_star = -logm(LAM_star) / tau
    if not np.all(np.isclose(expm(lam_star*tau), LAM_star, rtol=1e-01)):
        print("Warning: logarithm!")
    if np.any(np.iscomplex(lam_star)):
        lam_star = np.real(lam_star)
        print("Warning: complex values in lam_star; casting to real!")
    # TODO: implement bayes I for c_star
    c_star_bayesII = T3 / N
    twoD = np.dot(lam_star, c_star_bayesII) + np.dot(lam_star, c_star_bayesII).T
    C = -lam_star.copy()  # connectivity
    tau_x = -C.diagonal()
    C[np.eye(M, dtype=bool)] = 0
    if verbose > 0:
        J = -np.eye(M)/tau_x + C
        FC_t = sp.linalg.solve_lyapunov(J, -twoD)
        print("Correlation between theoretical and empirical FC: ",
              sp.stats.pearsonr(sp.cov(X).flatten(), FC_t.flatten()))
    if verbose > 1:
        mask_offDiag = [np.logical_not(np.eye(M, dtype=bool))]
        plt.figure()
        plt.scatter(sp.cov(X)[mask_offDiag], FC_t[mask_offDiag], color='b')
        plt.scatter(sp.cov(X).diagonal(), FC_t.diagonal(), color='c')
        plt.xlabel('FC0 emp')
        plt.ylabel('FC0 model')
    return C, twoD, tau_x
