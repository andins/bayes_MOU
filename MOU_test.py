#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:42:28 2017

@author: andrea
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from MOU import MOU_sim, matrix_norm_dist
from MOU_bayes import map_MOU


dt = 0.05
Ms = [5, 10, 50]
repetitions = 10

simil_lam1s_lam1 = np.zeros([len(Ms), repetitions])
simil_lam2s_lam2 = np.zeros([len(Ms), repetitions])
simil_lam1_lam2 = np.zeros(len(Ms))
simil_LAM1s_LAM1 = np.zeros([len(Ms), repetitions])
simil_LAM2s_LAM2 = np.zeros([len(Ms), repetitions])
simil_LAM1_LAM2 = np.zeros(len(Ms))
d_lam1s_lam1 = np.zeros([len(Ms), repetitions])
d_lam2s_lam2 = np.zeros([len(Ms), repetitions])
d_lam1_lam2 = np.zeros(len(Ms))
d_LAM1s_LAM1 = np.zeros([len(Ms), repetitions])
d_LAM2s_LAM2 = np.zeros([len(Ms), repetitions])
d_LAM1_LAM2 = np.zeros(len(Ms))


# for each number of dimensions
for i, m in enumerate(Ms):
    # define two different connectivity matrices corresponding to two generative models M1 and M2
    tau = 1
#    tmp1 = np.tri(m) * (np.random.randn(m,m) +1) * m/m  # gaussian
    tmp1 = np.random.choice([0,1], size=(m,m), p=[.8, .2]) * (np.random.randn(m,m)*10/m+5/m) # binary
    tmp1[np.eye(m, dtype=bool)] = 1/tau
#    tmp2 = np.tri(m).T * (np.random.randn(m,m) +1) * m/m  # gaussian
    tmp2 = np.random.choice([0,1], size=(m,m), p=[.8, .2]) * (np.random.randn(m,m)*10/m+5/m) # binary
    tmp2[np.eye(m, dtype=bool)] = 1/tau
    # a mask to filter out diagonal
    noDiag = np.logical_not(np.eye(m, dtype=bool))
    # simulate the models and estimate parameters for 50 repetitions
    LAM_star1 = np.zeros([m, m, repetitions])
    LAM_star2 = np.zeros([m, m, repetitions])
    lam_star1 = np.zeros([m, m, repetitions])
    lam_star2 = np.zeros([m, m, repetitions])
    C1 = np.zeros([m, m])
    C2 = np.zeros([m, m])
    C1s = np.zeros([m, m, repetitions])
    C2s = np.zeros([m, m, repetitions])
    mask_EC = np.logical_not(np.eye(m, dtype=bool))
    for r in range(repetitions):
        # simulation
        X1, lam1 = MOU_sim(M=m, duration=300, lam=tmp1)
        X2, lam2 = MOU_sim(M=m, duration=300, lam=tmp2)
#        X1 = X1[:, ::int(1/dt)]  # subsample to 1 time unit
#        X2 = X2[:, ::int(1/dt)]
        # estimation
        LAM1 = sp.linalg.expm(-lam1*dt)
        LAM_star1[:, :, r], SIGMA_star1 = map_MOU(X1)
        lam_star1[:, :, r] = - sp.linalg.logm(LAM_star1[:, :, r]) / dt
#        lam_star1[:, :, r] = MOU_estim_Lyapunov(X1.T, mask_EC)[0]
        C1s[:, :, r] = lam_star1[:, :, r].copy()
        C1s[np.eye(m, dtype=bool), r] = 0
        LAM2 = sp.linalg.expm(-lam2*dt)
        LAM_star2[:, :, r], SIGMA_star2 = map_MOU(X2)
        lam_star2[:, :, r] = - sp.linalg.logm(LAM_star2[:, :, r]) / dt
#        lam_star2[:, :, r] = MOU_estim_Lyapunov(X2.T, mask_EC)[0]
        C2s[:, :, r] = lam_star2[:, :, r].copy()
        C2s[np.eye(m, dtype=bool), r] = 0
        C1 = lam1.copy()
        C1[np.eye(m, dtype=bool)] = 0
        C2 = lam2.copy()
        C2[np.eye(m, dtype=bool)] = 0
        # difference of estimated lam and LAM for the two systems
        simil_lam1s_lam1[i, r] = pearsonr(C1s[:, :, r].flatten(), C1.flatten())[0]
        simil_lam2s_lam2[i, r] = pearsonr(C2s[:, :, r].flatten(), C2.flatten())[0]
        simil_LAM1s_LAM1[i, r] = pearsonr(LAM_star1[noDiag, r], LAM1[noDiag])[0]
        simil_LAM2s_LAM2[i, r] = pearsonr(LAM_star2[noDiag, r], LAM2[noDiag])[0]
        d_lam1s_lam1[i, r] = matrix_norm_dist(C1s[:, :, r], C1)
        d_lam2s_lam2[i, r] = matrix_norm_dist(C2s[:, :, r], C2)
        d_LAM1s_LAM1[i, r] = matrix_norm_dist(LAM_star1[noDiag, r], LAM1[noDiag])
        d_LAM2s_LAM2[i, r] = matrix_norm_dist(LAM_star2[noDiag, r], LAM2[noDiag])
    simil_lam1_lam2[i] = pearsonr(C1.flatten(), C2.flatten())[0]
    simil_LAM1_LAM2[i] = pearsonr(LAM1[noDiag], LAM2[noDiag])[0]
    d_lam1_lam2[i] = matrix_norm_dist(C1, C2)
    d_LAM1_LAM2[i] = matrix_norm_dist(LAM1[noDiag], LAM2[noDiag])
    


plt.figure()
plt.subplot(221)
plt.errorbar(Ms, simil_lam1s_lam1.mean(axis=1), simil_lam1s_lam1.std(axis=1),
             label= r'$\rho(\lambda_{1}^*,\lambda_{1})$')
plt.errorbar(Ms, simil_lam2s_lam2.mean(axis=1), simil_lam2s_lam2.std(axis=1),
             label= r'$\rho(\lambda_{2}^*,\lambda_{2})$')
plt.plot(Ms, simil_lam1_lam2,
         label = r'$\rho(\lambda_{1},\lambda_{2})$')
plt.legend()
plt.xlabel('# dimensions')
plt.ylabel('similarity')
plt.subplot(222)
plt.errorbar(Ms, simil_LAM1s_LAM1.mean(axis=1), simil_LAM1s_LAM1.std(axis=1),
             label = r'$\rho(\Lambda_{1}^*,\Lambda_{1})$')
plt.errorbar(Ms, simil_LAM2s_LAM2.mean(axis=1), simil_LAM2s_LAM2.std(axis=1),
             label = r'$\rho(\Lambda_{2}^*,\Lambda_{2})$')
plt.plot(Ms, simil_LAM1_LAM2,
         label = r'$\rho(\Lambda_{1},\Lambda_{2})$')
plt.legend()
plt.xlabel('# dimensions')
plt.subplot(223)
plt.errorbar(Ms, d_lam1s_lam1.mean(axis=1), d_lam1s_lam1.std(axis=1),
             label= r'$D(\lambda_{1}^*,\lambda_{1})$')
plt.errorbar(Ms, d_lam2s_lam2.mean(axis=1), d_lam2s_lam2.std(axis=1),
             label= r'$D(\lambda_{2}^*,\lambda_{2})$')
plt.plot(Ms, d_lam1_lam2,
         label = r'$D(\lambda_{1},\lambda_{2})$')
plt.legend()
plt.xlabel('# dimensions')
plt.ylabel('matrix distance')
plt.subplot(224)
plt.errorbar(Ms, d_LAM1s_LAM1.mean(axis=1), d_LAM1s_LAM1.std(axis=1),
             label = r'$D(\Lambda_{1}^*,\Lambda_{1})$')
plt.errorbar(Ms, d_LAM2s_LAM2.mean(axis=1), d_LAM2s_LAM2.std(axis=1),
             label = r'$D(\Lambda_{2}^*,\Lambda_{2})$')
plt.plot(Ms, d_LAM1_LAM2,
         label = r'$D(\Lambda_{1},\Lambda_{2})$')
plt.legend()
plt.xlabel('# dimensions')
           