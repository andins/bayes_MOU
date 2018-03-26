#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:35:54 2018

@author: andrea
"""
from MOU import MOU
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from make_rnd_connectivity import make_rnd_connectivity
from os import listdir

Ns = [10, 20, 50, 100]
repetitions = 100
file_name = "comparison_T500.npy"
if file_name in listdir():  # if file is in the directory load variables
    file = np.load(file_name)
    gen = file[0]
    lyap = file[1]
    mome = file[2]
else:  # else create the variables
    gen = dict()
    lyap = dict()
    mome = dict()
for N in Ns:
    if not(N in lyap.keys()):  # if the N has already been done skips
        gen[N] = dict()
        lyap[N] = dict()
        mome[N] = dict()
    for r in range(repetitions):
        if not(r in lyap[N].keys()):  # if the repetition has already been done skips
            try:  # if there is any error saves the results
                C = make_rnd_connectivity(N=N, density=0.2, connectivity_strength=0.5)
                generator = MOU(n_nodes=N, tau_x=1, mu=0.0, Sigma=None, C=C)
                ts = generator.simulate(T=500)
                gen[N][r] = generator
                model_l = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_l.fit(X=ts, method="lyapunov")
                lyap[N][r] = model_l
                model_m = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_m.fit(X=ts, method="moments")
                mome[N][r] = model_m
            except:  # here is the save
                np.save(file_name, (lyap, mome))
np.save(file_name, (gen, lyap, mome))

#%%
rCl = np.zeros([len(Ns), repetitions])
rCm = np.zeros([len(Ns), repetitions])
for i, N in enumerate(Ns):
    for r in range(repetitions):
        rCl[i, r] = pearsonr(lyap[N][r].C.flatten(), gen[N][r].C.flatten())[0]
        rCm[i, r] = pearsonr(mome[N][r].C.flatten(), gen[N][r].C.flatten())[0]
plt.figure()
plt.errorbar(Ns, rCl.mean(axis=1), rCl.std(axis=1))
plt.errorbar(Ns, rCm.mean(axis=1), rCm.std(axis=1))
